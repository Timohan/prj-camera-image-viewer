#include "camera_image.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "global_log.h"

#define MM_TO_OPENGL_UNIT           1000.0
#define MM_PER_METER                1000.0
// distances at least this far (mm) are the sky, the creator saves the sky as far as the sun
#define SKY_DISTANCE_MM             1.0e12
// sky and the predicted pixels above the horizon are put this far (mm)
#define FAR_DISTANCE_MM             500000.0
// without the surfaces a triangle is not made, if its largest distance is this many times its smallest distance
// (edge of an object)
#define DISTANCE_JUMP_RATIO         1.15
// camera and lens information block after the distances in the data file
#define DATA_CAMERA_INFO_MAGIC      "CAMI"
#define DATA_CAMERA_INFO_VERSION    2
#define DATA_CAMERA_INFO_VALUES     36
static_assert(DATA_CAMERA_INFO_VALUES == VisionTransformer::CAMERA_VALUE_COUNT, "camera values of the model");
// surface block after the camera information
#define DATA_SURFACE_MAGIC          "SURF"
#define DATA_SURFACE_VERSION        1
// what the surface numbers are, after the surfaces
#define DATA_OBJECTS_MAGIC          "OBJS"
#define DATA_OBJECTS_VERSION        1
// image colors are mixed this much with the color of the class of the object (object textures)
#define OBJECT_TINT                 0.55
// undistortPoints iterations and accuracy in pixels like in prj-camera-image-creator
#define UNDISTORT_MAX_ITERATIONS    200
#define UNDISTORT_EPSILON_PIXELS    0.001
// values of a vertex: x, y, z, s, t
#define VERTEX_VALUES               5

// first class of the people (man, woman, child) like in the learner (object_errors.h)
#define OBJECT_FIRST_PERSON_CLASS   5
// classes of the model (pixel_label.h), the class of the data file is one larger (0 is the sky there)
static const char *const OBJECT_NAMES[OBJECT_CLASS_COUNT] = { "ground", "other object", "sedan", "SUV", "sports car",
                                                              "man", "woman", "child" };
// RGB tint of the classes, the ground is not tinted
static const int OBJECT_COLORS[OBJECT_CLASS_COUNT][3] = {
    { -1, -1, -1 },         // ground
    { 60, 120, 255 },       // other object (cube, ball, cylinder, cone): blue
    { 255, 220, 0 },        // sedan: yellow
    { 255, 130, 0 },        // SUV: orange
    { 255, 30, 30 },        // sports car: red
    { 40, 220, 60 },        // man: green
    { 230, 60, 230 },       // woman: magenta
    { 0, 230, 230 }         // child: cyan
};
static_assert(OBJECT_CLASS_COUNT == 8, "names and colors of the object classes");

// RGB color of an object (id): the hues of the golden angle, so the neighbouring ids have different colors
static void idColor(int id, int *rgb)
{
    const double hue = std::fmod(static_cast<double>(id)*0.618033988749895, 1.0)*6.0;
    const int sector = static_cast<int>(hue);
    const double f = hue - sector;
    const double value = 255.0, low = 0.25*value, high = value;
    const double rising = low + (high - low)*f, falling = high - (high - low)*f;
    const double colors[6][3] = { { high, rising, low }, { falling, high, low }, { low, high, rising },
                                  { low, falling, high }, { rising, low, high }, { high, low, falling } };
    for (int c=0;c<3;c++) {
        rgb[c] = static_cast<int>(colors[sector%6][c]);
    }
}

CameraImage::CameraImage() :
    m_cameraNumber(0),
    m_position{ 0.0, 0.0, 0.0 },
    m_direction{ 0.0, 0.0, 1.0 },
    m_right{ 1.0, 0.0, 0.0 },
    m_up{ 0.0, 1.0, 0.0 },
    m_texture(0),
    m_correctObjectTexture(0),
    m_predictedObjectTexture(0),
    m_correctIdTexture(0),
    m_predictedIdTexture(0)
{
}

CameraImage::~CameraImage()
{
    deleteMesh(&m_correctMesh);
    deleteMesh(&m_predictedMesh);
    deleteMesh(&m_correctFillMesh);
    deleteMesh(&m_predictedFillMesh);
    for (GLuint texture : { m_texture, m_correctObjectTexture, m_predictedObjectTexture, m_correctIdTexture,
                            m_predictedIdTexture }) {
        if (texture) {
            glDeleteTextures(1, &texture);
        }
    }
}

/*!
 * \brief CameraImage::load
 * loads the png file and the data file (if it is there) and calculates the ray of every pixel
 * \param filename full path of the png and data files without the extension
 * \param camera camera and lens of the image from the model
 * \return true on success
 */
bool CameraImage::load(const std::string &filename, const VisionTransformer::Camera &camera)
{
    cv::Mat image = cv::imread(filename + ".png", cv::IMREAD_COLOR);
    if (image.empty()) {
        TG_ERROR_LOG("Cannot read", filename + ".png");
        return false;
    }
    cv::cvtColor(image, m_image, cv::COLOR_BGR2RGB);
    if (static_cast<int>(camera.width) != m_image.cols || static_cast<int>(camera.height) != m_image.rows) {
        TG_ERROR_LOG(filename + ".png", "is not of the size of camera", camera.number, "of the model");
        return false;
    }
    setCamera(camera);
    // the correct distances are only for comparison
    std::error_code error;
    if (!std::filesystem::exists(filename + ".dat", error)) {
        return true;
    }
    return readData(filename + ".dat");
}

/*!
 * \brief CameraImage::hasCorrectDistances
 * \return true if the data file had the correct distances
 */
bool CameraImage::hasCorrectDistances() const
{
    return !m_correctDistances.empty();
}

/*!
 * \brief CameraImage::hasCorrectObjects
 * \return true if the data file had the objects of the surface numbers
 */
bool CameraImage::hasCorrectObjects() const
{
    return !m_correctObjects.empty() && !m_correctSurfaces.empty();
}

/*!
 * \brief CameraImage::hasPredictedObjects
 * \return true if the model has the object output
 */
bool CameraImage::hasPredictedObjects() const
{
    return !m_predictedObjects.empty();
}

/*!
 * \brief CameraImage::readData
 * reads the correct distances of the data file of prj-camera-image-creator: uint32_t width, height, double distance (mm)
 * of every pixel, char[4] "CAMI", uint32_t version, camera number, distortion model and 36 doubles (the camera and
 * lens are from the model, only the camera number is checked), then optionally char[4] "SURF", uint32_t version,
 * uint32_t surface number of every pixel and the normals: neighbouring pixels of the same surface number are connected
 * (0 is the sky), then optionally char[4] "OBJS", uint32_t version, uint32_t count and of every surface uint32_t
 * surface number, uint32_t class and doubles center x/y/z (mm), sizes x/y/z (mm), yaw, pitch, roll (degrees) of the box
 * of its object.
 * \return true on success
 */
bool CameraImage::readData(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    uint32_t width = 0, height = 0;
    file.read(reinterpret_cast<char *>(&width), sizeof(width));
    file.read(reinterpret_cast<char *>(&height), sizeof(height));
    if (!file || static_cast<int>(width) != m_image.cols || static_cast<int>(height) != m_image.rows) {
        TG_ERROR_LOG(filename, "does not match the image");
        return false;
    }
    m_correctDistances.resize(static_cast<size_t>(width)*height);
    file.read(reinterpret_cast<char *>(m_correctDistances.data()),
              static_cast<std::streamsize>(m_correctDistances.size()*sizeof(double)));
    char magic[4];
    uint32_t version, cameraNumber, distortionModel;
    file.read(magic, sizeof(magic));
    file.read(reinterpret_cast<char *>(&version), sizeof(version));
    file.read(reinterpret_cast<char *>(&cameraNumber), sizeof(cameraNumber));
    file.read(reinterpret_cast<char *>(&distortionModel), sizeof(distortionModel));
    file.seekg(static_cast<std::streamoff>(DATA_CAMERA_INFO_VALUES*sizeof(double)), std::ios::cur);
    if (!file || memcmp(magic, DATA_CAMERA_INFO_MAGIC, sizeof(magic)) != 0 || version != DATA_CAMERA_INFO_VERSION) {
        TG_ERROR_LOG(filename, "has no camera information of version", DATA_CAMERA_INFO_VERSION);
        return false;
    }
    if (static_cast<int>(cameraNumber) != m_cameraNumber) {
        TG_ERROR_LOG(filename, "is camera", cameraNumber, "but the model camera is", m_cameraNumber);
        return false;
    }
    // older data files have no surfaces
    m_correctConnections.clear();
    m_correctSurfaces.clear();
    m_correctObjects.clear();
    uint32_t surfaceVersion = 0;
    file.read(magic, sizeof(magic));
    file.read(reinterpret_cast<char *>(&surfaceVersion), sizeof(surfaceVersion));
    if (file && memcmp(magic, DATA_SURFACE_MAGIC, sizeof(magic)) == 0 && surfaceVersion == DATA_SURFACE_VERSION) {
        std::vector<uint32_t> surfaces(static_cast<size_t>(width)*height);
        file.read(reinterpret_cast<char *>(surfaces.data()), static_cast<std::streamsize>(surfaces.size()*sizeof(uint32_t)));
        if (file) {
            m_correctSurfaces = surfaces;
            m_correctConnections.assign(surfaces.size(), 0);
            for (uint32_t y=0;y<height;y++) {
                for (uint32_t x=0;x<width;x++) {
                    const size_t i = static_cast<size_t>(y)*width + x;
                    if (surfaces[i] == 0) {
                        continue;
                    }
                    if (x + 1 < width && surfaces[i + 1] == surfaces[i]) {
                        m_correctConnections[i] |= ConnectedRight;
                    }
                    if (y + 1 < height && surfaces[i + width] == surfaces[i]) {
                        m_correctConnections[i] |= ConnectedDown;
                    }
                }
            }
            // normals of the pixels, then the objects of the surfaces (older data files have no objects)
            file.seekg(static_cast<std::streamoff>(surfaces.size()*3*sizeof(float)), std::ios::cur);
            uint32_t objectsVersion = 0, count = 0;
            file.read(magic, sizeof(magic));
            file.read(reinterpret_cast<char *>(&objectsVersion), sizeof(objectsVersion));
            file.read(reinterpret_cast<char *>(&count), sizeof(count));
            if (file && memcmp(magic, DATA_OBJECTS_MAGIC, sizeof(magic)) == 0 && objectsVersion == DATA_OBJECTS_VERSION) {
                std::vector<CorrectObject> objects;
                for (uint32_t i=0;i<count && file;i++) {
                    uint32_t surface = 0;
                    CorrectObject object;
                    double values[9];
                    file.read(reinterpret_cast<char *>(&surface), sizeof(surface));
                    file.read(reinterpret_cast<char *>(&object.objectClass), sizeof(object.objectClass));
                    file.read(reinterpret_cast<char *>(values), sizeof(values));
                    if (!file || surface > surfaces.size()) {
                        break;
                    }
                    for (int k=0;k<3;k++) {
                        object.center[k] = values[k];
                        object.size[k] = values[3 + k];
                    }
                    object.yaw = values[6];
                    if (objects.size() <= surface) {
                        objects.resize(static_cast<size_t>(surface) + 1);
                    }
                    objects[surface] = object;
                }
                if (file) {
                    m_correctObjects = std::move(objects);
                }
            }
        }
    }
    return true;
}

/*!
 * \brief CameraImage::setCamera
 * calculates the ray of every pixel through the lens
 * \param camera camera of the model: position, direction, up, ..., fx, fy, cx, cy (15-18), k1-k6 (19-24), p1, p2 (25, 26),
 * fisheye k1-k4 (27-30) like the camera information of the data file
 */
void CameraImage::setCamera(const VisionTransformer::Camera &camera)
{
    const uint32_t distortionModel = camera.distortionModel;
    const double *values = camera.values;
    const uint32_t width = static_cast<uint32_t>(m_image.cols);
    const uint32_t height = static_cast<uint32_t>(m_image.rows);
    m_cameraNumber = static_cast<int>(camera.number);
    for (int i=0;i<3;i++) {
        m_position[i] = values[i];
        m_direction[i] = values[3 + i];
    }
    const VECTOR3 forward = { static_cast<float>(values[3]), static_cast<float>(values[4]), static_cast<float>(values[5]) };
    const VECTOR3 up = { static_cast<float>(values[6]), static_cast<float>(values[7]), static_cast<float>(values[8]) };
    // left-handed world (x right, y up, z forward): right = up x forward
    const VECTOR3 right = Math3D::Vec3Normalize(Math3D::Vec3Cross(up, forward));
    // axes of the boxes of the model like in the learner (CameraPosition::toCamera): the up is made perpendicular to
    // the direction, up = forward x right
    {
        const double *f = values + 3;
        const double *u = values + 6;
        double r[3] = { u[1]*f[2] - u[2]*f[1], u[2]*f[0] - u[0]*f[2], u[0]*f[1] - u[1]*f[0] };
        const double length = std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
        for (int i=0;i<3;i++) {
            m_right[i] = length > 0.0 ? r[i]/length : 0.0;
        }
        m_up[0] = f[1]*m_right[2] - f[2]*m_right[1];
        m_up[1] = f[2]*m_right[0] - f[0]*m_right[2];
        m_up[2] = f[0]*m_right[1] - f[1]*m_right[0];
    }
    const cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << values[15], 0.0, values[17],
                                                            0.0, values[16], values[18],
                                                            0.0, 0.0, 1.0);

    // pixel centers to normalized image coordinates without the lens distortion
    std::vector<cv::Point2d> pixels, normalized;
    for (uint32_t y=0;y<height;y++) {
        for (uint32_t x=0;x<width;x++) {
            pixels.emplace_back(static_cast<double>(x), static_cast<double>(y));
        }
    }
    if (distortionModel == 1) {
        const cv::Mat distortion = (cv::Mat_<double>(1, 4) << values[27], values[28], values[29], values[30]);
        cv::fisheye::undistortPoints(pixels, normalized, cameraMatrix, distortion);
    } else {
        const cv::Mat distortion = (cv::Mat_<double>(1, 8) << values[19], values[20], values[25], values[26],
                                                              values[21], values[22], values[23], values[24]);
        cv::undistortPoints(pixels, normalized, cameraMatrix, distortion, cv::noArray(), cv::noArray(),
                            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, UNDISTORT_MAX_ITERATIONS,
                                             UNDISTORT_EPSILON_PIXELS));
    }
    // image y goes down, the up vector goes up
    m_rays.resize(normalized.size());
    for (size_t i=0;i<normalized.size();i++) {
        const float nx = static_cast<float>(normalized[i].x);
        const float ny = static_cast<float>(normalized[i].y);
        m_rays[i] = Math3D::Vec3Normalize({ forward.x + nx*right.x - ny*up.x, forward.y + nx*right.y - ny*up.y,
                                            forward.z + nx*right.z - ny*up.z });
    }
}

/*!
 * \brief CameraImage::setPrediction
 * \param distances predicted distance (mm) of every pixel
 * \param sky 1 if the model predicted that the pixel is sky
 * \param connections Connection of every pixel, empty if the model has no surface output
 * \param objects object of every pixel, empty if the model has no object output
 */
void CameraImage::setPrediction(const std::vector<double> &distances, const std::vector<uint8_t> &sky,
                                const std::vector<uint8_t> &connections, const std::vector<ObjectPixel> &objects)
{
    m_predictedDistances = distances;
    m_predictedSky = sky;
    m_predictedConnections = connections;
    m_predictedObjects = objects;
}

/*!
 * \brief CameraImage::createTexture
 * \param image RGB image of the size of the camera image
 * \return texture, opengl context must be current
 */
GLuint CameraImage::createTexture(const cv::Mat &image) const
{
    GLuint texture = 0;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    // rows of the image are 3*width bytes, not 4 byte aligned
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.cols, image.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, image.data);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
    return texture;
}

/*!
 * \brief CameraImage::tintImage
 * \param correctObjects true: the objects of the data file, false: the objects predicted by the model
 * \param ids true: every object in its own color (idColor() of getObjectId()), false: the color of the class
 * \return the image mixed with the color of the class of the object of every pixel (OBJECT_COLORS) or of the object,
 * the sky and the ground keep their colors
 */
cv::Mat CameraImage::tintImage(bool correctObjects, bool ids) const
{
    cv::Mat image = m_image.clone();
    for (int y=0;y<image.rows;y++) {
        cv::Vec3b *row = image.ptr<cv::Vec3b>(y);
        for (int x=0;x<image.cols;x++) {
            const size_t i = static_cast<size_t>(y)*image.cols + x;
            int objectClass = -1;
            if (correctObjects) {
                const uint32_t surface = m_correctSurfaces[i];
                if (surface < m_correctObjects.size() && m_correctObjects[surface].objectClass > 0
                    && m_correctObjects[surface].objectClass <= static_cast<uint32_t>(OBJECT_CLASS_COUNT)) {
                    objectClass = static_cast<int>(m_correctObjects[surface].objectClass) - 1;
                }
            } else if (m_predictedObjects[i].objectClass < OBJECT_CLASS_COUNT) {
                objectClass = m_predictedObjects[i].objectClass;
            }
            if (objectClass < 0 || OBJECT_COLORS[objectClass][0] < 0) {
                continue;
            }
            int color[3] = { OBJECT_COLORS[objectClass][0], OBJECT_COLORS[objectClass][1], OBJECT_COLORS[objectClass][2] };
            if (ids) {
                const int id = getObjectId(x, y, correctObjects);
                if (id < 0) {
                    continue;
                }
                idColor(id, color);
            }
            for (int c=0;c<3;c++) {
                row[x][c] = static_cast<uint8_t>(std::lround((1.0 - OBJECT_TINT)*row[x][c] + OBJECT_TINT*color[c]));
            }
        }
    }
    return image;
}

/*!
 * \brief CameraImage::createMeshes
 * creates the texture of the image and the meshes of the correct and the predicted distances,
 * opengl context must be current
 * \return 0 on success
 */
int CameraImage::createMeshes(GLint vertexAttribute, GLint textureCoordinateAttribute)
{
    m_texture = createTexture(m_image);
    if (hasCorrectObjects()) {
        m_correctObjectTexture = createTexture(tintImage(true, false));
        m_correctIdTexture = createTexture(tintImage(true, true));
    }
    if (hasPredictedObjects()) {
        m_predictedObjectTexture = createTexture(tintImage(false, false));
    }
    if (!m_predictedObjectIds.empty()) {
        m_predictedIdTexture = createTexture(tintImage(false, true));
    }
    // without the data file only the predicted distances are known
    if (hasCorrectDistances()) {
        // sky of the data file is very far or 0 (pixel sees nothing)
        std::vector<uint8_t> correctSky(m_correctDistances.size());
        for (size_t i=0;i<m_correctDistances.size();i++) {
            correctSky[i] = m_correctDistances[i] <= 0.0 || m_correctDistances[i] >= SKY_DISTANCE_MM ? 1 : 0;
        }
        if (createMesh(m_correctDistances, correctSky, m_correctConnections, vertexAttribute, textureCoordinateAttribute,
                       &m_correctMesh, &m_correctFillMesh) != 0) {
            return 1;
        }
    }
    return createMesh(m_predictedDistances, m_predictedSky, m_predictedConnections, vertexAttribute, textureCoordinateAttribute,
                      &m_predictedMesh, &m_predictedFillMesh);
}

/*!
 * \brief CameraImage::createMesh
 * \param distances distance (mm) of every pixel
 * \param sky 1 if the pixel is sky, it is put far away
 * \param connections Connection of every pixel: a triangle is made if its pixels are connected to each other,
 * empty: a triangle is made if its distances have no jump. The sky is connected only to the sky (sky dome).
 * \param mesh returns the triangles of the connected pixels
 * \param fillMesh returns the other triangles, every vertex has the color of the farthest pixel of the triangle
 * \return 0 on success
 */
int CameraImage::createMesh(const std::vector<double> &distances, const std::vector<uint8_t> &sky,
                            const std::vector<uint8_t> &connections, GLint vertexAttribute,
                            GLint textureCoordinateAttribute, Mesh *mesh, Mesh *fillMesh)
{
    const int width = m_image.cols;
    const int height = m_image.rows;
    if (distances.size() != static_cast<size_t>(width)*height || sky.size() != distances.size()) {
        TG_ERROR_LOG("Camera", m_cameraNumber, "has no distances");
        return 1;
    }
    // distance of every pixel, sky is far away
    std::vector<double> distance(distances.size());
    for (size_t i=0;i<distances.size();i++) {
        distance[i] = sky[i] ? FAR_DISTANCE_MM : distances[i];
    }
    std::vector<float> vertices, fillVertices;
    // position of pixel x, y and the texture coordinates of the center of pixel textureX, textureY
    auto addVertex = [&](std::vector<float> &list, int x, int y, int textureX, int textureY) {
        const size_t i = static_cast<size_t>(y)*width + x;
        const VECTOR3 &ray = m_rays[i];
        list.push_back(static_cast<float>((m_position[0] + distance[i]*ray.x)/MM_TO_OPENGL_UNIT));
        list.push_back(static_cast<float>((m_position[1] + distance[i]*ray.y)/MM_TO_OPENGL_UNIT));
        list.push_back(static_cast<float>((m_position[2] + distance[i]*ray.z)/MM_TO_OPENGL_UNIT));
        // texture coordinates of the pixel center
        list.push_back((static_cast<float>(textureX) + 0.5f)/static_cast<float>(width));
        list.push_back((static_cast<float>(textureY) + 0.5f)/static_cast<float>(height));
    };
    auto addTriangle = [&](bool isConnected, int x0, int y0, int x1, int y1, int x2, int y2) {
        if (isConnected) {
            addVertex(vertices, x0, y0, x0, y0);
            addVertex(vertices, x1, y1, x1, y1);
            addVertex(vertices, x2, y2, x2, y2);
            return;
        }
        // the farthest pixel is the background behind the edge
        const int xs[3] = { x0, x1, x2 };
        const int ys[3] = { y0, y1, y2 };
        int far = 0;
        for (int k=1;k<3;k++) {
            if (distance[static_cast<size_t>(ys[k])*width + xs[k]] > distance[static_cast<size_t>(ys[far])*width + xs[far]]) {
                far = k;
            }
        }
        for (int k=0;k<3;k++) {
            addVertex(fillVertices, xs[k], ys[k], xs[far], ys[far]);
        }
    };
    const bool surfaces = !connections.empty();
    auto connection = [&](int x, int y, uint8_t direction) {
        return (connections[static_cast<size_t>(y)*width + x] & direction) != 0;
    };
    // surfaceConnected: the pixels of the triangle are connected to each other (only used with the connections)
    auto connected = [&](int x0, int y0, int x1, int y1, int x2, int y2, bool surfaceConnected) {
        const size_t i0 = static_cast<size_t>(y0)*width + x0;
        const size_t i1 = static_cast<size_t>(y1)*width + x1;
        const size_t i2 = static_cast<size_t>(y2)*width + x2;
        if (surfaces) {
            if (sky[i0] || sky[i1] || sky[i2]) {
                return sky[i0] && sky[i1] && sky[i2];
            }
            return surfaceConnected;
        }
        const double a = distance[i0];
        const double b = distance[i1];
        const double c = distance[i2];
        return std::max(a, std::max(b, c)) <= DISTANCE_JUMP_RATIO*std::min(a, std::min(b, c));
    };
    // two triangles between every 2x2 pixels: the top left pixel with its right and lower neighbour,
    // and the bottom right pixel with its upper and left neighbour
    for (int y=0;y+1<height;y++) {
        for (int x=0;x+1<width;x++) {
            const bool first = connected(x, y, x + 1, y, x, y + 1,
                                         surfaces && connection(x, y, ConnectedRight) && connection(x, y, ConnectedDown));
            const bool second = connected(x + 1, y, x + 1, y + 1, x, y + 1,
                                          surfaces && connection(x + 1, y, ConnectedDown) && connection(x, y + 1, ConnectedRight));
            addTriangle(first, x, y, x + 1, y, x, y + 1);
            addTriangle(second, x + 1, y, x + 1, y + 1, x, y + 1);
        }
    }
    uploadMesh(vertices, vertexAttribute, textureCoordinateAttribute, mesh);
    uploadMesh(fillVertices, vertexAttribute, textureCoordinateAttribute, fillMesh);
    return 0;
}

/*!
 * \brief CameraImage::uploadMesh
 * \param vertices VERTEX_VALUES values of every vertex, every 3 vertices are a triangle
 */
void CameraImage::uploadMesh(const std::vector<float> &vertices, GLint vertexAttribute, GLint textureCoordinateAttribute,
                             Mesh *mesh)
{
    mesh->vertexCount = static_cast<GLsizei>(vertices.size()/VERTEX_VALUES);
    glGenVertexArrays(1, &mesh->vertexArray);
    glGenBuffers(1, &mesh->vertexBuffer);
    glBindVertexArray(mesh->vertexArray);
    glBindBuffer(GL_ARRAY_BUFFER, mesh->vertexBuffer);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size()*sizeof(float)), vertices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(static_cast<GLuint>(vertexAttribute), 3, GL_FLOAT, GL_FALSE, VERTEX_VALUES*sizeof(float),
                          reinterpret_cast<void *>(0));
    glEnableVertexAttribArray(static_cast<GLuint>(vertexAttribute));
    glVertexAttribPointer(static_cast<GLuint>(textureCoordinateAttribute), 2, GL_FLOAT, GL_FALSE, VERTEX_VALUES*sizeof(float),
                          reinterpret_cast<void *>(3*sizeof(float)));
    glEnableVertexAttribArray(static_cast<GLuint>(textureCoordinateAttribute));
    glBindVertexArray(0);
}

/*!
 * \brief CameraImage::render
 * \param correctDistances true: mesh of the correct distances, false: mesh of the predicted distances
 * \param fillMesh true: the fill mesh (triangles over the edges)
 * \param objectsMode the image tinted by the classes or the ids of the objects (the correct ones with the correct
 * distances), the image itself if those objects are not known
 */
void CameraImage::render(bool correctDistances, bool fillMesh, ObjectsMode objectsMode) const
{
    const Mesh &mesh = fillMesh ? (correctDistances ? m_correctFillMesh : m_predictedFillMesh)
                                : (correctDistances ? m_correctMesh : m_predictedMesh);
    GLuint texture = m_texture;
    if (objectsMode == ObjectsClasses) {
        texture = correctDistances ? m_correctObjectTexture : m_predictedObjectTexture;
    } else if (objectsMode == ObjectsIds) {
        texture = correctDistances ? m_correctIdTexture : m_predictedIdTexture;
    }
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture ? texture : m_texture);
    glBindVertexArray(mesh.vertexArray);
    glDrawArrays(GL_TRIANGLES, 0, mesh.vertexCount);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

/*!
 * \brief CameraImage::getDistanceText
 * \param x, y pixel of the image nearest to the point
 * \param correctDistances true: mesh of the correct distances of the data file, false: of the predicted distances
 * \param point point on the mesh (opengl units), between the pixels of its triangle
 * \return distance of the point from the camera in mm, "Sky" if the pixel is sky, or "N/A" if the pixel sees nothing
 * (correct distance 0). Near the horizon one pixel can be tens of meters long, so the point is used instead of the
 * distance of the pixel.
 */
std::string CameraImage::getDistanceText(int x, int y, bool correctDistances, const VECTOR3 &point) const
{
    if (x < 0 || y < 0 || x >= m_image.cols || y >= m_image.rows) {
        return "N/A";
    }
    const size_t i = static_cast<size_t>(y)*m_image.cols + x;
    if (correctDistances) {
        if (m_correctDistances[i] <= 0.0) {
            return "N/A";
        }
        if (m_correctDistances[i] >= SKY_DISTANCE_MM) {
            return "Sky";
        }
    } else if (m_predictedSky[i]) {
        return "Sky";
    }
    const double dx = point.x*MM_TO_OPENGL_UNIT - m_position[0];
    const double dy = point.y*MM_TO_OPENGL_UNIT - m_position[1];
    const double dz = point.z*MM_TO_OPENGL_UNIT - m_position[2];
    return std::to_string(static_cast<long long>(std::llround(std::sqrt(dx*dx + dy*dy + dz*dz)))) + " mm";
}

/*!
 * \brief CameraImage::getObjectText
 * \param x, y pixel of the image
 * \param correctObjects true: the object of the data file, false: the object predicted by the model
 * \return the object of the pixel, for example "SUV (97%), 4.80 m long, 1.86 m wide, 1.84 m high, 12.3 m away,
 * heading 35 degrees" (the predicted ones with the probability of the class), "Sky", or "N/A" if it is not known
 */
std::string CameraImage::getObjectText(int x, int y, bool correctObjects) const
{
    if (x < 0 || y < 0 || x >= m_image.cols || y >= m_image.rows) {
        return "N/A";
    }
    const size_t i = static_cast<size_t>(y)*m_image.cols + x;
    if (correctObjects) {
        if (!hasCorrectObjects()) {
            return "N/A (no objects in the data file)";
        }
        const uint32_t surface = m_correctSurfaces[i];
        if (surface >= m_correctObjects.size()) {
            return "N/A";
        }
        const CorrectObject &object = m_correctObjects[surface];
        if (object.objectClass == 0 || object.objectClass > static_cast<uint32_t>(OBJECT_CLASS_COUNT)) {
            return object.objectClass == 0 ? "Sky" : "N/A";
        }
        return objectText(static_cast<int>(object.objectClass) - 1, -1, object.center, object.size, object.yaw);
    }
    if (!hasPredictedObjects()) {
        return "N/A (the model has no object output)";
    }
    const ObjectPixel &pixel = m_predictedObjects[i];
    if (pixel.objectClass >= OBJECT_CLASS_COUNT) {
        return "Sky";
    }
    // box of the camera to the world: x right, y down, z forward
    double center[3], size[3], front[3];
    // the heading of the learner is the front of the object in the camera (ImageSamples::getObjectTable()), its forward
    // part is shorter by the cosine of the pitch of the camera, so the forward is the horizontal forward over the cosine
    const double cosPitch = std::max(1e-6, std::hypot(m_direction[0], m_direction[2]));
    for (int k=0;k<3;k++) {
        center[k] = m_position[k] + MM_PER_METER*(m_right[k]*pixel.center[0] - m_up[k]*pixel.center[1]
                                                  + m_direction[k]*pixel.center[2]);
        size[k] = MM_PER_METER*pixel.size[k];
        const double forward = k == 1 ? 0.0 : m_direction[k]/cosPitch;
        front[k] = forward*pixel.heading[0]/cosPitch + m_right[k]*pixel.heading[1];
    }
    const double yaw = std::atan2(front[0], front[2])*180.0/M_PI;
    return objectText(pixel.objectClass, pixel.probability, center, size, yaw);
}

void CameraImage::setRigCenter(double x, double z)
{
    m_rigCenter[0] = x;
    m_rigCenter[1] = z;
}

/*!
 * \brief CameraImage::objectText
 * \param objectClass class of the model
 * \param probability percent, -1: not shown
 * \param center center of the box in the world (mm)
 * \param size width, height, length of the box (mm)
 * \param yaw degrees in the world, 0 is +z, positive turns to +x (the same in the rig coordinates: 0 is +y)
 * \return text with the place in the rig coordinates (worldToRig(): origin at the center of the cameras on the floor,
 * x right, y forward)
 */
std::string CameraImage::objectText(int objectClass, int probability, const double *center, const double *size,
                                    double yaw) const
{
    char text[256];
    int length = snprintf(text, sizeof(text), "%s", OBJECT_NAMES[objectClass]);
    if (probability >= 0) {
        length += snprintf(text + length, sizeof(text) - static_cast<size_t>(length), " (%d%%)", probability);
    }
    if (objectClass < OBJECT_FIRST_BOX_CLASS) {
        return text;
    }
    double squared = 0.0;
    for (int k=0;k<3;k++) {
        squared += (center[k] - m_position[k])*(center[k] - m_position[k]);
    }
    // heading -180..180 degrees
    const double heading = yaw - 360.0*std::floor((yaw + 180.0)/360.0);
    double rig[3];
    worldToRig(center, m_rigCenter[0], m_rigCenter[1], rig);
    if (objectClass >= OBJECT_FIRST_PERSON_CLASS) {
        snprintf(text + length, sizeof(text) - static_cast<size_t>(length),
                 ", %.2f m tall, %.1f m away, at x %.1f m, y %.1f m, heading %.0f degrees", size[1]/MM_PER_METER,
                 std::sqrt(squared)/MM_PER_METER, rig[0]/MM_PER_METER, rig[1]/MM_PER_METER, heading);
    } else {
        snprintf(text + length, sizeof(text) - static_cast<size_t>(length),
                 ", %.2f m long, %.2f m wide, %.2f m high, %.1f m away, at x %.1f m, y %.1f m, heading %.0f degrees",
                 size[2]/MM_PER_METER, size[0]/MM_PER_METER, size[1]/MM_PER_METER, std::sqrt(squared)/MM_PER_METER,
                 rig[0]/MM_PER_METER, rig[1]/MM_PER_METER, heading);
    }
    return text;
}

/*!
 * \brief CameraImage::getObjectId
 * \param correctObjects true: the surface number of the data file if it is an object (not the sky or the ground),
 * false: the predicted object of setPredictedObjectIds()
 * \return the object of the pixel, -1: none
 */
int CameraImage::getObjectId(int x, int y, bool correctObjects) const
{
    if (x < 0 || y < 0 || x >= m_image.cols || y >= m_image.rows) {
        return -1;
    }
    const size_t i = static_cast<size_t>(y)*m_image.cols + x;
    if (correctObjects) {
        if (!hasCorrectObjects()) {
            return -1;
        }
        return getCorrectClass(m_correctSurfaces[i]) >= 1 ? static_cast<int>(m_correctSurfaces[i]) : -1;
    }
    return i < m_predictedObjectIds.size() ? m_predictedObjectIds[i] : -1;
}

const std::vector<CameraImage::ObjectPixel> &CameraImage::getPredictedObjects() const
{
    return m_predictedObjects;
}

const std::vector<uint32_t> &CameraImage::getCorrectSurfaces() const
{
    return m_correctSurfaces;
}

/*!
 * \brief CameraImage::getCorrectClass
 * \return class of the model (0 ground, 1 other object ... 7 child) of the correct object of the surface, -1 for the
 * sky and the surfaces that have no object
 */
int CameraImage::getCorrectClass(uint32_t surface) const
{
    if (surface >= m_correctObjects.size() || m_correctObjects[surface].objectClass == 0
        || m_correctObjects[surface].objectClass > static_cast<uint32_t>(OBJECT_CLASS_COUNT)) {
        return -1;
    }
    return static_cast<int>(m_correctObjects[surface].objectClass) - 1;
}

void CameraImage::setPredictedObjectIds(const std::vector<int> &ids)
{
    m_predictedObjectIds = ids;
}

const cv::Mat &CameraImage::getImage() const
{
    return m_image;
}

int CameraImage::getCameraNumber() const
{
    return m_cameraNumber;
}

/*!
 * \brief CameraImage::getPosition
 * \return position in opengl units
 */
VECTOR3 CameraImage::getPosition() const
{
    return { static_cast<float>(m_position[0]/MM_TO_OPENGL_UNIT), static_cast<float>(m_position[1]/MM_TO_OPENGL_UNIT),
             static_cast<float>(m_position[2]/MM_TO_OPENGL_UNIT) };
}

VECTOR3 CameraImage::getDirection() const
{
    return { static_cast<float>(m_direction[0]), static_cast<float>(m_direction[1]), static_cast<float>(m_direction[2]) };
}

void CameraImage::deleteMesh(Mesh *mesh)
{
    if (mesh->vertexBuffer) {
        glDeleteBuffers(1, &mesh->vertexBuffer);
    }
    if (mesh->vertexArray) {
        glDeleteVertexArrays(1, &mesh->vertexArray);
    }
}
