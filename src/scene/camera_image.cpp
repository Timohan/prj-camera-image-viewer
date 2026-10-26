#include "camera_image.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "global_log.h"

#define MM_TO_OPENGL_UNIT           1000.0
// distances at least this far (mm) are the sky, the creator saves the sky as far as the sun
#define SKY_DISTANCE_MM             1.0e12
// sky and the predicted pixels above the horizon are put this far (mm)
#define FAR_DISTANCE_MM             500000.0
// triangle is not made, if its largest distance is this many times its smallest distance (edge of an object)
#define DISTANCE_JUMP_RATIO         1.15
// camera and lens information block after the distances in the data file
#define DATA_CAMERA_INFO_MAGIC      "CAMI"
#define DATA_CAMERA_INFO_VERSION    2
#define DATA_CAMERA_INFO_VALUES     36
// undistortPoints iterations and accuracy in pixels like in prj-camera-image-creator
#define UNDISTORT_MAX_ITERATIONS    200
#define UNDISTORT_EPSILON_PIXELS    0.001
// values of a vertex: x, y, z, s, t
#define VERTEX_VALUES               5

CameraImage::CameraImage() :
    m_cameraNumber(0),
    m_position{ 0.0, 0.0, 0.0 },
    m_direction{ 0.0, 0.0, 1.0 },
    m_texture(0)
{
}

CameraImage::~CameraImage()
{
    deleteMesh(&m_correctMesh);
    deleteMesh(&m_predictedMesh);
    if (m_texture) {
        glDeleteTextures(1, &m_texture);
    }
}

/*!
 * \brief CameraImage::load
 * loads the png and data file and calculates the ray of every pixel
 * \param filename full path of the png and data files without the extension
 * \return true on success
 */
bool CameraImage::load(const std::string &filename)
{
    cv::Mat image = cv::imread(filename + ".png", cv::IMREAD_COLOR);
    if (image.empty()) {
        TG_ERROR_LOG("Cannot read", filename + ".png");
        return false;
    }
    cv::cvtColor(image, m_image, cv::COLOR_BGR2RGB);
    return readData(filename + ".dat");
}

/*!
 * \brief CameraImage::readData
 * reads the correct distances and the camera and lens information of the data file of prj-camera-image-creator:
 * uint32_t width, height, double distance (mm) of every pixel, char[4] "CAMI", uint32_t version, camera number,
 * distortion model (0 pinhole, 1 fisheye) and 36 doubles: position (mm), direction, up vector, roll, focal length,
 * sensor width/height, resolution width/height, fx, fy, cx, cy, k1-k6, p1-p2, fisheye k1-k4, field of view,
 * aperture, focus distance, yaw, pitch. Then calculates the ray of every pixel through the lens.
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
    double values[DATA_CAMERA_INFO_VALUES];
    file.read(magic, sizeof(magic));
    file.read(reinterpret_cast<char *>(&version), sizeof(version));
    file.read(reinterpret_cast<char *>(&cameraNumber), sizeof(cameraNumber));
    file.read(reinterpret_cast<char *>(&distortionModel), sizeof(distortionModel));
    file.read(reinterpret_cast<char *>(values), sizeof(values));
    if (!file || memcmp(magic, DATA_CAMERA_INFO_MAGIC, sizeof(magic)) != 0 || version != DATA_CAMERA_INFO_VERSION) {
        TG_ERROR_LOG(filename, "has no camera information of version", DATA_CAMERA_INFO_VERSION);
        return false;
    }
    m_cameraNumber = static_cast<int>(cameraNumber);
    for (int i=0;i<3;i++) {
        m_position[i] = values[i];
        m_direction[i] = values[3 + i];
    }
    const VECTOR3 forward = { static_cast<float>(values[3]), static_cast<float>(values[4]), static_cast<float>(values[5]) };
    const VECTOR3 up = { static_cast<float>(values[6]), static_cast<float>(values[7]), static_cast<float>(values[8]) };
    // left-handed world (x right, y up, z forward): right = up x forward
    const VECTOR3 right = Math3D::Vec3Normalize(Math3D::Vec3Cross(up, forward));
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
    return true;
}

/*!
 * \brief CameraImage::setPrediction
 * \param distances predicted distance (mm) of every pixel
 * \param sky 1 if the model predicted that the pixel is sky
 */
void CameraImage::setPrediction(const std::vector<double> &distances, const std::vector<uint8_t> &sky)
{
    m_predictedDistances = distances;
    m_predictedSky = sky;
}

/*!
 * \brief CameraImage::createMeshes
 * creates the texture of the image and the meshes of the correct and the predicted distances,
 * opengl context must be current
 * \return 0 on success
 */
int CameraImage::createMeshes(GLint vertexAttribute, GLint textureCoordinateAttribute)
{
    glGenTextures(1, &m_texture);
    glBindTexture(GL_TEXTURE_2D, m_texture);
    // rows of the image are 3*width bytes, not 4 byte aligned
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, m_image.cols, m_image.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, m_image.data);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);
    // sky of the data file is very far or 0 (pixel sees nothing)
    std::vector<uint8_t> correctSky(m_correctDistances.size());
    for (size_t i=0;i<m_correctDistances.size();i++) {
        correctSky[i] = m_correctDistances[i] <= 0.0 || m_correctDistances[i] >= SKY_DISTANCE_MM ? 1 : 0;
    }
    if (createMesh(m_correctDistances, correctSky, vertexAttribute, textureCoordinateAttribute, &m_correctMesh) != 0) {
        return 1;
    }
    return createMesh(m_predictedDistances, m_predictedSky, vertexAttribute, textureCoordinateAttribute, &m_predictedMesh);
}

/*!
 * \brief CameraImage::createMesh
 * \param distances distance (mm) of every pixel
 * \param sky 1 if the pixel is sky, it is put far away
 * \return 0 on success
 */
int CameraImage::createMesh(const std::vector<double> &distances, const std::vector<uint8_t> &sky, GLint vertexAttribute,
                            GLint textureCoordinateAttribute, Mesh *mesh)
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
    std::vector<float> vertices;
    auto addVertex = [&](int x, int y) {
        const size_t i = static_cast<size_t>(y)*width + x;
        const VECTOR3 &ray = m_rays[i];
        vertices.push_back(static_cast<float>((m_position[0] + distance[i]*ray.x)/MM_TO_OPENGL_UNIT));
        vertices.push_back(static_cast<float>((m_position[1] + distance[i]*ray.y)/MM_TO_OPENGL_UNIT));
        vertices.push_back(static_cast<float>((m_position[2] + distance[i]*ray.z)/MM_TO_OPENGL_UNIT));
        // texture coordinates of the pixel center
        vertices.push_back((static_cast<float>(x) + 0.5f)/static_cast<float>(width));
        vertices.push_back((static_cast<float>(y) + 0.5f)/static_cast<float>(height));
    };
    auto connected = [&](int x0, int y0, int x1, int y1, int x2, int y2) {
        const double a = distance[static_cast<size_t>(y0)*width + x0];
        const double b = distance[static_cast<size_t>(y1)*width + x1];
        const double c = distance[static_cast<size_t>(y2)*width + x2];
        return std::max(a, std::max(b, c)) <= DISTANCE_JUMP_RATIO*std::min(a, std::min(b, c));
    };
    // two triangles between every 2x2 pixels
    for (int y=0;y+1<height;y++) {
        for (int x=0;x+1<width;x++) {
            if (connected(x, y, x + 1, y, x, y + 1)) {
                addVertex(x, y);
                addVertex(x + 1, y);
                addVertex(x, y + 1);
            }
            if (connected(x + 1, y, x + 1, y + 1, x, y + 1)) {
                addVertex(x + 1, y);
                addVertex(x + 1, y + 1);
                addVertex(x, y + 1);
            }
        }
    }
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
    return 0;
}

/*!
 * \brief CameraImage::render
 * \param correctDistances true: mesh of the correct distances, false: mesh of the predicted distances
 */
void CameraImage::render(bool correctDistances) const
{
    const Mesh &mesh = correctDistances ? m_correctMesh : m_predictedMesh;
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_texture);
    glBindVertexArray(mesh.vertexArray);
    glDrawArrays(GL_TRIANGLES, 0, mesh.vertexCount);
    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
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
