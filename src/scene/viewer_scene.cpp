#include "viewer_scene.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <regex>
#include "distance_model.h"
#include "global_log.h"

#define VIEW_FIELD_OF_VIEW      (70.0f*PI_VALUE/180.0f)
#define VIEW_NEAR_PLANE         0.05f
#define VIEW_FAR_PLANE          1000.0f
// turning speed of the arrow keys in radians per second
#define TURN_SPEED              (PI_VALUE/2.0f)
// view camera does not turn over straight up or down
#define MAX_PITCH               (85.0f*PI_VALUE/180.0f)
// one mouse wheel step moves the view camera this far (opengl units)
#define MOVE_STEP               0.25f
#define BACKGROUND_COLOR        0.08f, 0.08f, 0.10f, 1.0f
// pixels whose ids are nearer than this to the mean id of an object are the object (the training pushes the means of
// two objects at least 2*OBJECT_ID_PUSH_MARGIN apart and pulls the pixels to OBJECT_ID_PULL_MARGIN of their mean)
#define OBJECT_GROUP_DISTANCE   OBJECT_ID_PUSH_MARGIN
// an object of fewer pixels is not an object, a camera sees an object if it sees this many pixels of it
#define OBJECT_GROUP_MIN_PIXELS 10

ViewerScene::ViewerScene() :
    m_correctDistances(false),
    m_fillGaps(false),
    m_objectsMode(CameraImage::ObjectsOff),
    m_objectIds(false),
    m_position({ 0.0f, 1.0f, 0.0f }),
    m_yaw(0.0f),
    m_pitch(0.0f),
    m_yawDirection(0),
    m_pitchDirection(0),
    m_lastUpdate(std::chrono::steady_clock::now()),
    m_pickPending(false),
    m_pickX(0.0f),
    m_pickY(0.0f),
    m_picked(false),
    m_pickDone(false),
    m_pickedImage(-1),
    m_pickedPixelX(0),
    m_pickedPixelY(0),
    m_pickedWindowX(0),
    m_pickedWindowY(0),
    m_pickedPoint({ 0.0f, 0.0f, 0.0f })
{
}

/*!
 * \brief ViewerScene::load
 * loads the images <scene>_1, <scene>_2, ... and predicts their distances with the model. Cameras and lenses are from
 * the model, the data files are read only for the correct distances if they are there.
 * \param imageDirectory output directory of prj-camera-image-creator
 * \param scene name of the scene, empty selects the scene with the largest number
 * \param modelFile model of prj-camera-image-learner
 * \return true on success
 */
bool ViewerScene::load(const std::string &imageDirectory, const std::string &scene, const std::string &modelFile)
{
    const std::string name = scene.empty() ? findScene(imageDirectory) : scene;
    if (name.empty()) {
        TG_ERROR_LOG("No scenes in", imageDirectory);
        return false;
    }
    TG_DEBUG_LOG("Using scene", name, "of", imageDirectory,
                 scene.empty() ? "(the scene with the largest number, select another with --scene=<name>)" : "(--scene)");
    DistanceModel model;
    if (!model.load(modelFile)) {
        return false;
    }
    const std::vector<VisionTransformer::Camera> &cameras = model.getCameras();
    if (cameras.empty()) {
        TG_ERROR_LOG(modelFile, "has no cameras, train a new model with prj-camera-image-learner");
        return false;
    }
    for (int camera=1;;camera++) {
        const std::string filename = (std::filesystem::path(imageDirectory)/(name + "_" + std::to_string(camera))).string();
        if (!std::filesystem::exists(filename + ".png")) {
            break;
        }
        const VisionTransformer::Camera *modelCamera = nullptr;
        for (const VisionTransformer::Camera &candidate : cameras) {
            if (static_cast<int>(candidate.number) == camera) {
                modelCamera = &candidate;
            }
        }
        if (!modelCamera) {
            TG_ERROR_LOG(modelFile, "has no camera", camera);
            return false;
        }
        std::unique_ptr<CameraImage> image = std::make_unique<CameraImage>();
        if (!image->load(filename, *modelCamera)) {
            return false;
        }
        m_listCameraImage.push_back(std::move(image));
    }
    if (m_listCameraImage.empty()) {
        TG_ERROR_LOG("No images", (std::filesystem::path(imageDirectory)/(name + "_1.png")).string());
        return false;
    }
    // rig coordinates of the world boxes: the center of all cameras of the model like the learner
    m_rigCenter[0] = m_rigCenter[1] = 0.0;
    for (const VisionTransformer::Camera &camera : cameras) {
        m_rigCenter[0] += camera.values[0]/static_cast<double>(cameras.size());
        m_rigCenter[1] += camera.values[2]/static_cast<double>(cameras.size());
    }
    for (const auto &image : m_listCameraImage) {
        image->setRigCenter(m_rigCenter[0], m_rigCenter[1]);
    }
    TG_DEBUG_LOG("Scene", name + ":", m_listCameraImage.size(), "cameras,",
                 hasCorrectDistances() ? "correct distances from the data files" : "no data files (no correct distances)");
    std::vector<const cv::Mat *> images;
    for (const auto &image : m_listCameraImage) {
        images.push_back(&image->getImage());
    }
    const auto start = std::chrono::steady_clock::now();
    const std::vector<DistanceModel::Prediction> predictions = model.predict(images);
    TG_DEBUG_LOG("Distances predicted in", std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(), "s");
    for (size_t i=0;i<m_listCameraImage.size();i++) {
        if (predictions[i].distances.empty()) {
            return false;
        }
        m_listCameraImage[i]->setPrediction(predictions[i].distances, predictions[i].sky, predictions[i].connections,
                                            predictions[i].objects);
    }
    m_objectIds = model.hasObjectIds();
    if (m_objectIds) {
        const auto clusterStart = std::chrono::steady_clock::now();
        clusterObjects();
        TG_DEBUG_LOG("Objects of the ids:", m_objectGroups.size(), "objects in",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - clusterStart).count(), "s");
    }
    // cameras that see the correct objects
    for (size_t i=0;i<m_listCameraImage.size() && i<32;i++) {
        const CameraImage &image = *m_listCameraImage[i];
        std::vector<uint32_t> pixels;
        for (uint32_t surface : image.getCorrectSurfaces()) {
            if (image.getCorrectClass(surface) < 1) {
                continue;
            }
            if (pixels.size() <= surface) {
                pixels.resize(static_cast<size_t>(surface) + 1, 0);
            }
            pixels[surface]++;
        }
        if (m_correctObjectCameras.size() < pixels.size()) {
            m_correctObjectCameras.resize(pixels.size(), 0);
        }
        for (size_t surface=0;surface<pixels.size();surface++) {
            if (pixels[surface] >= OBJECT_GROUP_MIN_PIXELS) {
                m_correctObjectCameras[surface] |= 1u << i;
            }
        }
    }
    moveToCamera(0);
    return true;
}

/*!
 * \brief ViewerScene::findScene
 * \return name of the scene that has the largest number (objects999 before objects12), empty if there are no scenes
 */
std::string ViewerScene::findScene(const std::string &imageDirectory)
{
    // <objects name><number>_<timestamp>_<camera number>.png
    const std::regex pattern(R"(^(.*?(\d+)_\d{8}-\d{6})_\d+\.png$)");
    std::string best;
    unsigned long bestNumber = 0;
    std::error_code error;
    for (const auto &entry : std::filesystem::directory_iterator(imageDirectory, error)) {
        const std::string filename = entry.path().filename().string();
        std::smatch match;
        if (!std::regex_match(filename, match, pattern)) {
            continue;
        }
        const unsigned long number = std::stoul(match[2].str());
        if (best.empty() || number > bestNumber || (number == bestNumber && match[1].str() > best)) {
            best = match[1].str();
            bestNumber = number;
        }
    }
    return best;
}

/*!
 * \brief ViewerScene::initGl
 * compiles the shader and creates the textures and meshes, opengl context must be current
 * \return 0 on success
 */
int ViewerScene::initGl()
{
    if (m_shader.init() != 0) {
        return 1;
    }
    for (const auto &image : m_listCameraImage) {
        if (image->createMeshes(m_shader.vertexAttribute(), m_shader.textureCoordinateAttribute()) != 0) {
            return 1;
        }
    }
    return 0;
}

void ViewerScene::render(float width, float height)
{
    const VECTOR3 direction = { sinf(m_yaw)*cosf(m_pitch), sinf(m_pitch), cosf(m_yaw)*cosf(m_pitch) };
    const VECTOR3 at = { m_position.x + direction.x, m_position.y + direction.y, m_position.z + direction.z };
    Matrix4x4 projection, view;
    Math3D::MatrixPerspectiveFovLH(&projection, VIEW_FIELD_OF_VIEW, width/height, VIEW_NEAR_PLANE, VIEW_FAR_PLANE);
    Math3D::MatrixLookAtLH(&view, m_position, at, { 0.0f, 1.0f, 0.0f });

    glViewport(0, 0, static_cast<GLsizei>(width), static_cast<GLsizei>(height));
    if (m_pickPending) {
        // drawn to the back buffer before the real frame, it is cleared again before it is shown
        renderPick(projection, view, width, height);
    }
    glClearColor(BACKGROUND_COLOR);
    glClearStencil(0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    glUseProgram(m_shader.program());
    glUniformMatrix4fv(glGetUniformLocation(m_shader.program(), "projection"), 1, GL_FALSE, projection.m);
    glUniformMatrix4fv(glGetUniformLocation(m_shader.program(), "view"), 1, GL_FALSE, view.m);
    glUniform1i(glGetUniformLocation(m_shader.program(), "texture"), 0);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    if (m_fillGaps) {
        // the meshes mark their pixels in the stencil buffer
        glEnable(GL_STENCIL_TEST);
        glStencilFunc(GL_ALWAYS, 1, 0xFF);
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
    }
    for (const auto &image : m_listCameraImage) {
        image->render(m_correctDistances, false, m_objectsMode);
    }
    if (m_fillGaps) {
        // fill meshes only on the pixels that no mesh has drawn (black areas), so they never hide a real surface
        glStencilFunc(GL_EQUAL, 0, 0xFF);
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
        for (const auto &image : m_listCameraImage) {
            image->render(m_correctDistances, true, m_objectsMode);
        }
        glDisable(GL_STENCIL_TEST);
    }
    glDisable(GL_DEPTH_TEST);
}

/*!
 * \brief ViewerScene::renderPick
 * draws the meshes with the pick program (camera image and pixel coded in the color, see ViewerShader)
 * and reads the color of the pick point
 */
void ViewerScene::renderPick(const Matrix4x4 &projection, const Matrix4x4 &view, float width, float height)
{
    m_pickPending = false;
    m_pickedWindowX = static_cast<int>(m_pickX);
    m_pickedWindowY = static_cast<int>(m_pickY);
    const GLuint program = m_shader.pickProgram();
    // blending would mix the coded colors, multisampling would average them at the edges of the triangles
    const GLboolean blend = glIsEnabled(GL_BLEND);
    glDisable(GL_BLEND);
    glDisable(GL_MULTISAMPLE);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(program);
    glUniformMatrix4fv(glGetUniformLocation(program, "projection"), 1, GL_FALSE, projection.m);
    glUniformMatrix4fv(glGetUniformLocation(program, "view"), 1, GL_FALSE, view.m);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    for (size_t i=0;i<m_listCameraImage.size();i++) {
        const cv::Mat &image = m_listCameraImage[i]->getImage();
        glUniform2f(glGetUniformLocation(program, "imageSize"), static_cast<float>(image.cols), static_cast<float>(image.rows));
        glUniform1f(glGetUniformLocation(program, "imageIndex"), static_cast<float>(i));
        m_listCameraImage[i]->render(m_correctDistances);
    }
    glDisable(GL_DEPTH_TEST);
    // opengl rows start from the bottom
    unsigned char color[4] = { 0, 0, 0, 0 };
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(m_pickedWindowX, static_cast<GLint>(height) - 1 - m_pickedWindowY, 1, 1,
                 GL_RGBA, GL_UNSIGNED_BYTE, color);
    // depth of the point on the mesh, 0..1 from the near to the far plane
    GLfloat depth = 1.0f;
    glReadPixels(m_pickedWindowX, static_cast<GLint>(height) - 1 - m_pickedWindowY, 1, 1,
                 GL_DEPTH_COMPONENT, GL_FLOAT, &depth);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glEnable(GL_MULTISAMPLE);
    if (blend) {
        glEnable(GL_BLEND);
    }
    const int imageIndex = static_cast<int>(color[3]) - 1;
    const long pixel = static_cast<long>(color[0]) + (static_cast<long>(color[1]) << 8) + (static_cast<long>(color[2]) << 16) - 1;
    m_pickedImage = -1;
    if (imageIndex >= 0 && imageIndex < static_cast<int>(m_listCameraImage.size()) && pixel >= 0) {
        const int imageWidth = m_listCameraImage[static_cast<size_t>(imageIndex)]->getImage().cols;
        m_pickedImage = imageIndex;
        m_pickedPixelX = static_cast<int>(pixel % imageWidth);
        m_pickedPixelY = static_cast<int>(pixel/imageWidth);
        // window pixel center and depth to normalized device coordinates (-1..1), opengl y goes up
        if (!unproject(projection, view, 2.0*(m_pickedWindowX + 0.5)/width - 1.0,
                       1.0 - 2.0*(m_pickedWindowY + 0.5)/height, 2.0*depth - 1.0, &m_pickedPoint)) {
            m_pickedImage = -1;
        }
    }
    m_picked = true;
    m_pickDone = true;
}

/*!
 * \brief ViewerScene::unproject
 * turns normalized device coordinates (x, y, z -1..1) back to the world with the inverse of projection*view
 * (the shader does projection*(view*vertex) with column-major matrices)
 * \return false if the matrix has no inverse or the point is at infinity
 */
bool ViewerScene::unproject(const Matrix4x4 &projection, const Matrix4x4 &view, double x, double y, double z, VECTOR3 *point)
{
    // m = projection*view, element (row r, column c) of a column-major matrix is m[c*4 + r]
    double a[4][8];
    for (int r=0;r<4;r++) {
        for (int c=0;c<4;c++) {
            double sum = 0.0;
            for (int k=0;k<4;k++) {
                sum += static_cast<double>(projection.m[k*4 + r])*static_cast<double>(view.m[c*4 + k]);
            }
            a[r][c] = sum;
            a[r][4 + c] = r == c ? 1.0 : 0.0;
        }
    }
    // Gauss-Jordan elimination with partial pivoting, the right half becomes the inverse
    for (int c=0;c<4;c++) {
        int pivot = c;
        for (int r=c + 1;r<4;r++) {
            if (std::fabs(a[r][c]) > std::fabs(a[pivot][c])) {
                pivot = r;
            }
        }
        if (std::fabs(a[pivot][c]) < 1e-12) {
            return false;
        }
        for (int k=0;k<8;k++) {
            std::swap(a[c][k], a[pivot][k]);
        }
        const double scale = 1.0/a[c][c];
        for (int k=0;k<8;k++) {
            a[c][k] *= scale;
        }
        for (int r=0;r<4;r++) {
            if (r != c) {
                const double factor = a[r][c];
                for (int k=0;k<8;k++) {
                    a[r][k] -= factor*a[c][k];
                }
            }
        }
    }
    const double ndc[4] = { x, y, z, 1.0 };
    double world[4];
    for (int r=0;r<4;r++) {
        world[r] = 0.0;
        for (int c=0;c<4;c++) {
            world[r] += a[r][4 + c]*ndc[c];
        }
    }
    if (std::fabs(world[3]) < 1e-12) {
        return false;
    }
    *point = { static_cast<float>(world[0]/world[3]), static_cast<float>(world[1]/world[3]),
               static_cast<float>(world[2]/world[3]) };
    return true;
}

/*!
 * \brief ViewerScene::pick
 * the next render() finds the camera image pixel that is shown at the point
 * \param x, y window pixels from the top left corner
 */
void ViewerScene::pick(float x, float y)
{
    m_pickPending = true;
    m_pickX = x;
    m_pickY = y;
}

/*!
 * \brief ViewerScene::takePicked
 * \return true once after every pick that render() has done
 */
bool ViewerScene::takePicked()
{
    const bool picked = m_picked;
    m_picked = false;
    return picked;
}

/*!
 * \brief ViewerScene::getPickedText
 * \return "Camera <number> <x>/<y> Distance from camera: <distance>" of the picked point (window pixels) with the
 * shown distances, the distance of the point on the mesh from the camera of the image in mm, "Sky", or "N/A" (the pixel
 * sees nothing). A black area (no image) is "<x>/<y> Distance from camera: N/A".
 */
std::string ViewerScene::getPickedText() const
{
    const std::string point = std::to_string(m_pickedWindowX) + "/" + std::to_string(m_pickedWindowY);
    if (m_pickedImage < 0) {
        return point + " Distance from camera: N/A";
    }
    const CameraImage &image = *m_listCameraImage[static_cast<size_t>(m_pickedImage)];
    return "Camera " + std::to_string(image.getCameraNumber()) + " " + point + " Distance from camera: "
        + image.getDistanceText(m_pickedPixelX, m_pickedPixelY, m_correctDistances, m_pickedPoint);
}

/*!
 * \brief ViewerScene::getPickedObjectText
 * \return "Object: <object>" of the picked pixel (CameraImage::getObjectText()), the correct object of the data file
 * with the correct distances, the predicted one with the predicted distances, empty before the first pick
 */
std::string ViewerScene::getPickedObjectText() const
{
    if (!m_pickDone) {
        return "";
    }
    if (m_pickedImage < 0) {
        return "Object: N/A";
    }
    const CameraImage &image = *m_listCameraImage[static_cast<size_t>(m_pickedImage)];
    const int id = image.getObjectId(m_pickedPixelX, m_pickedPixelY, m_correctDistances);
    if (m_correctDistances) {
        std::string text = "Correct object";
        if (id >= 0 && static_cast<size_t>(id) < m_correctObjectCameras.size()) {
            text += " " + std::to_string(id) + " (" + cameraList(m_correctObjectCameras[static_cast<size_t>(id)]) + ")";
        }
        return text + ": " + image.getObjectText(m_pickedPixelX, m_pickedPixelY, true);
    }
    if (id < 0 || static_cast<size_t>(id) >= m_objectGroups.size()) {
        return "Predicted object: " + image.getObjectText(m_pickedPixelX, m_pickedPixelY, false);
    }
    // the object of the ids: its class is the class of most of its pixels, its box the mean of the world boxes
    const ObjectGroup &group = m_objectGroups[static_cast<size_t>(id)];
    const int objectClass = static_cast<int>(std::max_element(group.classPixels, group.classPixels + OBJECT_CLASS_COUNT)
                                             - group.classPixels);
    const double count = static_cast<double>(std::max<size_t>(1, group.pixels));
    // the world box of the model is in the rig coordinates (worldToRig())
    double rig[3], center[3], size[3];
    for (int k=0;k<3;k++) {
        rig[k] = 1000.0*group.center[k]/count;
        size[k] = 1000.0*std::exp(group.logSize[k]/count);
    }
    rigToWorld(rig, m_rigCenter[0], m_rigCenter[1], center);
    const double yaw = std::atan2(group.heading[1], group.heading[0])*180.0/M_PI;
    return "Predicted object " + std::to_string(id + 1) + " (" + cameraList(group.cameras) + "): "
        + image.objectText(objectClass, static_cast<int>(std::lround(100.0*group.probability/count)), center, size, yaw);
}

/*!
 * \brief ViewerScene::cameraList
 * \return "camera 2" or "cameras 1, 3, 5" of the bits of the cameras (bit 0 is camera 1)
 */
std::string ViewerScene::cameraList(uint32_t cameras)
{
    std::string list;
    int count = 0;
    for (int i=0;i<32;i++) {
        if (cameras & (1u << i)) {
            list += (count++ > 0 ? ", " : "") + std::to_string(i + 1);
        }
    }
    return (count == 1 ? "camera " : "cameras ") + list;
}

/*!
 * \brief ViewerScene::clusterObjects
 * makes the objects of the ids of the pixels of all images: the pixels of the objects (not the sky and the ground) are
 * taken one by one, a pixel is a part of the object whose mean id is the nearest, if it is nearer than
 * OBJECT_GROUP_DISTANCE, otherwise it starts a new object (the mean is updated with every pixel). Then every pixel is
 * given to the nearest final mean again (nearer than OBJECT_GROUP_DISTANCE), objects of fewer than
 * OBJECT_GROUP_MIN_PIXELS pixels are left out, the largest object is 0. The objects of every image are set to the image
 * (CameraImage::setPredictedObjectIds()).
 */
void ViewerScene::clusterObjects()
{
    std::vector<std::vector<double>> means;    // OBJECT_ID_SIZE sums of the ids of every object
    std::vector<size_t> counts;
    auto nearest = [&](const float *id, bool sums, double *distance) {
        int best = -1;
        double bestSquared = OBJECT_GROUP_DISTANCE*OBJECT_GROUP_DISTANCE;
        for (size_t g=0;g<means.size();g++) {
            if (counts[g] == 0) {
                continue;
            }
            double squared = 0.0;
            for (int k=0;k<OBJECT_ID_SIZE;k++) {
                const double mean = sums ? means[g][static_cast<size_t>(k)]/static_cast<double>(counts[g]) : means[g][static_cast<size_t>(k)];
                squared += (id[k] - mean)*(id[k] - mean);
                if (squared >= bestSquared) {
                    break;
                }
            }
            if (squared < bestSquared) {
                bestSquared = squared;
                best = static_cast<int>(g);
            }
        }
        *distance = std::sqrt(bestSquared);
        return best;
    };
    auto isObject = [](const CameraImage::ObjectPixel &pixel) {
        return pixel.hasId && pixel.objectClass >= 1 && pixel.objectClass < OBJECT_CLASS_COUNT;
    };
    // 1. one pass: the objects and their means
    for (const auto &image : m_listCameraImage) {
        for (const CameraImage::ObjectPixel &pixel : image->getPredictedObjects()) {
            if (!isObject(pixel)) {
                continue;
            }
            double distance;
            const int g = nearest(pixel.id, true, &distance);
            if (g < 0) {
                means.emplace_back(pixel.id, pixel.id + OBJECT_ID_SIZE);
                counts.push_back(1);
                continue;
            }
            for (int k=0;k<OBJECT_ID_SIZE;k++) {
                means[static_cast<size_t>(g)][static_cast<size_t>(k)] += pixel.id[k];
            }
            counts[static_cast<size_t>(g)]++;
        }
    }
    for (size_t g=0;g<means.size();g++) {
        for (double &value : means[g]) {
            value /= static_cast<double>(counts[g]);
        }
    }
    // 2. every pixel to the nearest final mean
    std::vector<std::vector<int>> ids(m_listCameraImage.size());
    std::vector<ObjectGroup> groups(means.size());
    for (size_t i=0;i<m_listCameraImage.size();i++) {
        const std::vector<CameraImage::ObjectPixel> &pixels = m_listCameraImage[i]->getPredictedObjects();
        ids[i].assign(pixels.size(), -1);
        for (size_t p=0;p<pixels.size();p++) {
            const CameraImage::ObjectPixel &pixel = pixels[p];
            if (!isObject(pixel)) {
                continue;
            }
            double distance;
            const int g = nearest(pixel.id, false, &distance);
            if (g < 0) {
                continue;
            }
            ids[i][p] = g;
            ObjectGroup &group = groups[static_cast<size_t>(g)];
            group.pixels++;
            if (i < 32) {
                group.cameraPixels[i]++;
            }
            group.classPixels[pixel.objectClass]++;
            group.probability += pixel.probability;
            for (int k=0;k<3;k++) {
                group.center[k] += pixel.worldCenter[k];
                group.logSize[k] += std::log(std::max(1e-3f, pixel.size[k]));
            }
            group.heading[0] += pixel.worldHeading[0];
            group.heading[1] += pixel.worldHeading[1];
        }
    }
    // 3. the objects of enough pixels, the largest first
    std::vector<size_t> order;
    for (size_t g=0;g<groups.size();g++) {
        if (groups[g].pixels >= OBJECT_GROUP_MIN_PIXELS) {
            order.push_back(g);
        }
    }
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return groups[a].pixels > groups[b].pixels; });
    std::vector<int> number(groups.size(), -1);
    m_objectGroups.clear();
    for (size_t g : order) {
        number[g] = static_cast<int>(m_objectGroups.size());
        ObjectGroup group = groups[g];
        for (int i=0;i<32;i++) {
            if (group.cameraPixels[i] >= OBJECT_GROUP_MIN_PIXELS) {
                group.cameras |= 1u << i;
            }
        }
        m_objectGroups.push_back(group);
    }
    for (size_t i=0;i<m_listCameraImage.size();i++) {
        for (int &id : ids[i]) {
            id = id >= 0 ? number[static_cast<size_t>(id)] : -1;
        }
        m_listCameraImage[i]->setPredictedObjectIds(ids[i]);
    }
}

/*!
 * \brief ViewerScene::moveToCamera
 * \param index 0: middle of the cameras looking to +z, 1-n: place and direction of camera 1-n
 */
void ViewerScene::moveToCamera(int index)
{
    if (index < 0 || index > static_cast<int>(m_listCameraImage.size())) {
        return;
    }
    if (index == 0) {
        VECTOR3 sum = { 0.0f, 0.0f, 0.0f };
        for (const auto &image : m_listCameraImage) {
            const VECTOR3 position = image->getPosition();
            sum = { sum.x + position.x, sum.y + position.y, sum.z + position.z };
        }
        const float count = static_cast<float>(m_listCameraImage.size());
        m_position = { sum.x/count, sum.y/count, sum.z/count };
        m_yaw = 0.0f;
        m_pitch = 0.0f;
        return;
    }
    const CameraImage &image = *m_listCameraImage[static_cast<size_t>(index - 1)];
    const VECTOR3 direction = Math3D::Vec3Normalize(image.getDirection());
    m_position = image.getPosition();
    m_yaw = atan2f(direction.x, direction.z);
    m_pitch = asinf(std::max(-1.0f, std::min(1.0f, direction.y)));
}

/*!
 * \brief ViewerScene::toggleDistances
 * changes between the predicted and the correct distances
 */
void ViewerScene::toggleDistances()
{
    if (!hasCorrectDistances()) {
        TG_DEBUG_LOG("No data files, the correct distances are not known");
        return;
    }
    m_correctDistances = !m_correctDistances;
    TG_DEBUG_LOG(m_correctDistances ? "Correct distances of the data files" : "Distances predicted by the model");
    // the picked point is on the other mesh now, it is picked again in the next render
    if (m_pickDone) {
        pick(static_cast<float>(m_pickedWindowX), static_cast<float>(m_pickedWindowY));
    }
}

/*!
 * \brief ViewerScene::toggleFillGaps
 * changes between filled and black areas that no camera sees
 */
void ViewerScene::toggleFillGaps()
{
    m_fillGaps = !m_fillGaps;
    TG_DEBUG_LOG(m_fillGaps ? "Black areas are filled" : "Black areas are not filled");
}

/*!
 * \brief ViewerScene::toggleObjects
 * changes between the images and the images tinted by the classes of the objects
 */
void ViewerScene::toggleObjects()
{
    if (!hasObjects()) {
        TG_DEBUG_LOG("The model has no object output and the data files have no objects");
        return;
    }
    // off, classes, objects (ids: the predicted ones need the object ids of the model, the correct ones are the
    // surface numbers), off
    if (m_objectsMode == CameraImage::ObjectsOff) {
        m_objectsMode = CameraImage::ObjectsClasses;
    } else if (m_objectsMode == CameraImage::ObjectsClasses && (m_objectIds || hasCorrectDistances())) {
        m_objectsMode = CameraImage::ObjectsIds;
    } else {
        m_objectsMode = CameraImage::ObjectsOff;
    }
    TG_DEBUG_LOG(m_objectsMode == CameraImage::ObjectsClasses
                     ? "Classes of the objects are shown (other object blue, sedan yellow, SUV orange, sports car red, "
                       "man green, woman magenta, child cyan)"
                     : m_objectsMode == CameraImage::ObjectsIds ? "Every object is shown in its own color"
                                                                : "Objects are not shown");
}

/*!
 * \brief ViewerScene::getObjectsMode
 * \return tint of the images: none, the classes or the ids of the objects
 */
CameraImage::ObjectsMode ViewerScene::getObjectsMode() const
{
    return m_objectsMode;
}

/*!
 * \brief ViewerScene::hasObjectIds
 * \return true if the model has the object ids (every object of the predicted ids is one object in all cameras)
 */
bool ViewerScene::hasObjectIds() const
{
    return m_objectIds;
}

/*!
 * \brief ViewerScene::hasObjects
 * \return true if the model predicts the objects or the data files have them
 */
bool ViewerScene::hasObjects() const
{
    for (const auto &image : m_listCameraImage) {
        if (image->hasPredictedObjects() || image->hasCorrectObjects()) {
            return true;
        }
    }
    return false;
}

/*!
 * \brief ViewerScene::getFillGaps
 * \return true if the black areas are filled
 */
bool ViewerScene::getFillGaps() const
{
    return m_fillGaps;
}

/*!
 * \brief ViewerScene::hasCorrectDistances
 * \return true if every image has the correct distances of its data file
 */
bool ViewerScene::hasCorrectDistances() const
{
    for (const auto &image : m_listCameraImage) {
        if (!image->hasCorrectDistances()) {
            return false;
        }
    }
    return !m_listCameraImage.empty();
}

/*!
 * \brief ViewerScene::getCorrectDistances
 * \return true if the correct distances are shown, false if the predicted distances
 */
bool ViewerScene::getCorrectDistances() const
{
    return m_correctDistances;
}

void ViewerScene::startTurn(int direction)
{
    update();
    m_yawDirection = direction;
}

void ViewerScene::stopTurn(int direction)
{
    update();
    if (direction == m_yawDirection) {
        m_yawDirection = 0;
    }
}

void ViewerScene::startPitch(int direction)
{
    update();
    m_pitchDirection = direction;
}

void ViewerScene::stopPitch(int direction)
{
    update();
    if (direction == m_pitchDirection) {
        m_pitchDirection = 0;
    }
}

/*!
 * \brief ViewerScene::move
 * \param steps mouse wheel steps, positive moves forward
 */
void ViewerScene::move(float steps)
{
    const VECTOR3 direction = { sinf(m_yaw)*cosf(m_pitch), sinf(m_pitch), cosf(m_yaw)*cosf(m_pitch) };
    m_position = { m_position.x + direction.x*steps*MOVE_STEP, m_position.y + direction.y*steps*MOVE_STEP,
                   m_position.z + direction.z*steps*MOVE_STEP };
}

/*!
 * \brief ViewerScene::update
 * turns the view camera by the time from the previous update while an arrow key is down, called before every render
 * \return true if the camera is turning
 */
bool ViewerScene::update()
{
    const auto now = std::chrono::steady_clock::now();
    const float seconds = std::chrono::duration<float>(now - m_lastUpdate).count();
    m_lastUpdate = now;
    m_yaw += static_cast<float>(m_yawDirection)*TURN_SPEED*seconds;
    m_pitch = std::max(-MAX_PITCH, std::min(MAX_PITCH, m_pitch + static_cast<float>(m_pitchDirection)*TURN_SPEED*seconds));
    return m_yawDirection != 0 || m_pitchDirection != 0;
}
