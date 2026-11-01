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

ViewerScene::ViewerScene() :
    m_correctDistances(false),
    m_fillGaps(true),
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
    m_pickedImage(-1),
    m_pickedPixelX(0),
    m_pickedPixelY(0),
    m_pickedWindowX(0),
    m_pickedWindowY(0)
{
}

/*!
 * \brief ViewerScene::load
 * loads the images <scene>_1, <scene>_2, ... and predicts their distances with the model
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
    for (int camera=1;;camera++) {
        const std::string filename = (std::filesystem::path(imageDirectory)/(name + "_" + std::to_string(camera))).string();
        if (!std::filesystem::exists(filename + ".png")) {
            break;
        }
        std::unique_ptr<CameraImage> image = std::make_unique<CameraImage>();
        if (!image->load(filename)) {
            return false;
        }
        m_listCameraImage.push_back(std::move(image));
    }
    if (m_listCameraImage.empty()) {
        TG_ERROR_LOG("No images", (std::filesystem::path(imageDirectory)/(name + "_1.png")).string());
        return false;
    }
    TG_DEBUG_LOG("Scene", name + ":", m_listCameraImage.size(), "cameras");

    DistanceModel model;
    if (!model.load(modelFile)) {
        return false;
    }
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
        m_listCameraImage[i]->setPrediction(predictions[i].distances, predictions[i].sky, predictions[i].connections);
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
        renderPick(projection, view, height);
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
        image->render(m_correctDistances);
    }
    if (m_fillGaps) {
        // fill meshes only on the pixels that no mesh has drawn (black areas), so they never hide a real surface
        glStencilFunc(GL_EQUAL, 0, 0xFF);
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
        for (const auto &image : m_listCameraImage) {
            image->render(m_correctDistances, true);
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
void ViewerScene::renderPick(const Matrix4x4 &projection, const Matrix4x4 &view, float height)
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
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glEnable(GL_MULTISAMPLE);
    if (blend) {
        glEnable(GL_BLEND);
    }
    const int imageIndex = static_cast<int>(color[3]) - 1;
    const long pixel = static_cast<long>(color[0]) + (static_cast<long>(color[1]) << 8) + (static_cast<long>(color[2]) << 16) - 1;
    m_pickedImage = -1;
    if (imageIndex >= 0 && imageIndex < static_cast<int>(m_listCameraImage.size()) && pixel >= 0) {
        const int width = m_listCameraImage[static_cast<size_t>(imageIndex)]->getImage().cols;
        m_pickedImage = imageIndex;
        m_pickedPixelX = static_cast<int>(pixel % width);
        m_pickedPixelY = static_cast<int>(pixel/width);
    }
    m_picked = true;
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
 * shown distances, the distance is from the camera of the image at the point in mm, "Sky", or "N/A" (the pixel sees
 * nothing). A black area (no image) is "<x>/<y> Distance from camera: N/A".
 */
std::string ViewerScene::getPickedText() const
{
    const std::string point = std::to_string(m_pickedWindowX) + "/" + std::to_string(m_pickedWindowY);
    if (m_pickedImage < 0) {
        return point + " Distance from camera: N/A";
    }
    const CameraImage &image = *m_listCameraImage[static_cast<size_t>(m_pickedImage)];
    return "Camera " + std::to_string(image.getCameraNumber()) + " " + point + " Distance from camera: "
        + image.getDistanceText(m_pickedPixelX, m_pickedPixelY, m_correctDistances);
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
    m_correctDistances = !m_correctDistances;
    TG_DEBUG_LOG(m_correctDistances ? "Correct distances of the data files" : "Distances predicted by the model");
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
 * \brief ViewerScene::getFillGaps
 * \return true if the black areas are filled
 */
bool ViewerScene::getFillGaps() const
{
    return m_fillGaps;
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
