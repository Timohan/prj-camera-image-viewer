#ifndef VIEWER_SCENE_H
#define VIEWER_SCENE_H

#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include "camera_image.h"
#include "shader/viewer_shader.h"

/*!
 * \brief ViewerScene
 * 3D surfaces of the images of all cameras of one scene and the view camera that looks around them.
 * The view camera starts in the middle of the cameras and can turn to any direction (360 degrees),
 * move forward and backward and jump to the place of every camera.
 * Distances are the ones that the model predicted, or the correct ones of the data files for comparison.
 * A point of the window can be picked: the next render finds the camera image pixel that is shown there.
 */
class ViewerScene
{
public:
    ViewerScene();

    bool load(const std::string &imageDirectory, const std::string &scene, const std::string &modelFile);
    int initGl();
    void render(float width, float height);
    void pick(float x, float y);
    bool takePicked();
    std::string getPickedText() const;
    void moveToCamera(int index);
    void toggleDistances();
    bool getCorrectDistances() const;
    void startTurn(int direction);
    void stopTurn(int direction);
    void startPitch(int direction);
    void stopPitch(int direction);
    void move(float steps);
    bool update();

private:
    ViewerShader m_shader;
    std::vector<std::unique_ptr<CameraImage>> m_listCameraImage;
    bool m_correctDistances;    // true: correct distances, false: predicted distances
    VECTOR3 m_position;         // view camera, opengl units
    float m_yaw;                // radians, 0 looks to +z, positive turns to +x
    float m_pitch;              // radians, positive looks up
    int m_yawDirection;         // -1, 0 or 1 while an arrow key is down
    int m_pitchDirection;
    std::chrono::steady_clock::time_point m_lastUpdate;
    bool m_pickPending;         // pick at m_pickX, m_pickY in the next render
    float m_pickX;              // window pixels from the top left corner
    float m_pickY;
    bool m_picked;              // a pick was done and takePicked() has not returned it yet
    int m_pickedImage;          // index of the camera image at the picked point, -1: no image (black area)
    int m_pickedPixelX;         // pixel of the camera image at the picked point
    int m_pickedPixelY;
    int m_pickedWindowX;        // picked point, window pixels from the top left corner
    int m_pickedWindowY;

    void renderPick(const Matrix4x4 &projection, const Matrix4x4 &view, float height);

    static std::string findScene(const std::string &imageDirectory);
};

#endif // VIEWER_SCENE_H
