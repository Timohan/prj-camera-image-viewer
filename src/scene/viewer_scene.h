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
 */
class ViewerScene
{
public:
    ViewerScene();

    bool load(const std::string &imageDirectory, const std::string &scene, const std::string &modelFile);
    int initGl();
    void render(float width, float height);
    void moveToCamera(int index);
    void toggleDistances();
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

    static std::string findScene(const std::string &imageDirectory);
};

#endif // VIEWER_SCENE_H
