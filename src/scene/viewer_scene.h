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
 * Black areas (places that no camera sees, behind the edges of the objects) can be filled with the background
 * behind the edge (fill meshes of CameraImage).
 * Objects: the images can be tinted by the classes of the objects of their pixels (predicted by a model with the object
 * output, or the correct ones of the data files with the correct distances), and the picked pixel shows its object.
 * A model with the object ids gives every pixel the id of its object: the pixels of all images whose ids are near
 * each other are one object (clusterObjects()), so an object that many cameras see is one object, its box is the mean
 * of the world boxes of its pixels. The images can be tinted by the objects (every object in its own color, the
 * correct objects are the surface numbers of the data files).
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
    std::string getPickedObjectText() const;
    void moveToCamera(int index);
    void toggleDistances();
    bool getCorrectDistances() const;
    bool hasCorrectDistances() const;
    void toggleFillGaps();
    bool getFillGaps() const;
    void toggleObjects();
    CameraImage::ObjectsMode getObjectsMode() const;
    bool hasObjects() const;
    bool hasObjectIds() const;
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
    bool m_fillGaps;            // black areas are filled with the fill meshes
    CameraImage::ObjectsMode m_objectsMode;  // images are tinted by the classes or the ids of the objects
    // predicted object of the ids (clusterObjects()): the cameras that see it (bit 0 is camera 1), its pixels, the
    // sums of the world boxes of its pixels (rig coordinates, m) and the pixels of every predicted class
    struct ObjectGroup {
        uint32_t cameras = 0;
        uint32_t cameraPixels[32] = {};
        size_t pixels = 0;
        double center[3] = { 0.0, 0.0, 0.0 };
        double heading[2] = { 0.0, 0.0 };
        double logSize[3] = { 0.0, 0.0, 0.0 };
        double probability = 0.0;
        size_t classPixels[OBJECT_CLASS_COUNT] = {};
    };
    std::vector<ObjectGroup> m_objectGroups;
    double m_rigCenter[2] = { 0.0, 0.0 };   // x and z of the center of the model cameras in the world, mm (worldToRig())
    bool m_objectIds;           // the model has the object ids
    // cameras that see at least OBJECT_GROUP_MIN_PIXELS pixels of the correct objects (surface numbers)
    std::vector<uint32_t> m_correctObjectCameras;

    void clusterObjects();
    static std::string cameraList(uint32_t cameras);
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
    bool m_pickDone;            // at least one pick was done
    int m_pickedImage;          // index of the camera image at the picked point, -1: no image (black area)
    int m_pickedPixelX;         // pixel of the camera image at the picked point
    int m_pickedPixelY;
    int m_pickedWindowX;        // picked point, window pixels from the top left corner
    int m_pickedWindowY;
    VECTOR3 m_pickedPoint;      // picked point on the mesh, opengl units

    void renderPick(const Matrix4x4 &projection, const Matrix4x4 &view, float width, float height);
    static bool unproject(const Matrix4x4 &projection, const Matrix4x4 &view, double x, double y, double z, VECTOR3 *point);

    static std::string findScene(const std::string &imageDirectory);
};

#endif // VIEWER_SCENE_H
