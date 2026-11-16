#ifndef CAMERA_IMAGE_H
#define CAMERA_IMAGE_H

#include <string>
#include <vector>
#include <GL/glew.h>
#include <opencv2/core.hpp>
#include "math/math3d.h"
#include "network/vision_transformer.h"

/*!
 * \brief CameraImage
 * One camera image of prj-camera-image-creator: colors from the png file, camera and lens information from the model,
 * correct distances from the data file if it is there (only for comparison), predicted distances from the model. The pixels are turned to 3D points
 * (camera position + distance along the ray of the pixel through the lens) and neighbouring points are joined
 * to textured triangles, so the image becomes a 3D surface. Triangles are not made over the edges of the surfaces:
 * the correct edges are from the surface numbers of the data file and the predicted edges from the model (surface output),
 * without them a jump of the distance is an edge. Sky (the correct sky of the data file, or the sky that the model
 * predicted) is put far away (sky dome). Positions are in opengl units (1000 mm = 1.0).
 * The fill meshes have the triangles over the edges that the meshes do not have, colored by their farthest pixel
 * (the background behind the edge), they fill the black areas that no camera sees (see ViewerScene::render()).
 * Objects: a model with the object output predicts the object of every pixel (ground, other object, sedan, SUV, sports
 * car, man, woman, child and the box of a car or a person), the data file has the correct object of every surface
 * ("OBJS" of the newer creator). The object textures are the image tinted by the class of the object of every pixel
 * (OBJECT_COLORS of camera_image.cpp).
 */
class CameraImage
{
public:
    // connections of a pixel: it is on the same surface as its right / lower neighbour
    enum Connection : uint8_t {
        ConnectedRight = 1,
        ConnectedDown = 2
    };

    // object of a pixel predicted by the model (pixel_label.h object outputs)
    struct ObjectPixel {
        uint8_t objectClass = NO_OBJECT;    // class of the model (0 ground ... 7 child), NO_OBJECT: sky
        uint8_t probability = 0;            // probability of the class, percent
        float center[3] = { 0.0f, 0.0f, 0.0f };     // center of the box in the camera (x right, y down, z forward), m
        float heading[2] = { 1.0f, 0.0f };          // cos and sin of the angle from the forward of the camera to the front
        float size[3] = { 0.0f, 0.0f, 0.0f };       // width, height and length of the box, m
        // with the object ids of the model (Config::objectIds): the box in the world and the id of the object
        bool hasId = false;
        float worldCenter[3] = { 0.0f, 0.0f, 0.0f };    // center of the box in the world, m
        float worldHeading[2] = { 1.0f, 0.0f };         // cos and sin of the yaw in the world
        float id[OBJECT_ID_SIZE] = {};
    };
    // tint of the images: none, the classes of the objects, every object (id) in its own color
    enum ObjectsMode {
        ObjectsOff = 0,
        ObjectsClasses,
        ObjectsIds
    };
    static const uint8_t NO_OBJECT = 255;

    CameraImage();
    ~CameraImage();
    CameraImage(const CameraImage &) = delete;
    CameraImage &operator=(const CameraImage &) = delete;

    bool load(const std::string &filename, const VisionTransformer::Camera &camera);
    bool hasCorrectDistances() const;
    bool hasCorrectObjects() const;
    bool hasPredictedObjects() const;
    void setPrediction(const std::vector<double> &distances, const std::vector<uint8_t> &sky,
                       const std::vector<uint8_t> &connections, const std::vector<ObjectPixel> &objects);
    int createMeshes(GLint vertexAttribute, GLint textureCoordinateAttribute);
    // objectsMode: the image is tinted by the classes or the ids of the objects (correct or predicted ones like the
    // distances)
    void render(bool correctDistances, bool fillMesh = false, ObjectsMode objectsMode = ObjectsOff) const;
    std::string getDistanceText(int x, int y, bool correctDistances, const VECTOR3 &point) const;
    std::string getObjectText(int x, int y, bool correctObjects) const;
    // object of the pixel for the object ids: the surface number of the data file (correct) or the predicted object
    // (setPredictedObjectIds()), -1: none
    int getObjectId(int x, int y, bool correctObjects) const;
    const std::vector<ObjectPixel> &getPredictedObjects() const;
    const std::vector<uint32_t> &getCorrectSurfaces() const;
    // class of the model of the correct object of the surface, -1: no object (sky, ground or not known)
    int getCorrectClass(uint32_t surface) const;
    // predicted object of every pixel (-1: none), made by ViewerScene from the ids of all images
    void setPredictedObjectIds(const std::vector<int> &ids);
    // "<class> (<probability>%), <sizes>, <distance from the camera> away, heading <yaw> degrees" of a box in the world
    std::string objectText(int objectClass, int probability, const double *center, const double *size, double yaw) const;

    const cv::Mat &getImage() const;
    int getCameraNumber() const;
    VECTOR3 getPosition() const;
    VECTOR3 getDirection() const;

private:
    struct Mesh {
        GLuint vertexArray = 0;
        GLuint vertexBuffer = 0;
        GLsizei vertexCount = 0;
    };
    cv::Mat m_image;                        // RGB
    std::vector<double> m_correctDistances; // mm, width x height
    std::vector<double> m_predictedDistances;
    std::vector<uint8_t> m_predictedSky;    // 1 = model predicted sky
    std::vector<uint8_t> m_correctConnections;      // Connection of every pixel, empty if the data file has no surfaces
    std::vector<uint8_t> m_predictedConnections;    // empty if the model has no surface output
    // correct object of a surface number of the data file
    struct CorrectObject {
        uint32_t objectClass = 0;           // class of the data file (0 sky, 1 ground, 2 other object, 3 sedan ... 8 child)
        double center[3] = { 0.0, 0.0, 0.0 };   // center of the box in the world, mm
        double size[3] = { 0.0, 0.0, 0.0 };     // width, height and length of the box, mm
        double yaw = 0.0;                       // degrees, 0 is +z, positive turns to +x
    };
    std::vector<uint32_t> m_correctSurfaces;        // surface number of every pixel, empty if the data file has none
    std::vector<CorrectObject> m_correctObjects;    // by the surface number, empty if the data file has no objects
    std::vector<ObjectPixel> m_predictedObjects;    // of every pixel, empty if the model has no object output
    std::vector<VECTOR3> m_rays;            // normalized ray of every pixel in the world
    int m_cameraNumber;
    double m_position[3];                   // mm
    double m_direction[3];
    double m_right[3];                      // axes of the camera in the world: right, up (perpendicular to the direction)
    double m_up[3];
    GLuint m_texture;
    GLuint m_correctObjectTexture;          // image tinted by the classes, 0 if the objects are not known
    GLuint m_predictedObjectTexture;
    GLuint m_correctIdTexture;              // image tinted by the objects, 0 if they are not known
    GLuint m_predictedIdTexture;
    std::vector<int> m_predictedObjectIds;  // predicted object of every pixel, empty without the object ids
    Mesh m_correctMesh;
    Mesh m_predictedMesh;
    Mesh m_correctFillMesh;
    Mesh m_predictedFillMesh;

    bool readData(const std::string &filename);
    void setCamera(const VisionTransformer::Camera &camera);
    int createMesh(const std::vector<double> &distances, const std::vector<uint8_t> &sky,
                   const std::vector<uint8_t> &connections, GLint vertexAttribute,
                   GLint textureCoordinateAttribute, Mesh *mesh, Mesh *fillMesh);
    static void uploadMesh(const std::vector<float> &vertices, GLint vertexAttribute, GLint textureCoordinateAttribute,
                           Mesh *mesh);
    static void deleteMesh(Mesh *mesh);
    GLuint createTexture(const cv::Mat &image) const;
    cv::Mat tintImage(bool correctObjects, bool ids) const;
};

#endif // CAMERA_IMAGE_H
