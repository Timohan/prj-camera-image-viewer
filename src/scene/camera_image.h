#ifndef CAMERA_IMAGE_H
#define CAMERA_IMAGE_H

#include <string>
#include <vector>
#include <GL/glew.h>
#include <opencv2/core.hpp>
#include "math/math3d.h"

/*!
 * \brief CameraImage
 * One camera image of prj-camera-image-creator: colors from the png file, camera and lens information and
 * correct distances from the data file, predicted distances from the model. The pixels are turned to 3D points
 * (camera position + distance along the ray of the pixel through the lens) and neighbouring points are joined
 * to textured triangles, so the image becomes a 3D surface. Triangles are not made over a jump of the distance
 * (edges of the objects). Sky (the correct sky of the data file, or the sky that the model predicted) is put far away
 * (sky dome). Positions are in opengl units (1000 mm = 1.0).
 */
class CameraImage
{
public:
    CameraImage();
    ~CameraImage();
    CameraImage(const CameraImage &) = delete;
    CameraImage &operator=(const CameraImage &) = delete;

    bool load(const std::string &filename);
    void setPrediction(const std::vector<double> &distances, const std::vector<uint8_t> &sky);
    int createMeshes(GLint vertexAttribute, GLint textureCoordinateAttribute);
    void render(bool correctDistances) const;

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
    std::vector<VECTOR3> m_rays;            // normalized ray of every pixel in the world
    int m_cameraNumber;
    double m_position[3];                   // mm
    double m_direction[3];
    GLuint m_texture;
    Mesh m_correctMesh;
    Mesh m_predictedMesh;

    bool readData(const std::string &filename);
    int createMesh(const std::vector<double> &distances, const std::vector<uint8_t> &sky, GLint vertexAttribute,
                   GLint textureCoordinateAttribute, Mesh *mesh);
    static void deleteMesh(Mesh *mesh);
};

#endif // CAMERA_IMAGE_H
