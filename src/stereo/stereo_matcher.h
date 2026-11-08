#ifndef STEREO_MATCHER_H
#define STEREO_MATCHER_H

#include <cstdint>
#include <string>
#include <vector>
#include "network/vision_transformer.h"

/*!
 * \brief StereoMatcher
 * Distance of the pixels that are seen by many cameras, by classical matching along the ray of the pixel (no learning).
 * For every camera A and every other camera B that sees the same area: distances along the ray of the pixel of A
 * (uniform in 1/distance), a 7x7 gray patch around the pixel of A is compared (ZNCC) with camera B, the patch is warped
 * by the plane through the point (the surface can face A, be horizontal or tilted by 45 degrees). A coarse search with
 * a thinned patch finds the candidates, the full patch decides between them, the distance is refined between the
 * steps. The match is kept only if camera B matches back to the same point (left-right check). If many cameras match,
 * the one with the longest baseline across the ray is used, because it gives the most exact distance.
 * The cameras are fixed (the cameras of the model file), pinhole (OpenCV k1-k6, p1, p2) and fisheye (OpenCV fisheye k1-k4,
 * up to the field of view of the lens, also over 180 degrees) cameras are supported. A fisheye camera warps the patch by
 * the planes that face its pixel (the ray) instead of the direction of the camera, pixels outside of the field of view
 * are not matched.
 * The result is the input of the model (GeoMap), so the training and the viewer must use the same matching.
 */
class StereoMatcher
{
public:
    // distance of every pixel from the other cameras: 0 if no camera matched, score and unique are 0 then
    struct GeoMap {
        int width = 0;
        int height = 0;
        std::vector<float> distance;    // mm along the ray of the pixel
        std::vector<int8_t> score;      // ZNCC of the match*127
        std::vector<int8_t> unique;     // ZNCC of the match - ZNCC of the next candidate, *127
    };

    StereoMatcher();
    // cameras of the images, every image of match() is from the camera of the same index
    bool setCameras(const std::vector<VisionTransformer::Camera> &cameras);
    // rgbImages: width*height*3 bytes (RGB) of every camera, result of every camera
    std::vector<GeoMap> match(const std::vector<const uint8_t *> &rgbImages) const;

    static bool save(const GeoMap &map, const std::string &filename);
    // bytes of the file of save()
    static size_t getFileSize(int width, int height);
    static bool load(const std::string &filename, GeoMap *map);

private:
    struct Vector { double x, y, z; };
    struct CameraData {
        int width = 0;
        int height = 0;
        Vector position, forward, up, right;
        double fx, fy, cx, cy;
        double k[6];
        double p1, p2;
        bool fisheye = false;
        double fisheyeK[4];
        double maxTheta = 0.0;      // fisheye: largest angle from the direction of the camera (radians)
        std::vector<Vector> rays;   // ray of every pixel center
        std::vector<uint8_t> valid; // 1 if the pixel sees something (fisheye: inside of the field of view)
        bool project(const Vector &point, double *u, double *v) const;
        bool isValid(int x, int y) const;
    };
    struct Candidate {
        float distance = -1.0f;
        float score = -2.0f;
        float unique = 0.0f;
    };
    std::vector<CameraData> m_cameras;
    std::vector<std::vector<bool>> m_overlap;   // m_overlap[a][b]: B sees a part of the view of A

    static void setFisheyeRays(const double *values, CameraData *camera);
    void matchPair(int a, int b, const std::vector<std::vector<float>> &gray, std::vector<Candidate> *result) const;
    float patchScore(const CameraData &A, const CameraData &B, const std::vector<float> &grayA,
                     const std::vector<float> &grayB, int x, int y, double distance, int step) const;
};

#endif // STEREO_MATCHER_H
