#ifndef STEREO_MATCHER_H
#define STEREO_MATCHER_H

#include <cstdint>
#include <string>
#include <vector>
#include "network/vision_transformer.h"
#include "stereo/stereo_core.h"

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
 * The matching of a pixel is in stereo_core.h, CudaStereoMatcher of prj-camera-image-learner runs the same code on the GPU.
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
    // the cameras point to their own rays
    StereoMatcher(const StereoMatcher &) = delete;
    StereoMatcher &operator=(const StereoMatcher &) = delete;
    // cameras of the images, every image of match() is from the camera of the same index
    bool setCameras(const std::vector<VisionTransformer::Camera> &cameras);
    // rgbImages: width*height*3 bytes (RGB) of every camera, result of every camera
    std::vector<GeoMap> match(const std::vector<const uint8_t *> &rgbImages) const;

    static bool save(const GeoMap &map, const std::string &filename);
    // bytes of the file of save()
    static size_t getFileSize(int width, int height);
    static bool load(const std::string &filename, GeoMap *map);

private:
    friend class CudaStereoMatcher;
    typedef stereo::Vector Vector;
    typedef stereo::Candidate Candidate;
    // rays and valid of the camera point to rayData and validData
    struct CameraData : stereo::Camera {
        std::vector<Vector> rayData;
        std::vector<uint8_t> validData;
    };
    std::vector<CameraData> m_cameras;
    std::vector<std::vector<bool>> m_overlap;   // m_overlap[a][b]: B sees a part of the view of A

    static void setFisheyeRays(const double *values, CameraData *camera);
    static std::vector<std::vector<float>> toGray(const std::vector<CameraData> &cameras,
                                                  const std::vector<const uint8_t *> &rgbImages);
    void matchPair(int a, int b, const std::vector<std::vector<float>> &gray, std::vector<Candidate> *result) const;
    // left-right check and the camera with the longest baseline, pairs[a][b] is the result of matchPair(a, b)
    std::vector<GeoMap> combine(const std::vector<std::vector<std::vector<Candidate>>> &pairs) const;
};

#endif // STEREO_MATCHER_H
