#include "stereo_matcher.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <thread>
#include <opencv2/calib3d.hpp>
#include "global_log.h"

// a match is kept if its score is at least this and B matches back to a point this near (part of the distance)
#define STEREO_MIN_SCORE            0.5f
#define STEREO_BACK_MATCH_TOLERANCE 0.03
// the views overlap if a ray of A at one of these distances (mm) is inside B
#define STEREO_OVERLAP_DISTANCES    { 1500.0, 4000.0, 15000.0 }
#define STEREO_OVERLAP_GRID         8
#define STEREO_SCORE_SCALE          127.0f
#define UNDISTORT_MAX_ITERATIONS    200
#define UNDISTORT_EPSILON_PIXELS    0.001
// fisheye: Newton iterations of the angle of the pixel (like createFisheyeImage() of prj-camera-image-creator) and
// steps of 0-180 degrees when the largest angle with a growing distorted angle is searched
#define FISHEYE_NEWTON_ITERATIONS   20
#define FISHEYE_LIMIT_STEPS         3600
#define FISHEYE_FIELD_OF_VIEW_INDEX 31
// file format
#define GEO_MAGIC                   "GEOM"
#define GEO_VERSION                 1u

using stereo::dot;

StereoMatcher::StereoMatcher()
{
}

/*!
 * \brief StereoMatcher::setFisheyeRays
 * rays of the pixels of the fisheye camera like createFisheyeImage() of prj-camera-image-creator renders them: the angle
 * of the pixel from theta_d = theta*(1 + k1*theta^2 + ... + k4*theta^8) with Newton's method, the pixels over the field
 * of view (or where theta_d does not grow anymore) do not see anything
 * \param values camera values (fisheye k1-k4 27-30, field of view in degrees 31)
 * \param camera camera with its size, directions, fx, fy, cx and cy
 */
void StereoMatcher::setFisheyeRays(const double *values, CameraData *camera)
{
    CameraData &c = *camera;
    c.fisheye = true;
    for (int j=0;j<4;j++) {
        c.fisheyeK[j] = values[27 + j];
    }
    const double *k = c.fisheyeK;
    c.maxTheta = std::min(M_PI, values[FISHEYE_FIELD_OF_VIEW_INDEX]/2.0*M_PI/180.0);
    for (int i=1;i<=FISHEYE_LIMIT_STEPS;i++) {
        const double theta = M_PI*static_cast<double>(i)/FISHEYE_LIMIT_STEPS;
        const double theta2 = theta*theta;
        if (1.0 + theta2*(3.0*k[0] + theta2*(5.0*k[1] + theta2*(7.0*k[2] + theta2*9.0*k[3]))) <= 0.0) {
            c.maxTheta = std::min(c.maxTheta, theta);
            break;
        }
    }
    c.rayData.resize(static_cast<size_t>(c.width)*c.height);
    for (int y=0;y<c.height;y++) {
        for (int x=0;x<c.width;x++) {
            const size_t i = static_cast<size_t>(y)*c.width + x;
            const double xd = (static_cast<double>(x) - c.cx)/c.fx, yd = (static_cast<double>(y) - c.cy)/c.fy;
            const double thetaD = std::sqrt(xd*xd + yd*yd);
            double theta = thetaD;
            bool valid = true;
            for (int iteration=0;iteration<FISHEYE_NEWTON_ITERATIONS;iteration++) {
                const double theta2 = theta*theta;
                const double value = theta*(1.0 + theta2*(k[0] + theta2*(k[1] + theta2*(k[2] + theta2*k[3])))) - thetaD;
                const double derivative = 1.0 + theta2*(3.0*k[0] + theta2*(5.0*k[1] + theta2*(7.0*k[2] + theta2*9.0*k[3])));
                if (derivative <= 0.0) {
                    valid = false;
                    break;
                }
                const double step = value/derivative;
                theta -= step;
                if (std::fabs(step) < 1e-10) {
                    break;
                }
            }
            if (!valid || theta < 0.0 || theta > c.maxTheta) {
                c.validData[i] = 0;
                c.rayData[i] = c.forward;
                continue;
            }
            // camera coordinates x right, y down, z forward
            const double sinTheta = std::sin(theta);
            const double rx = thetaD > 0.0 ? sinTheta*xd/thetaD : 0.0, ry = thetaD > 0.0 ? sinTheta*yd/thetaD : 0.0;
            const double rz = std::cos(theta);
            c.rayData[i] = { c.right.x*rx - c.up.x*ry + c.forward.x*rz, c.right.y*rx - c.up.y*ry + c.forward.y*rz,
                          c.right.z*rx - c.up.z*ry + c.forward.z*rz };
        }
    }
}

/*!
 * \brief StereoMatcher::setCameras
 * calculates the rays of the pixels and which cameras see the same areas
 * \param cameras cameras of the images (values like the data file of prj-camera-image-creator)
 * \return false if a camera has an unknown distortion model
 */
bool StereoMatcher::setCameras(const std::vector<VisionTransformer::Camera> &cameras)
{
    m_cameras.assign(cameras.size(), CameraData());
    for (size_t i=0;i<cameras.size();i++) {
        const VisionTransformer::Camera &camera = cameras[i];
        if (camera.distortionModel != 0 && camera.distortionModel != 1) {
            TG_ERROR_LOG("Camera", camera.number, "has unknown distortion model", camera.distortionModel);
            m_cameras.clear();
            return false;
        }
        const double *v = camera.values;
        CameraData &c = m_cameras[i];
        c.width = static_cast<int>(camera.width);
        c.height = static_cast<int>(camera.height);
        auto normalize = [](double x, double y, double z) {
            const double length = std::sqrt(x*x + y*y + z*z);
            return Vector{ x/length, y/length, z/length };
        };
        c.position = { v[0], v[1], v[2] };
        c.forward = normalize(v[3], v[4], v[5]);
        c.up = normalize(v[6], v[7], v[8]);
        // left-handed world (x right, y up, z forward): right = up x forward
        c.right = normalize(c.up.y*c.forward.z - c.up.z*c.forward.y, c.up.z*c.forward.x - c.up.x*c.forward.z,
                            c.up.x*c.forward.y - c.up.y*c.forward.x);
        c.fx = v[15];
        c.fy = v[16];
        c.cx = v[17];
        c.cy = v[18];
        for (int j=0;j<6;j++) {
            c.k[j] = v[19 + j];
        }
        c.p1 = v[25];
        c.p2 = v[26];
        const size_t pixelCount = static_cast<size_t>(c.width)*c.height;
        c.validData.assign(pixelCount, 1);
        if (camera.distortionModel == 1) {
            setFisheyeRays(v, &c);
            continue;
        }
        const cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << c.fx, 0.0, c.cx, 0.0, c.fy, c.cy, 0.0, 0.0, 1.0);
        const cv::Mat distortion = (cv::Mat_<double>(1, 8) << v[19], v[20], v[25], v[26], v[21], v[22], v[23], v[24]);
        std::vector<cv::Point2d> pixels, normalized;
        for (int y=0;y<c.height;y++) {
            for (int x=0;x<c.width;x++) {
                pixels.emplace_back(static_cast<double>(x), static_cast<double>(y));
            }
        }
        cv::undistortPoints(pixels, normalized, cameraMatrix, distortion, cv::noArray(), cv::noArray(),
                            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, UNDISTORT_MAX_ITERATIONS,
                                             UNDISTORT_EPSILON_PIXELS));
        c.rayData.resize(normalized.size());
        for (size_t j=0;j<normalized.size();j++) {
            const double nx = normalized[j].x, ny = normalized[j].y;
            c.rayData[j] = normalize(c.forward.x + nx*c.right.x - ny*c.up.x, c.forward.y + nx*c.right.y - ny*c.up.y,
                                  c.forward.z + nx*c.right.z - ny*c.up.z);
        }
    }
    for (CameraData &c : m_cameras) {
        c.rays = c.rayData.data();
        c.valid = c.validData.data();
    }
    m_overlap.assign(m_cameras.size(), std::vector<bool>(m_cameras.size(), false));
    for (size_t a=0;a<m_cameras.size();a++) {
        const CameraData &A = m_cameras[a];
        for (size_t b=0;b<m_cameras.size();b++) {
            if (a == b) {
                continue;
            }
            const CameraData &B = m_cameras[b];
            for (int y=0;y<A.height && !m_overlap[a][b];y+=STEREO_OVERLAP_GRID) {
                for (int x=0;x<A.width && !m_overlap[a][b];x+=STEREO_OVERLAP_GRID) {
                    if (!A.valid[static_cast<size_t>(y)*A.width + x]) {
                        continue;
                    }
                    const Vector &r = A.rays[static_cast<size_t>(y)*A.width + x];
                    for (double t : STEREO_OVERLAP_DISTANCES) {
                        double u, v;
                        if (stereo::project(B, { A.position.x + r.x*t, A.position.y + r.y*t, A.position.z + r.z*t }, &u, &v)
                            && u >= 0.0 && v >= 0.0 && u < B.width && v < B.height) {
                            m_overlap[a][b] = true;
                            break;
                        }
                    }
                }
            }
        }
    }
    return true;
}

/*!
 * \brief StereoMatcher::matchPair
 * best distance of every pixel of A in B (stereo::matchPixel), the rows in all CPU threads
 */
void StereoMatcher::matchPair(int a, int b, const std::vector<std::vector<float>> &gray, std::vector<Candidate> *result) const
{
    const CameraData &A = m_cameras[a], &B = m_cameras[b];
    result->assign(static_cast<size_t>(A.width)*A.height, Candidate{ -1.0f, -2.0f, 0.0f });
    std::atomic<int> nextRow(STEREO_PATCH_HALF);
    auto work = [&]() {
        for (int y; (y = nextRow++) < A.height - STEREO_PATCH_HALF;) {
            for (int x=STEREO_PATCH_HALF;x<A.width - STEREO_PATCH_HALF;x++) {
                (*result)[static_cast<size_t>(y)*A.width + x] = stereo::matchPixel(A, B, gray[a].data(), gray[b].data(), x, y);
            }
        }
    };
    std::vector<std::thread> threads;
    const unsigned threadCount = std::max(1u, std::thread::hardware_concurrency());
    for (unsigned i=0;i<threadCount;i++) {
        threads.emplace_back(work);
    }
    for (std::thread &thread : threads) {
        thread.join();
    }
}

/*!
 * \brief StereoMatcher::toGray
 * \return gray 0..1 of every pixel of every camera
 */
std::vector<std::vector<float>> StereoMatcher::toGray(const std::vector<CameraData> &cameras,
                                                      const std::vector<const uint8_t *> &rgbImages)
{
    std::vector<std::vector<float>> gray(cameras.size());
    for (size_t c=0;c<cameras.size();c++) {
        const size_t pixels = static_cast<size_t>(cameras[c].width)*cameras[c].height;
        gray[c].resize(pixels);
        for (size_t i=0;i<pixels;i++) {
            const uint8_t *rgb = rgbImages[c] + i*3;
            gray[c][i] = (0.299f*rgb[0] + 0.587f*rgb[1] + 0.114f*rgb[2])/255.0f;
        }
    }
    return gray;
}

/*!
 * \brief StereoMatcher::match
 * \param rgbImages RGB image of every camera of setCameras()
 * \return distance, score and uniqueness of every pixel of every camera, from the camera that matches back and has
 * the longest baseline across the ray
 */
std::vector<StereoMatcher::GeoMap> StereoMatcher::match(const std::vector<const uint8_t *> &rgbImages) const
{
    const int count = static_cast<int>(m_cameras.size());
    if (static_cast<int>(rgbImages.size()) != count) {
        TG_ERROR_LOG("Matching needs", count, "images, got", rgbImages.size());
        return std::vector<GeoMap>();
    }
    const std::vector<std::vector<float>> gray = toGray(m_cameras, rgbImages);
    std::vector<std::vector<std::vector<Candidate>>> pairs(m_cameras.size(), std::vector<std::vector<Candidate>>(m_cameras.size()));
    for (int a=0;a<count;a++) {
        for (int b=0;b<count;b++) {
            if (m_overlap[a][b]) {
                matchPair(a, b, gray, &pairs[a][b]);
            }
        }
    }
    return combine(pairs);
}

/*!
 * \brief StereoMatcher::combine
 * the match of a pixel of A in B is kept if B matches back to the same point, of them the camera with the longest
 * baseline across the ray is used
 * \param pairs pairs[a][b] result of matchPair(a, b), empty if the cameras do not overlap
 * \return distance, score and uniqueness of every pixel of every camera
 */
std::vector<StereoMatcher::GeoMap> StereoMatcher::combine(const std::vector<std::vector<std::vector<Candidate>>> &pairs) const
{
    const int count = static_cast<int>(m_cameras.size());
    std::vector<GeoMap> result(m_cameras.size());
    for (int a=0;a<count;a++) {
        const CameraData &A = m_cameras[a];
        GeoMap &map = result[a];
        const size_t pixels = static_cast<size_t>(A.width)*A.height;
        map.width = A.width;
        map.height = A.height;
        map.distance.assign(pixels, 0.0f);
        map.score.assign(pixels, 0);
        map.unique.assign(pixels, 0);
        for (size_t i=0;i<pixels;i++) {
            if (!A.valid[i]) {
                continue;
            }
            double bestBaseline = -1.0;
            for (int b=0;b<count;b++) {
                if (pairs[a][b].empty() || pairs[b][a].empty()) {
                    continue;
                }
                const Candidate &candidate = pairs[a][b][i];
                if (candidate.distance <= 0.0f || candidate.score < STEREO_MIN_SCORE) {
                    continue;
                }
                // B must match back to the same point
                const CameraData &B = m_cameras[b];
                const Vector &r = A.rays[i];
                const Vector point = { A.position.x + r.x*candidate.distance, A.position.y + r.y*candidate.distance,
                                       A.position.z + r.z*candidate.distance };
                double u, v;
                if (!stereo::project(B, point, &u, &v)) {
                    continue;
                }
                const int ui = static_cast<int>(std::lround(u)), vi = static_cast<int>(std::lround(v));
                if (ui < 0 || vi < 0 || ui >= B.width || vi >= B.height) {
                    continue;
                }
                const size_t j = static_cast<size_t>(vi)*B.width + ui;
                const Candidate &back = pairs[b][a][j];
                if (!B.valid[j] || back.distance <= 0.0f) {
                    continue;
                }
                const Vector &rb = B.rays[j];
                const double ex = B.position.x + rb.x*back.distance - point.x, ey = B.position.y + rb.y*back.distance - point.y;
                const double ez = B.position.z + rb.z*back.distance - point.z;
                if (std::sqrt(ex*ex + ey*ey + ez*ez) > STEREO_BACK_MATCH_TOLERANCE*candidate.distance) {
                    continue;
                }
                // baseline across the ray: the longer, the more exact the distance
                const double bx = B.position.x - A.position.x, by = B.position.y - A.position.y, bz = B.position.z - A.position.z;
                const double along = dot(bx, by, bz, r.x, r.y, r.z);
                const double baseline = std::sqrt(std::max(0.0, bx*bx + by*by + bz*bz - along*along));
                if (baseline > bestBaseline) {
                    bestBaseline = baseline;
                    map.distance[i] = candidate.distance;
                    map.score[i] = static_cast<int8_t>(std::lround(std::max(0.0f, candidate.score)*STEREO_SCORE_SCALE));
                    map.unique[i] = static_cast<int8_t>(std::lround(std::min(1.0f, std::max(0.0f, candidate.unique))*STEREO_SCORE_SCALE));
                }
            }
        }
    }
    return result;
}

/*!
 * \brief StereoMatcher::save
 * file: char[4] "GEOM", uint32_t version, uint32_t width, uint32_t height, float distance (mm, 0: no match) of every
 * pixel row by row, int8_t score of every pixel (ZNCC*127), int8_t unique of every pixel (*127), native byte order
 */
bool StereoMatcher::save(const GeoMap &map, const std::string &filename)
{
    std::ofstream file(filename, std::ios::binary | std::ios::trunc);
    const uint32_t version = GEO_VERSION, width = static_cast<uint32_t>(map.width), height = static_cast<uint32_t>(map.height);
    file.write(GEO_MAGIC, 4);
    file.write(reinterpret_cast<const char *>(&version), sizeof(version));
    file.write(reinterpret_cast<const char *>(&width), sizeof(width));
    file.write(reinterpret_cast<const char *>(&height), sizeof(height));
    file.write(reinterpret_cast<const char *>(map.distance.data()), static_cast<std::streamsize>(map.distance.size()*sizeof(float)));
    file.write(reinterpret_cast<const char *>(map.score.data()), static_cast<std::streamsize>(map.score.size()));
    file.write(reinterpret_cast<const char *>(map.unique.data()), static_cast<std::streamsize>(map.unique.size()));
    if (!file) {
        TG_ERROR_LOG("Cannot write", filename);
        return false;
    }
    return true;
}

size_t StereoMatcher::getFileSize(int width, int height)
{
    return 4 + 3*sizeof(uint32_t) + static_cast<size_t>(width)*height*(sizeof(float) + 2*sizeof(int8_t));
}

/*!
 * \brief StereoMatcher::load
 * reads a file of save()
 */
bool StereoMatcher::load(const std::string &filename, GeoMap *map)
{
    std::ifstream file(filename, std::ios::binary);
    char magic[4];
    uint32_t version = 0, width = 0, height = 0;
    file.read(magic, 4);
    file.read(reinterpret_cast<char *>(&version), sizeof(version));
    file.read(reinterpret_cast<char *>(&width), sizeof(width));
    file.read(reinterpret_cast<char *>(&height), sizeof(height));
    if (!file || memcmp(magic, GEO_MAGIC, 4) != 0 || version != GEO_VERSION || width == 0 || height == 0
        || width > 100000 || height > 100000) {
        TG_ERROR_LOG("Cannot read", filename);
        return false;
    }
    const size_t pixels = static_cast<size_t>(width)*height;
    map->width = static_cast<int>(width);
    map->height = static_cast<int>(height);
    map->distance.resize(pixels);
    map->score.resize(pixels);
    map->unique.resize(pixels);
    file.read(reinterpret_cast<char *>(map->distance.data()), static_cast<std::streamsize>(pixels*sizeof(float)));
    file.read(reinterpret_cast<char *>(map->score.data()), static_cast<std::streamsize>(pixels));
    file.read(reinterpret_cast<char *>(map->unique.data()), static_cast<std::streamsize>(pixels));
    if (!file) {
        TG_ERROR_LOG(filename, "is too short");
        return false;
    }
    return true;
}
