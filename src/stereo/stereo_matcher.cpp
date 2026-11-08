#include "stereo_matcher.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <thread>
#include <opencv2/calib3d.hpp>
#include "global_log.h"

// distances along the ray: uniform in 1/distance from NEAR to FAR, the fine steps are half of the coarse steps
#define STEREO_NEAR_MM              400.0
#define STEREO_FAR_MM               80000.0
#define STEREO_COARSE_STEPS         160
#define STEREO_FINE_STEPS           (2*STEREO_COARSE_STEPS - 1)
// candidates of the coarse search that the full patch decides between, they are this many coarse steps apart
#define STEREO_CANDIDATES           3
#define STEREO_CANDIDATE_SEPARATION 3
// fine steps on both sides of a candidate
#define STEREO_FINE_RADIUS          3
// half width of the patch, the coarse search uses every second pixel of it
#define STEREO_PATCH_HALF           3
// patches with less variance have no texture (gray 0..1)
#define STEREO_MIN_VARIANCE         1e-4
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

namespace {
inline double dot(double ax, double ay, double az, double bx, double by, double bz)
{
    return ax*bx + ay*by + az*bz;
}
}

StereoMatcher::StereoMatcher()
{
}

/*!
 * \brief StereoMatcher::CameraData::project
 * world point (mm) to the pixel of the camera with the pinhole lens distortion of OpenCV or the fisheye model of OpenCV
 * \return false if the point is behind the pinhole camera or outside of the field of view of the fisheye camera
 */
bool StereoMatcher::CameraData::project(const Vector &point, double *u, double *v) const
{
    const double dx = point.x - position.x, dy = point.y - position.y, dz = point.z - position.z;
    const double z = dot(dx, dy, dz, forward.x, forward.y, forward.z);
    if (fisheye) {
        // image y goes down
        const double x = dot(dx, dy, dz, right.x, right.y, right.z);
        const double y = -dot(dx, dy, dz, up.x, up.y, up.z);
        const double radius = std::sqrt(x*x + y*y);
        if (radius*radius + z*z <= 1.0) {
            return false;
        }
        const double theta = std::atan2(radius, z);
        if (theta > maxTheta) {
            return false;
        }
        const double theta2 = theta*theta;
        const double thetaD = theta*(1.0 + theta2*(fisheyeK[0] + theta2*(fisheyeK[1] + theta2*(fisheyeK[2] + theta2*fisheyeK[3]))));
        const double scale = radius > 1e-12 ? thetaD/radius : 0.0;
        *u = fx*x*scale + cx;
        *v = fy*y*scale + cy;
        return true;
    }
    if (z <= 1.0) {
        return false;
    }
    // image y goes down
    const double x = dot(dx, dy, dz, right.x, right.y, right.z)/z;
    const double y = -dot(dx, dy, dz, up.x, up.y, up.z)/z;
    const double r2 = x*x + y*y;
    const double radial = (1.0 + r2*(k[0] + r2*(k[1] + r2*k[2])))/(1.0 + r2*(k[3] + r2*(k[4] + r2*k[5])));
    const double xd = x*radial + 2.0*p1*x*y + p2*(r2 + 2.0*x*x);
    const double yd = y*radial + p1*(r2 + 2.0*y*y) + 2.0*p2*x*y;
    *u = fx*xd + cx;
    *v = fy*yd + cy;
    return true;
}

/*!
 * \brief StereoMatcher::CameraData::isValid
 * \return true if the pixel is inside of the image and sees something
 */
bool StereoMatcher::CameraData::isValid(int x, int y) const
{
    return x >= 0 && y >= 0 && x < width && y < height && valid[static_cast<size_t>(y)*width + x];
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
    c.rays.resize(static_cast<size_t>(c.width)*c.height);
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
                c.valid[i] = 0;
                c.rays[i] = c.forward;
                continue;
            }
            // camera coordinates x right, y down, z forward
            const double sinTheta = std::sin(theta);
            const double rx = thetaD > 0.0 ? sinTheta*xd/thetaD : 0.0, ry = thetaD > 0.0 ? sinTheta*yd/thetaD : 0.0;
            const double rz = std::cos(theta);
            c.rays[i] = { c.right.x*rx - c.up.x*ry + c.forward.x*rz, c.right.y*rx - c.up.y*ry + c.forward.y*rz,
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
        c.valid.assign(pixelCount, 1);
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
        c.rays.resize(normalized.size());
        for (size_t j=0;j<normalized.size();j++) {
            const double nx = normalized[j].x, ny = normalized[j].y;
            c.rays[j] = normalize(c.forward.x + nx*c.right.x - ny*c.up.x, c.forward.y + nx*c.right.y - ny*c.up.y,
                                  c.forward.z + nx*c.right.z - ny*c.up.z);
        }
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
                        if (B.project({ A.position.x + r.x*t, A.position.y + r.y*t, A.position.z + r.z*t }, &u, &v)
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
 * \brief StereoMatcher::patchScore
 * ZNCC of the gray patch around pixel x, y of A and camera B, when the pixel is at the distance along its ray.
 * The patch is warped by planes through the point that face A, are horizontal or are tilted by 45 degrees (the coarse
 * search uses only the first two),
 * the best of them counts.
 * \param step 1: full 7x7 patch with all planes, 2: every second pixel of it with the planes that face A or are
 * horizontal
 * \return ZNCC -1..1, -2 if the patch is not inside both images or has no texture
 */
float StereoMatcher::patchScore(const CameraData &A, const CameraData &B, const std::vector<float> &grayA,
                                const std::vector<float> &grayB, int x, int y, double distance, int step) const
{
    const size_t i = static_cast<size_t>(y)*A.width + x;
    // the field of view of a fisheye is round, the patch is inside of it if its corners are
    if (A.fisheye && (!A.isValid(x - STEREO_PATCH_HALF, y - STEREO_PATCH_HALF) || !A.isValid(x + STEREO_PATCH_HALF, y - STEREO_PATCH_HALF)
                      || !A.isValid(x - STEREO_PATCH_HALF, y + STEREO_PATCH_HALF) || !A.isValid(x + STEREO_PATCH_HALF, y + STEREO_PATCH_HALF))) {
        return -2.0f;
    }
    const Vector &r = A.rays[i], &rx = A.rays[i + 1], &ry = A.rays[i + A.width];
    const Vector point = { A.position.x + r.x*distance, A.position.y + r.y*distance, A.position.z + r.z*distance };
    double u0, v0;
    if (!B.project(point, &u0, &v0) || u0 < -STEREO_PATCH_HALF*4 || v0 < -STEREO_PATCH_HALF*4
        || u0 > B.width + STEREO_PATCH_HALF*4 || v0 > B.height + STEREO_PATCH_HALF*4) {
        return -2.0f;
    }
    // patch of A
    float patchA[(2*STEREO_PATCH_HALF + 1)*(2*STEREO_PATCH_HALF + 1)];
    float patchB[(2*STEREO_PATCH_HALF + 1)*(2*STEREO_PATCH_HALF + 1)];
    int count = 0;
    double meanA = 0.0;
    for (int dy=-STEREO_PATCH_HALF;dy<=STEREO_PATCH_HALF;dy+=step) {
        for (int dx=-STEREO_PATCH_HALF;dx<=STEREO_PATCH_HALF;dx+=step) {
            patchA[count] = grayA[i + static_cast<ptrdiff_t>(dy)*A.width + dx];
            meanA += patchA[count++];
        }
    }
    meanA /= count;
    double varianceA = 0.0;
    for (int n=0;n<count;n++) {
        patchA[n] -= static_cast<float>(meanA);
        varianceA += patchA[n]*patchA[n];
    }
    if (varianceA < STEREO_MIN_VARIANCE*count/49.0) {
        return -2.0f;
    }
    const double c = std::sqrt(0.5);
    // the plane faces the camera: pinhole the direction of the camera, fisheye the ray of the pixel (it can be 90 degrees
    // from the direction), tilted around the right and up of the camera or of the ray
    Vector facing = A.forward, sideways = A.right, upwards = A.up;
    if (A.fisheye) {
        facing = r;
        // right of the ray: horizontal, perpendicular to the ray (the right of the camera if the ray is vertical)
        double sx = r.z, sz = -r.x;
        double length = std::sqrt(sx*sx + sz*sz);
        if (length < 0.1) {
            const double along = dot(A.right.x, A.right.y, A.right.z, r.x, r.y, r.z);
            sx = A.right.x - r.x*along;
            sz = A.right.z - r.z*along;
            const double sy = A.right.y - r.y*along;
            length = std::sqrt(sx*sx + sy*sy + sz*sz);
            sideways = { sx/length, sy/length, sz/length };
        } else {
            sideways = { sx/length, 0.0, sz/length };
        }
        // up of the ray = ray x right (left-handed world like setCameras())
        upwards = { r.y*sideways.z - r.z*sideways.y, r.z*sideways.x - r.x*sideways.z, r.x*sideways.y - r.y*sideways.x };
    }
    const Vector normals[] = {
        facing, { 0.0, 1.0, 0.0 },
        { (facing.x + sideways.x)*c, (facing.y + sideways.y)*c, (facing.z + sideways.z)*c },
        { (facing.x - sideways.x)*c, (facing.y - sideways.y)*c, (facing.z - sideways.z)*c },
        { (facing.x + upwards.x)*c, (facing.y + upwards.y)*c, (facing.z + upwards.z)*c },
        { (facing.x - upwards.x)*c, (facing.y - upwards.y)*c, (facing.z - upwards.z)*c } };
    const double px = point.x - A.position.x, py = point.y - A.position.y, pz = point.z - A.position.z;
    float best = -2.0f;
    // the coarse search uses only the planes that face A and the horizontal plane (floor), the full patch all of them
    const int normalCount = step == 1 ? static_cast<int>(sizeof(normals)/sizeof(normals[0])) : 2;
    for (int normalIndex=0;normalIndex<normalCount;normalIndex++) {
        const Vector &n = normals[normalIndex];
        // neighbour rays hit the plane through the point
        const double dr = dot(r.x, r.y, r.z, n.x, n.y, n.z), drx = dot(rx.x, rx.y, rx.z, n.x, n.y, n.z);
        const double dry = dot(ry.x, ry.y, ry.z, n.x, n.y, n.z), dp = dot(px, py, pz, n.x, n.y, n.z);
        if (std::fabs(dr) < 0.05 || std::fabs(drx) < 1e-6 || std::fabs(dry) < 1e-6 || dp/drx <= 0.0 || dp/dry <= 0.0) {
            continue;
        }
        const double tx = dp/drx, ty = dp/dry;
        double ux, vx, uy, vy;
        if (!B.project({ A.position.x + rx.x*tx, A.position.y + rx.y*tx, A.position.z + rx.z*tx }, &ux, &vx)
            || !B.project({ A.position.x + ry.x*ty, A.position.y + ry.y*ty, A.position.z + ry.z*ty }, &uy, &vy)) {
            continue;
        }
        const double jxu = ux - u0, jxv = vx - v0, jyu = uy - u0, jyv = vy - v0;
        bool inside = true;
        double meanB = 0.0;
        int n2 = 0;
        for (int dy=-STEREO_PATCH_HALF;dy<=STEREO_PATCH_HALF && inside;dy+=step) {
            for (int dx=-STEREO_PATCH_HALF;dx<=STEREO_PATCH_HALF;dx+=step) {
                const double u = u0 + dx*jxu + dy*jyu, v = v0 + dx*jxv + dy*jyv;
                const int x0 = static_cast<int>(std::floor(u)), y0 = static_cast<int>(std::floor(v));
                if (x0 < 0 || y0 < 0 || x0 + 1 >= B.width || y0 + 1 >= B.height) {
                    inside = false;
                    break;
                }
                const size_t j = static_cast<size_t>(y0)*B.width + x0;
                if (B.fisheye && (!B.valid[j] || !B.valid[j + 1] || !B.valid[j + B.width] || !B.valid[j + B.width + 1])) {
                    inside = false;
                    break;
                }
                const float fx = static_cast<float>(u - x0), fy = static_cast<float>(v - y0);
                patchB[n2] = (grayB[j]*(1.0f - fx) + grayB[j + 1]*fx)*(1.0f - fy)
                             + (grayB[j + B.width]*(1.0f - fx) + grayB[j + B.width + 1]*fx)*fy;
                meanB += patchB[n2++];
            }
        }
        if (!inside) {
            continue;
        }
        meanB /= count;
        double varianceB = 0.0, covariance = 0.0;
        for (int k=0;k<count;k++) {
            const double b = patchB[k] - meanB;
            varianceB += b*b;
            covariance += patchA[k]*b;
        }
        if (varianceB < STEREO_MIN_VARIANCE*count/49.0) {
            continue;
        }
        best = std::max(best, static_cast<float>(covariance/std::sqrt(varianceA*varianceB)));
    }
    return best;
}

/*!
 * \brief StereoMatcher::matchPair
 * best distance of every pixel of A in B: coarse search with every second pixel of the patch, the full patch decides
 * between the best coarse candidates on the fine steps around them, parabola between the fine steps
 */
void StereoMatcher::matchPair(int a, int b, const std::vector<std::vector<float>> &gray, std::vector<Candidate> *result) const
{
    const CameraData &A = m_cameras[a], &B = m_cameras[b];
    result->assign(static_cast<size_t>(A.width)*A.height, Candidate());
    auto inverseDistance = [](double fineIndex) {
        return 1.0/STEREO_NEAR_MM + (1.0/STEREO_FAR_MM - 1.0/STEREO_NEAR_MM)*fineIndex/(STEREO_FINE_STEPS - 1);
    };
    std::atomic<int> nextRow(STEREO_PATCH_HALF);
    auto work = [&]() {
        std::vector<float> coarse(STEREO_COARSE_STEPS);
        for (int y; (y = nextRow++) < A.height - STEREO_PATCH_HALF;) {
            for (int x=STEREO_PATCH_HALF;x<A.width - STEREO_PATCH_HALF;x++) {
                for (int k=0;k<STEREO_COARSE_STEPS;k++) {
                    coarse[k] = patchScore(A, B, gray[a], gray[b], x, y, 1.0/inverseDistance(2.0*k), 2);
                }
                // best coarse candidates that are apart
                int candidates[STEREO_CANDIDATES];
                int candidateCount = 0;
                for (int c=0;c<STEREO_CANDIDATES;c++) {
                    int best = -1;
                    for (int k=0;k<STEREO_COARSE_STEPS;k++) {
                        bool near = false;
                        for (int j=0;j<candidateCount;j++) {
                            near = near || std::abs(k - candidates[j]) <= STEREO_CANDIDATE_SEPARATION;
                        }
                        if (!near && coarse[k] > -1.5f && (best < 0 || coarse[k] > coarse[best])) {
                            best = k;
                        }
                    }
                    if (best < 0) {
                        break;
                    }
                    candidates[candidateCount++] = best;
                }
                if (candidateCount == 0) {
                    continue;
                }
                // full patch on the fine steps around every candidate
                float bestScore = -2.0f, secondScore = -1.0f;
                double bestIndex = -1.0;
                for (int c=0;c<candidateCount;c++) {
                    float fine[2*STEREO_FINE_RADIUS + 1];
                    int bestFine = -1;
                    for (int f=-STEREO_FINE_RADIUS;f<=STEREO_FINE_RADIUS;f++) {
                        const int index = 2*candidates[c] + f;
                        fine[f + STEREO_FINE_RADIUS] = index < 0 || index >= STEREO_FINE_STEPS ? -2.0f
                            : patchScore(A, B, gray[a], gray[b], x, y, 1.0/inverseDistance(index), 1);
                        if (fine[f + STEREO_FINE_RADIUS] > -1.5f
                            && (bestFine < 0 || fine[f + STEREO_FINE_RADIUS] > fine[bestFine])) {
                            bestFine = f + STEREO_FINE_RADIUS;
                        }
                    }
                    if (bestFine < 0) {
                        continue;
                    }
                    const float score = fine[bestFine];
                    double index = 2*candidates[c] + bestFine - STEREO_FINE_RADIUS;
                    if (bestFine > 0 && bestFine < 2*STEREO_FINE_RADIUS && fine[bestFine - 1] > -1.5f && fine[bestFine + 1] > -1.5f) {
                        const double c0 = fine[bestFine - 1], c1 = fine[bestFine], c2 = fine[bestFine + 1];
                        const double denominator = c0 - 2.0*c1 + c2;
                        if (denominator < 0.0) {
                            index += std::max(-0.5, std::min(0.5, 0.5*(c0 - c2)/denominator));
                        }
                    }
                    if (score > bestScore) {
                        secondScore = std::max(secondScore, bestScore);
                        bestScore = score;
                        bestIndex = index;
                    } else {
                        secondScore = std::max(secondScore, score);
                    }
                }
                if (bestIndex < 0.0) {
                    continue;
                }
                Candidate &candidate = (*result)[static_cast<size_t>(y)*A.width + x];
                candidate.distance = static_cast<float>(1.0/inverseDistance(bestIndex));
                candidate.score = bestScore;
                candidate.unique = bestScore - std::max(secondScore, -1.0f);
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
 * \brief StereoMatcher::match
 * \param rgbImages RGB image of every camera of setCameras()
 * \return distance, score and uniqueness of every pixel of every camera, from the camera that matches back and has
 * the longest baseline across the ray
 */
std::vector<StereoMatcher::GeoMap> StereoMatcher::match(const std::vector<const uint8_t *> &rgbImages) const
{
    const int count = static_cast<int>(m_cameras.size());
    std::vector<GeoMap> result(m_cameras.size());
    if (static_cast<int>(rgbImages.size()) != count) {
        TG_ERROR_LOG("Matching needs", count, "images, got", rgbImages.size());
        return std::vector<GeoMap>();
    }
    std::vector<std::vector<float>> gray(m_cameras.size());
    for (int c=0;c<count;c++) {
        const size_t pixels = static_cast<size_t>(m_cameras[c].width)*m_cameras[c].height;
        gray[c].resize(pixels);
        for (size_t i=0;i<pixels;i++) {
            const uint8_t *rgb = rgbImages[c] + i*3;
            gray[c][i] = (0.299f*rgb[0] + 0.587f*rgb[1] + 0.114f*rgb[2])/255.0f;
        }
    }
    std::vector<std::vector<std::vector<Candidate>>> pairs(m_cameras.size(), std::vector<std::vector<Candidate>>(m_cameras.size()));
    for (int a=0;a<count;a++) {
        for (int b=0;b<count;b++) {
            if (m_overlap[a][b]) {
                matchPair(a, b, gray, &pairs[a][b]);
            }
        }
    }
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
                if (!B.project(point, &u, &v)) {
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
