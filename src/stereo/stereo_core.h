#ifndef STEREO_CORE_H
#define STEREO_CORE_H

/*!
 * Matching of one pixel of camera A in camera B (StereoMatcher): projection of the camera model, ZNCC of the warped
 * patch and the coarse and fine search along the ray of the pixel. The same code is compiled for the CPU
 * (StereoMatcher) and for CUDA (CudaStereoMatcher of prj-camera-image-learner), so both give the same distances.
 * Without CUDA (__CUDACC__ is not defined) this is plain C++.
 */
#include <cmath>
#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define STEREO_FUNCTION __host__ __device__ inline
#else
#define STEREO_FUNCTION inline
#endif

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

namespace stereo {

struct Vector { double x, y, z; };

// camera of the matching, rays and valid point to the data of every pixel
struct Camera {
    int width = 0;
    int height = 0;
    Vector position, forward, up, right;
    double fx, fy, cx, cy;
    double k[6];
    double p1, p2;
    bool fisheye = false;
    double fisheyeK[4];
    double maxTheta = 0.0;          // fisheye: largest angle from the direction of the camera (radians)
    const Vector *rays = nullptr;   // ray of every pixel center
    const uint8_t *valid = nullptr; // 1 if the pixel sees something (fisheye: inside of the field of view)
};

// best distance of a pixel of A in B
struct Candidate {
    float distance;     // mm, -1 if there is no match
    float score;        // ZNCC
    float unique;       // ZNCC - ZNCC of the next candidate
};

STEREO_FUNCTION double dot(double ax, double ay, double az, double bx, double by, double bz)
{
    return ax*bx + ay*by + az*bz;
}

/*!
 * world point (mm) to the pixel of the camera with the pinhole lens distortion of OpenCV or the fisheye model of OpenCV
 * \return false if the point is behind the pinhole camera or outside of the field of view of the fisheye camera
 */
STEREO_FUNCTION bool project(const Camera &c, const Vector &point, double *u, double *v)
{
    const double dx = point.x - c.position.x, dy = point.y - c.position.y, dz = point.z - c.position.z;
    const double z = dot(dx, dy, dz, c.forward.x, c.forward.y, c.forward.z);
    if (c.fisheye) {
        // image y goes down
        const double x = dot(dx, dy, dz, c.right.x, c.right.y, c.right.z);
        const double y = -dot(dx, dy, dz, c.up.x, c.up.y, c.up.z);
        const double radius = sqrt(x*x + y*y);
        if (radius*radius + z*z <= 1.0) {
            return false;
        }
        const double theta = atan2(radius, z);
        if (theta > c.maxTheta) {
            return false;
        }
        const double *k = c.fisheyeK;
        const double theta2 = theta*theta;
        const double thetaD = theta*(1.0 + theta2*(k[0] + theta2*(k[1] + theta2*(k[2] + theta2*k[3]))));
        const double scale = radius > 1e-12 ? thetaD/radius : 0.0;
        *u = c.fx*x*scale + c.cx;
        *v = c.fy*y*scale + c.cy;
        return true;
    }
    if (z <= 1.0) {
        return false;
    }
    // image y goes down
    const double x = dot(dx, dy, dz, c.right.x, c.right.y, c.right.z)/z;
    const double y = -dot(dx, dy, dz, c.up.x, c.up.y, c.up.z)/z;
    const double r2 = x*x + y*y;
    const double *k = c.k;
    const double radial = (1.0 + r2*(k[0] + r2*(k[1] + r2*k[2])))/(1.0 + r2*(k[3] + r2*(k[4] + r2*k[5])));
    const double xd = x*radial + 2.0*c.p1*x*y + c.p2*(r2 + 2.0*x*x);
    const double yd = y*radial + c.p1*(r2 + 2.0*y*y) + 2.0*c.p2*x*y;
    *u = c.fx*xd + c.cx;
    *v = c.fy*yd + c.cy;
    return true;
}

/*!
 * \return true if the pixel is inside of the image and sees something
 */
STEREO_FUNCTION bool isValid(const Camera &c, int x, int y)
{
    return x >= 0 && y >= 0 && x < c.width && y < c.height && c.valid[static_cast<size_t>(y)*c.width + x];
}

/*!
 * ZNCC of the gray patch around pixel x, y of A and camera B, when the pixel is at the distance along its ray.
 * The patch is warped by planes through the point that face A, are horizontal or are tilted by 45 degrees (the coarse
 * search uses only the first two), the best of them counts.
 * \param step 1: full 7x7 patch with all planes, 2: every second pixel of it with the planes that face A or are
 * horizontal
 * \return ZNCC -1..1, -2 if the patch is not inside both images or has no texture
 */
STEREO_FUNCTION float patchScore(const Camera &A, const Camera &B, const float *grayA, const float *grayB, int x, int y,
                                 double distance, int step)
{
    const size_t i = static_cast<size_t>(y)*A.width + x;
    // the field of view of a fisheye is round, the patch is inside of it if its corners are
    if (A.fisheye && (!isValid(A, x - STEREO_PATCH_HALF, y - STEREO_PATCH_HALF) || !isValid(A, x + STEREO_PATCH_HALF, y - STEREO_PATCH_HALF)
                      || !isValid(A, x - STEREO_PATCH_HALF, y + STEREO_PATCH_HALF) || !isValid(A, x + STEREO_PATCH_HALF, y + STEREO_PATCH_HALF))) {
        return -2.0f;
    }
    const Vector &r = A.rays[i], &rx = A.rays[i + 1], &ry = A.rays[i + A.width];
    const Vector point = { A.position.x + r.x*distance, A.position.y + r.y*distance, A.position.z + r.z*distance };
    double u0, v0;
    if (!project(B, point, &u0, &v0) || u0 < -STEREO_PATCH_HALF*4 || v0 < -STEREO_PATCH_HALF*4
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
    const double c = sqrt(0.5);
    // the plane faces the camera: pinhole the direction of the camera, fisheye the ray of the pixel (it can be 90 degrees
    // from the direction), tilted around the right and up of the camera or of the ray
    Vector facing = A.forward, sideways = A.right, upwards = A.up;
    if (A.fisheye) {
        facing = r;
        // right of the ray: horizontal, perpendicular to the ray (the right of the camera if the ray is vertical)
        double sx = r.z, sz = -r.x;
        double length = sqrt(sx*sx + sz*sz);
        if (length < 0.1) {
            const double along = dot(A.right.x, A.right.y, A.right.z, r.x, r.y, r.z);
            sx = A.right.x - r.x*along;
            sz = A.right.z - r.z*along;
            const double sy = A.right.y - r.y*along;
            length = sqrt(sx*sx + sy*sy + sz*sz);
            sideways = { sx/length, sy/length, sz/length };
        } else {
            sideways = { sx/length, 0.0, sz/length };
        }
        // up of the ray = ray x right (left-handed world like StereoMatcher::setCameras())
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
        if (fabs(dr) < 0.05 || fabs(drx) < 1e-6 || fabs(dry) < 1e-6 || dp/drx <= 0.0 || dp/dry <= 0.0) {
            continue;
        }
        const double tx = dp/drx, ty = dp/dry;
        double ux, vx, uy, vy;
        if (!project(B, { A.position.x + rx.x*tx, A.position.y + rx.y*tx, A.position.z + rx.z*tx }, &ux, &vx)
            || !project(B, { A.position.x + ry.x*ty, A.position.y + ry.y*ty, A.position.z + ry.z*ty }, &uy, &vy)) {
            continue;
        }
        const double jxu = ux - u0, jxv = vx - v0, jyu = uy - u0, jyv = vy - v0;
        bool inside = true;
        double meanB = 0.0;
        int n2 = 0;
        for (int dy=-STEREO_PATCH_HALF;dy<=STEREO_PATCH_HALF && inside;dy+=step) {
            for (int dx=-STEREO_PATCH_HALF;dx<=STEREO_PATCH_HALF;dx+=step) {
                const double u = u0 + dx*jxu + dy*jyu, v = v0 + dx*jxv + dy*jyv;
                const int x0 = static_cast<int>(floor(u)), y0 = static_cast<int>(floor(v));
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
        const float score = static_cast<float>(covariance/sqrt(varianceA*varianceB));
        best = score > best ? score : best;
    }
    return best;
}

// 1/distance (1/mm) of a fine step, coarse step k is fine step 2*k
STEREO_FUNCTION double inverseDistance(double fineIndex)
{
    return 1.0/STEREO_NEAR_MM + (1.0/STEREO_FAR_MM - 1.0/STEREO_NEAR_MM)*fineIndex/(STEREO_FINE_STEPS - 1);
}

/*!
 * best distance of pixel x, y of A in B: coarse search with every second pixel of the patch, the full patch decides
 * between the best coarse candidates on the fine steps around them, parabola between the fine steps
 * \return distance -1 if there is no match
 */
STEREO_FUNCTION Candidate matchPixel(const Camera &A, const Camera &B, const float *grayA, const float *grayB, int x, int y)
{
    Candidate result = { -1.0f, -2.0f, 0.0f };
    float coarse[STEREO_COARSE_STEPS];
    for (int k=0;k<STEREO_COARSE_STEPS;k++) {
        coarse[k] = patchScore(A, B, grayA, grayB, x, y, 1.0/inverseDistance(2.0*k), 2);
    }
    // best coarse candidates that are apart
    int candidates[STEREO_CANDIDATES];
    int candidateCount = 0;
    for (int c=0;c<STEREO_CANDIDATES;c++) {
        int best = -1;
        for (int k=0;k<STEREO_COARSE_STEPS;k++) {
            bool near = false;
            for (int j=0;j<candidateCount;j++) {
                const int separation = k > candidates[j] ? k - candidates[j] : candidates[j] - k;
                near = near || separation <= STEREO_CANDIDATE_SEPARATION;
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
        return result;
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
                : patchScore(A, B, grayA, grayB, x, y, 1.0/inverseDistance(index), 1);
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
                const double shift = 0.5*(c0 - c2)/denominator;
                index += shift < -0.5 ? -0.5 : (shift > 0.5 ? 0.5 : shift);
            }
        }
        if (score > bestScore) {
            secondScore = bestScore > secondScore ? bestScore : secondScore;
            bestScore = score;
            bestIndex = index;
        } else {
            secondScore = score > secondScore ? score : secondScore;
        }
    }
    if (bestIndex < 0.0) {
        return result;
    }
    result.distance = static_cast<float>(1.0/inverseDistance(bestIndex));
    result.score = bestScore;
    result.unique = bestScore - (secondScore > -1.0f ? secondScore : -1.0f);
    return result;
}

} // namespace stereo

#endif // STEREO_CORE_H
