#ifndef PIXEL_LABEL_H
#define PIXEL_LABEL_H

#include <cmath>
#include <cstdint>

/*!
 * \brief PixelLabel
 * what the training knows about a pixel: its distance is in the distance loss, the sky output learns
 * sky against not sky from every pixel that is not ignored (padding and pixels that see nothing are ignored)
 */
enum PixelLabel : uint8_t {
    PixelIgnore = 0,
    PixelDistance = 1,
    PixelSky = 2
};

// sky loss (binary cross-entropy of the sky output) is multiplied by this before it is added to the distance loss
#define SKY_LOSS_WEIGHT     1.0f

// surface outputs of every pixel: normal x/y/z of the surface in the camera (x right, y down, z forward, the normal is
// turned toward the camera) and the logits that the pixel is on the same surface (object) as its right neighbour and
// as its lower neighbour. Neighbours on the same surface can be joined to triangles, others are edges.
#define SURFACE_OUTPUT_COUNT    5
#define SURFACE_RIGHT           3
#define SURFACE_DOWN            4
// squared error of the normal and binary cross-entropy of the connections are multiplied by these
// before they are added to the distance loss
#define NORMAL_LOSS_WEIGHT      0.1f
#define CONNECTION_LOSS_WEIGHT  0.1f
// connection target of a pixel that has no target
#define CONNECTION_IGNORE       -1

// the same functions are used on the CPU and in the CUDA kernels
#ifdef __CUDACC__
#define PIXEL_LABEL_FUNCTION    __host__ __device__ inline
#else
#define PIXEL_LABEL_FUNCTION    inline
#endif

/*!
 * \brief connectionTarget
 * \param label, neighbourLabel PixelLabel of the pixel and of its neighbour, PixelIgnore if the neighbour is outside
 * \param surface, neighbourSurface surface numbers of the creator
 * \return 1 if the pixel and its neighbour are on the same surface, 0 if the neighbour is another surface or the sky,
 * CONNECTION_IGNORE if the pixel has no surface or the neighbour is not known
 */
PIXEL_LABEL_FUNCTION int connectionTarget(uint8_t label, uint8_t neighbourLabel, int32_t surface, int32_t neighbourSurface)
{
    if (label != PixelDistance || neighbourLabel == PixelIgnore) {
        return CONNECTION_IGNORE;
    }
    return neighbourLabel == PixelDistance && surface == neighbourSurface ? 1 : 0;
}

/*!
 * \brief surfacePixelLoss
 * NORMAL_LOSS_WEIGHT*squared error of the normal + CONNECTION_LOSS_WEIGHT*binary cross-entropy of the connections
 * \param values SURFACE_OUTPUT_COUNT outputs of the pixel
 * \param normal correct normal of the pixel, nullptr if the pixel has no surface (only the connections are learned)
 * \param right, down connectionTarget() of the right and the lower neighbour
 * \param gradient returns SURFACE_OUTPUT_COUNT derivatives of the loss
 * \return loss
 */
PIXEL_LABEL_FUNCTION float surfacePixelLoss(const float *values, const float *normal, int right, int down, float *gradient)
{
    float loss = 0.0f;
    for (int i=0;i<3;i++) {
        const float error = normal ? values[i] - normal[i] : 0.0f;
        loss += NORMAL_LOSS_WEIGHT*error*error;
        gradient[i] = NORMAL_LOSS_WEIGHT*2.0f*error;
    }
    const int targets[2] = { right, down };
    for (int i=0;i<2;i++) {
        const int output = i == 0 ? SURFACE_RIGHT : SURFACE_DOWN;
        if (targets[i] == CONNECTION_IGNORE) {
            gradient[output] = 0.0f;
            continue;
        }
        // loss = log(1 + exp(z)) - y*z, its derivative is sigmoid(z) - y
        const float z = values[output];
        const float y = static_cast<float>(targets[i]);
        loss += CONNECTION_LOSS_WEIGHT*(fmaxf(z, 0.0f) - z*y + log1pf(expf(-fabsf(z))));
        gradient[output] = CONNECTION_LOSS_WEIGHT*(1.0f/(1.0f + expf(-z)) - y);
    }
    return loss;
}

// distance loss of the pixels at most EDGE_RADIUS pixels (also diagonally) from a depth jump is multiplied by the edge
// weight (VisionTransformer::setEdgeWeight()), most of the distance error is at the edges of the objects. A depth jump
// is between two neighbour PixelDistance pixels whose distances differ more than EDGE_JUMP (relative).
#define EDGE_RADIUS             3
#define EDGE_JUMP               0.15f

/*!
 * \brief edgePixelWeight
 * \param target height x width targets (normalized logarithms of the distances)
 * \param labels height x width PixelLabel values
 * \param x, y pixel
 * \param jump logarithm of 1 + EDGE_JUMP in the units of the targets
 * \param weight weight of the edge pixels
 * \return weight if the pixel is at most EDGE_RADIUS pixels from a pixel of a depth jump, else 1
 */
PIXEL_LABEL_FUNCTION float edgePixelWeight(const float *target, const uint8_t *labels, int width, int height, int x, int y,
                                           float jump, float weight)
{
    if (weight == 1.0f) {
        return 1.0f;
    }
    const int top = y > EDGE_RADIUS ? y - EDGE_RADIUS : 0;
    const int bottom = y + EDGE_RADIUS < height ? y + EDGE_RADIUS : height - 1;
    const int left = x > EDGE_RADIUS ? x - EDGE_RADIUS : 0;
    const int right = x + EDGE_RADIUS < width ? x + EDGE_RADIUS : width - 1;
    for (int ny=top;ny<=bottom;ny++) {
        for (int nx=left;nx<=right;nx++) {
            const int i = ny*width + nx;
            if (labels[i] != PixelDistance) {
                continue;
            }
            // the neighbours on the left, right, above and below
            const int neighbours[4] = { nx > 0 ? i - 1 : -1, nx + 1 < width ? i + 1 : -1,
                                        ny > 0 ? i - width : -1, ny + 1 < height ? i + width : -1 };
            for (int n=0;n<4;n++) {
                if (neighbours[n] >= 0 && labels[neighbours[n]] == PixelDistance
                    && fabsf(target[i] - target[neighbours[n]]) > jump) {
                    return weight;
                }
            }
        }
    }
    return 1.0f;
}

// two distances of a pixel (VisionTransformer::Config::twoDistances): the near distance is the first output and the far
// distance is near + softplus(second output), so the far one is never nearer. The loss of the one that is nearer to the
// target is learned fully and the other one is multiplied by OTHER_DISTANCE_WEIGHT, so at an edge the near distance can
// learn the near surface and the far distance the far surface instead of their mean, the choice logit learns which
// surface the pixel is on
#define OTHER_DISTANCE_WEIGHT   0.2f
// binary cross-entropy of the choice is multiplied by this and by min(1, (far - near)/CHOICE_DIFFERENCE),
// the choice does not matter when both distances are the same
#define CHOICE_LOSS_WEIGHT      0.1f
#define CHOICE_DIFFERENCE       0.05f
// starting bias of the gap output, the far distance starts softplus(-2) = 0.13 farther than the near distance
#define FAR_GAP_INIT            -2.0f

/*!
 * \brief distanceLoss
 * \param error output - target
 * \param delta 0: squared error, > 0: Huber loss, error^2 up to delta, then delta*(2*|error| - delta)
 * \param gradient returns the derivative of the loss
 * \return loss
 */
PIXEL_LABEL_FUNCTION float distanceLoss(float error, float delta, float *gradient)
{
    if (delta <= 0.0f || fabsf(error) <= delta) {
        *gradient = 2.0f*error;
        return error*error;
    }
    *gradient = error > 0.0f ? 2.0f*delta : -2.0f*delta;
    return delta*(2.0f*fabsf(error) - delta);
}

/*!
 * \brief farDistance
 * \param near the near distance (first output)
 * \param gap second output, far - near = softplus(gap) = log(1 + exp(gap)) > 0
 */
PIXEL_LABEL_FUNCTION float farDistance(float near, float gap)
{
    return near + fmaxf(gap, 0.0f) + log1pf(expf(-fabsf(gap)));
}

/*!
 * \brief chosenDistance
 * \return near distance if the choice logit is above 0 (probability above 0.5), else the far distance
 */
PIXEL_LABEL_FUNCTION float chosenDistance(float near, float gap, float choice)
{
    return choice > 0.0f ? near : farDistance(near, gap);
}

/*!
 * \brief twoDistanceLoss
 * loss of the two distances and the choice logit of a PixelDistance pixel: distanceLoss() of the distance that is
 * nearer to the target, OTHER_DISTANCE_WEIGHT*distanceLoss() of the other one and the weighted binary cross-entropy
 * of the choice
 * \param near, gap first and second output, see farDistance()
 * \param choice logit that the near distance is right
 * \param target correct distance
 * \param delta Huber delta of distanceLoss()
 * \param gradient returns the derivatives of the loss by near, gap and choice
 * \return loss
 */
PIXEL_LABEL_FUNCTION float twoDistanceLoss(float near, float gap, float choice, float target, float delta, float *gradient)
{
    const float far = farDistance(near, gap);
    float nearGradient, farGradient;
    const float nearLoss = distanceLoss(near - target, delta, &nearGradient);
    const float farLoss = distanceLoss(far - target, delta, &farGradient);
    const bool nearRight = nearLoss <= farLoss;
    float loss;
    if (nearRight) {
        loss = nearLoss + OTHER_DISTANCE_WEIGHT*farLoss;
        farGradient *= OTHER_DISTANCE_WEIGHT;
    } else {
        loss = farLoss + OTHER_DISTANCE_WEIGHT*nearLoss;
        nearGradient *= OTHER_DISTANCE_WEIGHT;
    }
    // far = near + softplus(gap), the derivative of softplus is sigmoid
    gradient[0] = nearGradient + farGradient;
    gradient[1] = farGradient/(1.0f + expf(-gap));
    // loss = log(1 + exp(z)) - y*z, its derivative is sigmoid(z) - y, the weight has no gradient
    const float weight = CHOICE_LOSS_WEIGHT*fminf(1.0f, (far - near)/CHOICE_DIFFERENCE);
    const float y = nearRight ? 1.0f : 0.0f;
    loss += weight*(fmaxf(choice, 0.0f) - choice*y + log1pf(expf(-fabsf(choice))));
    gradient[2] = weight*(1.0f/(1.0f + expf(-choice)) - y);
    return loss;
}

#endif // PIXEL_LABEL_H
