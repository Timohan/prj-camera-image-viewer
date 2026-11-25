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
// cross-camera loss of the joint model (CudaVisionTransformer): the predicted point of a distance pixel of camera A
// (along its ray) is projected into every other camera B of the scene, the predicted distance of B there (bilinear) must
// be the distance of the point from B (normalized logarithms, Huber loss). Only points that B sees are compared: the
// true point projected into B hits a pixel with the true distance within CROSS_CAMERA_VISIBLE (logarithm), and the four
// pixels of the bilinear sample are distance pixels without a depth jump between them (with the setting crossPredicted
// the predicted distances and sky decide instead, so it works also without true distances).
// The loss is multiplied by the weight of the training (crossWeight, default CROSS_CAMERA_LOSS_WEIGHT) for every pair of
// cameras before it is added to the distance loss.
#define CROSS_CAMERA_LOSS_WEIGHT    0.25f
#define CROSS_CAMERA_VISIBLE        0.02f

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

// object outputs of every pixel (VisionTransformer::Config::objectOutput): OBJECT_CLASS_COUNT logits of the class of
// the object of the pixel (ground, unknown object (cube, ball, cylinder, cone), sedan, SUV, sports car, man, woman,
// child, the sky has the sky output), then of a car or a person: the center of its box in the camera (x right, y down,
// z forward, in OBJECT_CENTER_SCALE m), its heading in the camera (cos and sin of the angle from the forward of the
// camera to the front of the object, positive toward x, around the y axis of the camera) and the logarithms of the
// sizes of its box (m). The box gives the type, the size, the place and the heading of the car or the person.
#define OBJECT_CLASS_COUNT      8
#define OBJECT_FIRST_BOX_CLASS  2       // classes from the sedan have the box values
#define OBJECT_CENTER           OBJECT_CLASS_COUNT
#define OBJECT_HEADING          (OBJECT_CENTER + 3)
#define OBJECT_SIZE             (OBJECT_HEADING + 2)
#define OBJECT_OUTPUT_COUNT     (OBJECT_SIZE + 3)
#define OBJECT_CENTER_SCALE     10.0f
// with the object ids (VisionTransformer::Config::objectIds) every pixel has more object outputs after them: of a car
// or a person the center of its box in the rig coordinates (x, y, z, in OBJECT_CENTER_SCALE m, worldToRig()) and its
// heading in them (cos and sin of the yaw of the objects file, 0 is forward +y, positive toward +x), the same for all
// cameras that see the object,
// then OBJECT_ID_SIZE values of the id of the object (of every object that is not the ground, also the cubes, balls,
// cylinders and cones): the pixels of the same object have near ids in all cameras of a scene and the means of the
// ids of different objects are far from each other (objectIdLoss of the training, not a loss of one pixel)
#define OBJECT_WORLD_CENTER     OBJECT_OUTPUT_COUNT
// rig coordinates of the world box: the origin is the center of the cameras (mean of the camera positions) on the floor,
// x right, y forward, z up (the floor is z = 0). The world of prj-camera-image-creator is x right, y up (the floor is
// y = 0), z forward. center is the mean x and z of the camera positions of the world, all values in mm.
inline void worldToRig(const double *world, double centerX, double centerZ, double *rig)
{
    rig[0] = world[0] - centerX;
    rig[1] = world[2] - centerZ;
    rig[2] = world[1];
}
inline void rigToWorld(const double *rig, double centerX, double centerZ, double *world)
{
    world[0] = rig[0] + centerX;
    world[1] = rig[2];
    world[2] = rig[1] + centerZ;
}
#define OBJECT_WORLD_HEADING    (OBJECT_WORLD_CENTER + 3)
#define OBJECT_ID               (OBJECT_WORLD_HEADING + 2)
#define OBJECT_ID_SIZE          8
#define OBJECT_ID_OUTPUT_COUNT  (OBJECT_ID + OBJECT_ID_SIZE)
// count of the object outputs of a pixel
PIXEL_LABEL_FUNCTION int objectOutputCount(bool objectIds)
{
    return objectIds ? OBJECT_ID_OUTPUT_COUNT : OBJECT_OUTPUT_COUNT;
}
// object target of every surface number of an image (ImageSamples::getObjectTable()): class (-1: no target, also
// the sky), center x/y/z, heading cos/sin, logarithms of the sizes x/y/z, like the outputs, then the world center x/y/z
// in the rig coordinates (worldToRig(), in OBJECT_CENTER_SCALE m), world heading cos/sin and 1 if the world values are targets (0 in a mirrored image)
#define OBJECT_TABLE_VALUES     15
#define OBJECT_TABLE_WORLD      9
#define OBJECT_TABLE_WORLD_VALID 14
#define OBJECT_TABLE_SIZE       256     // surface numbers of an image that have targets, larger ones have none
// cross-entropy of the class, Huber loss of the center and squared errors of the heading and the sizes are
// multiplied by these before they are added to the distance loss
#define OBJECT_CLASS_LOSS_WEIGHT    0.1f
#define OBJECT_CENTER_LOSS_WEIGHT   0.1f
#define OBJECT_HEADING_LOSS_WEIGHT  0.05f
#define OBJECT_SIZE_LOSS_WEIGHT     0.1f
#define OBJECT_CENTER_DELTA         0.1f    // Huber delta of the center, 0.1 is 1 m
// loss of the ids (objectIdLoss): the ids of a pixel are pulled to the mean of its object if they are farther than
// OBJECT_ID_PULL_MARGIN, the means of two objects of a scene are pushed apart if they are nearer than
// 2*OBJECT_ID_PUSH_MARGIN, every pixel has the loss of its pull and of the pushes of its object
#define OBJECT_ID_LOSS_WEIGHT       0.1f
#define OBJECT_ID_PULL_MARGIN       0.5f
#define OBJECT_ID_PUSH_MARGIN       1.5f

/*!
 * \brief objectTarget
 * \param label PixelLabel of the pixel
 * \param surface surface number of the pixel
 * \param table OBJECT_TABLE_SIZE x OBJECT_TABLE_VALUES object targets of the image, nullptr: no targets
 * \return the OBJECT_TABLE_VALUES target values of the pixel, nullptr if the pixel has no object target
 */
PIXEL_LABEL_FUNCTION const float *objectTarget(uint8_t label, int32_t surface, const float *table)
{
    if (!table || label != PixelDistance || surface < 0 || surface >= OBJECT_TABLE_SIZE) {
        return nullptr;
    }
    const float *target = table + surface*OBJECT_TABLE_VALUES;
    return target[0] >= 0.0f ? target : nullptr;
}

/*!
 * \brief objectPixelLoss
 * OBJECT_CLASS_LOSS_WEIGHT*cross-entropy of the class logits and of a car or a person also
 * OBJECT_CENTER_LOSS_WEIGHT*Huber loss of the center + OBJECT_HEADING_LOSS_WEIGHT*squared error of the heading
 * + OBJECT_SIZE_LOSS_WEIGHT*squared error of the sizes, with the object ids also of the world center and heading
 * (OBJECT_CENTER_LOSS_WEIGHT, OBJECT_HEADING_LOSS_WEIGHT) if they are targets; the ids have no loss here (objectIdLoss
 * of the training sees all pixels of the scene)
 * \param values objectOutputCount(objectIds) outputs of the pixel
 * \param target objectTarget() of the pixel, nullptr: nothing is learned
 * \param gradient returns objectOutputCount(objectIds) derivatives of the loss
 * \param objectIds true: the outputs have the world box and the ids
 * \return loss
 */
PIXEL_LABEL_FUNCTION float objectPixelLoss(const float *values, const float *target, float *gradient, bool objectIds = false)
{
    for (int i=0;i<objectOutputCount(objectIds);i++) {
        gradient[i] = 0.0f;
    }
    if (!target) {
        return 0.0f;
    }
    const int objectClass = static_cast<int>(target[0]);
    // softmax cross-entropy: loss = log(sum exp(z)) - z[class], its derivative is softmax(z) - one hot
    float largest = values[0];
    for (int i=1;i<OBJECT_CLASS_COUNT;i++) {
        largest = fmaxf(largest, values[i]);
    }
    float sum = 0.0f;
    for (int i=0;i<OBJECT_CLASS_COUNT;i++) {
        sum += expf(values[i] - largest);
    }
    float loss = OBJECT_CLASS_LOSS_WEIGHT*(logf(sum) + largest - values[objectClass]);
    for (int i=0;i<OBJECT_CLASS_COUNT;i++) {
        gradient[i] = OBJECT_CLASS_LOSS_WEIGHT*(expf(values[i] - largest)/sum - (i == objectClass ? 1.0f : 0.0f));
    }
    if (objectClass < OBJECT_FIRST_BOX_CLASS) {
        return loss;
    }
    for (int i=0;i<3;i++) {
        float g;
        loss += OBJECT_CENTER_LOSS_WEIGHT*distanceLoss(values[OBJECT_CENTER + i] - target[1 + i], OBJECT_CENTER_DELTA, &g);
        gradient[OBJECT_CENTER + i] = OBJECT_CENTER_LOSS_WEIGHT*g;
    }
    for (int i=0;i<2;i++) {
        const float error = values[OBJECT_HEADING + i] - target[4 + i];
        loss += OBJECT_HEADING_LOSS_WEIGHT*error*error;
        gradient[OBJECT_HEADING + i] = OBJECT_HEADING_LOSS_WEIGHT*2.0f*error;
    }
    for (int i=0;i<3;i++) {
        const float error = values[OBJECT_SIZE + i] - target[6 + i];
        loss += OBJECT_SIZE_LOSS_WEIGHT*error*error;
        gradient[OBJECT_SIZE + i] = OBJECT_SIZE_LOSS_WEIGHT*2.0f*error;
    }
    if (objectIds && target[OBJECT_TABLE_WORLD_VALID] > 0.5f) {
        for (int i=0;i<3;i++) {
            float g;
            loss += OBJECT_CENTER_LOSS_WEIGHT*distanceLoss(values[OBJECT_WORLD_CENTER + i] - target[OBJECT_TABLE_WORLD + i],
                                                           OBJECT_CENTER_DELTA, &g);
            gradient[OBJECT_WORLD_CENTER + i] = OBJECT_CENTER_LOSS_WEIGHT*g;
        }
        for (int i=0;i<2;i++) {
            const float error = values[OBJECT_WORLD_HEADING + i] - target[OBJECT_TABLE_WORLD + 3 + i];
            loss += OBJECT_HEADING_LOSS_WEIGHT*error*error;
            gradient[OBJECT_WORLD_HEADING + i] = OBJECT_HEADING_LOSS_WEIGHT*2.0f*error;
        }
    }
    return loss;
}

#endif // PIXEL_LABEL_H
