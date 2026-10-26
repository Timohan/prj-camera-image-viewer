#ifndef DISTANCE_MODEL_H
#define DISTANCE_MODEL_H

#include <memory>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "network/vision_transformer.h"

/*!
 * \brief DistanceModel
 * vision transformer of prj-camera-image-learner that predicts the distance of every pixel of an image and which pixels
 * are sky, the model must have the sky output
 */
class DistanceModel
{
public:
    DistanceModel();

    bool load(const std::string &filename);
    // distances (mm) and sky (1) or not (0) of every pixel of an image
    struct Prediction {
        std::vector<double> distances;
        std::vector<uint8_t> sky;
    };

    std::vector<Prediction> predict(const std::vector<const cv::Mat *> &images) const;

private:
    std::unique_ptr<VisionTransformer> m_network;
    float m_targetMean;     // distance in meters = exp(output*m_targetScale + m_targetMean)
    float m_targetScale;
};

#endif // DISTANCE_MODEL_H
