#include "distance_model.h"
#include <chrono>
#include <cmath>
#include <thread>
#include "global_log.h"
#include "network/pixel_label.h"
#include "camera_image.h"
#include "stereo/stereo_matcher.h"

#define MM_PER_METER        1000.0
// colors 0..1 are moved to -0.5..0.5 like in the training of prj-camera-image-learner (ImageSamples)
#define COLOR_OFFSET        0.5f

DistanceModel::DistanceModel() :
    m_targetMean(0.0f),
    m_targetScale(1.0f)
{
}

/*!
 * \brief DistanceModel::load
 * \param filename model file of prj-camera-image-learner
 * \return true on success
 */
bool DistanceModel::load(const std::string &filename)
{
    m_network = VisionTransformer::load(filename, &m_targetMean, &m_targetScale);
    if (!m_network) {
        return false;
    }
    const VisionTransformer::Config &config = m_network->getConfig();
    if (!config.skyOutput) {
        TG_ERROR_LOG(filename, "has no sky output, train a new model with prj-camera-image-learner");
        m_network.reset();
        return false;
    }
    TG_DEBUG_LOG("Model", filename + ":", std::to_string(config.width) + "x" + std::to_string(config.height), "pixels,",
                 std::to_string(config.patchSize) + "x" + std::to_string(config.patchSize), "tokens,",
                 m_network->getParameterCount(), "weights,",
                 config.surfaceOutput ? "edges of the surfaces from the model" : "edges from the distance jumps",
                 config.geoInputs > 0 ? ", distances also from the other cameras (matching)" : "");
    return true;
}

/*!
 * \brief DistanceModel::getCameras
 * \return cameras and lenses of the training images, empty if the model file has not them
 */
const std::vector<VisionTransformer::Camera> &DistanceModel::getCameras() const
{
    return m_network->getCameras();
}

/*!
 * \brief DistanceModel::predict
 * predicts the distances of the images, every image in its own thread. A model with the geo inputs gets also the
 * distances from the other cameras: the images are matched first like in the training (StereoMatcher, .geo files of
 * the learner), with the cameras of the model.
 * \param images RGB images of the cameras 1, 2, ... of the model, they must fit into the image size of the model
 * \return distance (mm), sky and connections of every pixel of every image, empty lists for an image that does not fit
 */
std::vector<DistanceModel::Prediction> DistanceModel::predict(const std::vector<const cv::Mat *> &images) const
{
    std::vector<Prediction> result(images.size());
    const VisionTransformer::Config &config = m_network->getConfig();
    std::vector<StereoMatcher::GeoMap> geo;
    if (config.geoInputs > 0) {
        // image i is from camera i + 1 of the model
        std::vector<VisionTransformer::Camera> cameras;
        std::vector<const uint8_t *> colors;
        for (size_t i=0;i<images.size();i++) {
            const VisionTransformer::Camera *found = nullptr;
            for (const VisionTransformer::Camera &camera : m_network->getCameras()) {
                if (camera.number == i + 1) {
                    found = &camera;
                }
            }
            const cv::Mat &image = *images[i];
            if (!found || image.type() != CV_8UC3 || !image.isContinuous() || image.cols != static_cast<int>(found->width)
                || image.rows != static_cast<int>(found->height)) {
                TG_ERROR_LOG("Image", i + 1, "is not of camera", i + 1, "of the model, the cameras cannot be matched");
                return result;
            }
            cameras.push_back(*found);
            colors.push_back(image.data);
        }
        StereoMatcher matcher;
        if (!matcher.setCameras(cameras)) {
            return result;
        }
        const auto start = std::chrono::steady_clock::now();
        geo = matcher.match(colors);
        if (geo.size() != images.size()) {
            return result;
        }
        TG_DEBUG_LOG("Cameras matched in", std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(), "s");
    }
    auto predictImage = [&](size_t index) {
        const cv::Mat &image = *images[index];
        if (image.cols > config.width || image.rows > config.height) {
            TG_ERROR_LOG("Image is", std::to_string(image.cols) + "x" + std::to_string(image.rows),
                         "but the model is for", std::to_string(config.width) + "x" + std::to_string(config.height));
            return;
        }
        // image is padded with 0 on the right and bottom to the size of the model
        const size_t pixels = static_cast<size_t>(config.width)*config.height;
        const int channels = config.inputChannels();
        std::vector<float> input(pixels*channels, 0.0f), output(pixels), skyOutput(pixels);
        std::vector<float> surfaceOutput(config.surfaceOutput ? pixels*SURFACE_OUTPUT_COUNT : 0);
        for (int y=0;y<image.rows;y++) {
            const cv::Vec3b *row = image.ptr<cv::Vec3b>(y);
            for (int x=0;x<image.cols;x++) {
                float *pixel = input.data() + (static_cast<size_t>(y)*config.width + x)*channels;
                // the images are RGB (CameraImage::load converts the BGR of OpenCV), like the training of the learner
                // (ImageSamples::getInput), so the colors are given in their own order
                for (int c=0;c<3;c++) {
                    pixel[c] = static_cast<float>(row[x][c])/255.0f - COLOR_OFFSET;
                }
                if (!geo.empty()) {
                    const size_t i = static_cast<size_t>(y)*image.cols + x;
                    VisionTransformer::geoInput(geo[index].distance[i], geo[index].score[i], geo[index].unique[i],
                                                m_targetMean, m_targetScale, pixel + 3);
                }
            }
        }
        VisionTransformer::Workspace workspace(*m_network);
        m_network->forward(&workspace, input.data(), output.data(), skyOutput.data(),
                           config.surfaceOutput ? surfaceOutput.data() : nullptr);
        Prediction &prediction = result[index];
        prediction.distances.resize(static_cast<size_t>(image.cols)*image.rows);
        prediction.sky.resize(prediction.distances.size());
        prediction.connections.resize(config.surfaceOutput ? prediction.distances.size() : 0);
        for (int y=0;y<image.rows;y++) {
            for (int x=0;x<image.cols;x++) {
                const size_t source = static_cast<size_t>(y)*config.width + x;
                const size_t pixel = static_cast<size_t>(y)*image.cols + x;
                prediction.distances[pixel] = std::exp(static_cast<double>(output[source]*m_targetScale + m_targetMean))*MM_PER_METER;
                // sky logit above 0 is probability above 0.5
                prediction.sky[pixel] = skyOutput[source] > 0.0f ? 1 : 0;
                if (config.surfaceOutput) {
                    // connection logit above 0 is the same surface
                    const float *surface = surfaceOutput.data() + source*SURFACE_OUTPUT_COUNT;
                    prediction.connections[pixel] = static_cast<uint8_t>((surface[SURFACE_RIGHT] > 0.0f ? CameraImage::ConnectedRight : 0)
                                                                         | (surface[SURFACE_DOWN] > 0.0f ? CameraImage::ConnectedDown : 0));
                }
            }
        }
    };
    std::vector<std::thread> threads;
    for (size_t i=0;i<images.size();i++) {
        threads.emplace_back(predictImage, i);
    }
    for (std::thread &thread : threads) {
        thread.join();
    }
    return result;
}
