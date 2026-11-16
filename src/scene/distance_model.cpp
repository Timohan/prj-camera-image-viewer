#include "distance_model.h"
#include <algorithm>
#include <chrono>
#include <functional>
#include <memory>
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
                 config.geoInputs > 0 ? ", distances also from the other cameras (matching)" : "",
                 config.objectOutput ? ", objects of the pixels (cars and people)" : "",
                 config.cameraGroup > 1 ? ", joint model of " + std::to_string(config.cameraGroup) + " cameras" : std::string());
    return true;
}

/*!
 * \brief DistanceModel::hasObjectOutput
 * \return true if the model predicts the objects of the pixels
 */
bool DistanceModel::hasObjectOutput() const
{
    return m_network && m_network->getConfig().objectOutput;
}

/*!
 * \brief DistanceModel::hasObjectIds
 * \return true if the model predicts also the world boxes and the ids of the objects
 */
bool DistanceModel::hasObjectIds() const
{
    return m_network && m_network->getConfig().objectIds;
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
 * the learner), with the cameras of the model. The joint model (cameraGroup > 1) predicts the images of all cameras at
 * once (VisionTransformer::forwardGroup()), the images must be the cameras 1, 2, ... of the model.
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
    for (const cv::Mat *image : images) {
        if (image->cols > config.width || image->rows > config.height) {
            TG_ERROR_LOG("Image is", std::to_string(image->cols) + "x" + std::to_string(image->rows),
                         "but the model is for", std::to_string(config.width) + "x" + std::to_string(config.height));
            return result;
        }
    }
    // the joint model predicts the images of all cameras of the scene together
    const bool joint = config.cameraGroup > 1;
    if (joint && images.size() != static_cast<size_t>(config.cameraGroup)) {
        TG_ERROR_LOG("The model sees", config.cameraGroup, "cameras at once, but the scene has", images.size(), "images");
        return result;
    }
    // network inputs and outputs of every image
    const size_t pixels = static_cast<size_t>(config.width)*config.height;
    const int channels = config.inputChannels();
    struct Buffers {
        std::vector<float> input, output, skyOutput, surfaceOutput, objectOutput;
    };
    std::vector<Buffers> buffers(images.size());
    auto prepareImage = [&](size_t index) {
        const cv::Mat &image = *images[index];
        // image is padded with 0 on the right and bottom to the size of the model
        Buffers &buffer = buffers[index];
        buffer.input.assign(pixels*channels, 0.0f);
        buffer.output.assign(pixels, 0.0f);
        buffer.skyOutput.assign(pixels, 0.0f);
        buffer.surfaceOutput.assign(config.surfaceOutput ? pixels*SURFACE_OUTPUT_COUNT : 0, 0.0f);
        buffer.objectOutput.assign(pixels*static_cast<size_t>(config.objectValues()), 0.0f);
        std::vector<float> &input = buffer.input;
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
    };
    auto convertImage = [&](size_t index) {
        const cv::Mat &image = *images[index];
        const std::vector<float> &output = buffers[index].output;
        const std::vector<float> &skyOutput = buffers[index].skyOutput;
        const std::vector<float> &surfaceOutput = buffers[index].surfaceOutput;
        const std::vector<float> &objectOutput = buffers[index].objectOutput;
        Prediction &prediction = result[index];
        prediction.distances.resize(static_cast<size_t>(image.cols)*image.rows);
        prediction.sky.resize(prediction.distances.size());
        prediction.connections.resize(config.surfaceOutput ? prediction.distances.size() : 0);
        prediction.objects.resize(config.objectOutput ? prediction.distances.size() : 0);
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
                if (config.objectOutput && !prediction.sky[pixel]) {
                    // class of the largest logit and its softmax probability, the box values like ImageSamples::getObjectTable()
                    const float *values = objectOutput.data() + source*static_cast<size_t>(config.objectValues());
                    const int best = static_cast<int>(std::max_element(values, values + OBJECT_CLASS_COUNT) - values);
                    double sum = 0.0;
                    for (int c=0;c<OBJECT_CLASS_COUNT;c++) {
                        sum += std::exp(static_cast<double>(values[c] - values[best]));
                    }
                    CameraImage::ObjectPixel &object = prediction.objects[pixel];
                    object.objectClass = static_cast<uint8_t>(best);
                    object.probability = static_cast<uint8_t>(std::lround(100.0/sum));
                    for (int k=0;k<3;k++) {
                        object.center[k] = values[OBJECT_CENTER + k]*OBJECT_CENTER_SCALE;
                        object.size[k] = std::exp(values[OBJECT_SIZE + k]);
                    }
                    const float length = std::hypot(values[OBJECT_HEADING], values[OBJECT_HEADING + 1]);
                    object.heading[0] = length > 0.0f ? values[OBJECT_HEADING]/length : 1.0f;
                    object.heading[1] = length > 0.0f ? values[OBJECT_HEADING + 1]/length : 0.0f;
                    if (config.objectIds) {
                        object.hasId = true;
                        for (int k=0;k<3;k++) {
                            object.worldCenter[k] = values[OBJECT_WORLD_CENTER + k]*OBJECT_CENTER_SCALE;
                        }
                        const float worldLength = std::hypot(values[OBJECT_WORLD_HEADING], values[OBJECT_WORLD_HEADING + 1]);
                        object.worldHeading[0] = worldLength > 0.0f ? values[OBJECT_WORLD_HEADING]/worldLength : 1.0f;
                        object.worldHeading[1] = worldLength > 0.0f ? values[OBJECT_WORLD_HEADING + 1]/worldLength : 0.0f;
                        std::copy_n(values + OBJECT_ID, OBJECT_ID_SIZE, object.id);
                    }
                }
            }
        }
    };
    auto runThreads = [&](const std::function<void(size_t)> &function) {
        std::vector<std::thread> threads;
        for (size_t i=0;i<images.size();i++) {
            threads.emplace_back(function, i);
        }
        for (std::thread &thread : threads) {
            thread.join();
        }
    };
    if (joint) {
        // the joint blocks of the model compare the tokens of all images, forwardGroup() uses all CPU threads
        runThreads(prepareImage);
        std::vector<std::unique_ptr<VisionTransformer::Workspace>> workspaceList;
        std::vector<VisionTransformer::Workspace *> workspaces;
        std::vector<const float *> inputs;
        VisionTransformer::GroupOutputs outputs;
        for (Buffers &buffer : buffers) {
            workspaceList.push_back(std::make_unique<VisionTransformer::Workspace>(*m_network));
            workspaces.push_back(workspaceList.back().get());
            inputs.push_back(buffer.input.data());
            outputs.output.push_back(buffer.output.data());
            outputs.skyOutput.push_back(buffer.skyOutput.data());
            outputs.surfaceOutput.push_back(config.surfaceOutput ? buffer.surfaceOutput.data() : nullptr);
            outputs.objectOutput.push_back(config.objectOutput ? buffer.objectOutput.data() : nullptr);
        }
        m_network->forwardGroup(workspaces, inputs, outputs,
                                static_cast<int>(std::max(1u, std::thread::hardware_concurrency())));
        runThreads(convertImage);
        return result;
    }
    // every image alone, every image in its own thread
    runThreads([&](size_t index) {
        prepareImage(index);
        Buffers &buffer = buffers[index];
        VisionTransformer::Workspace workspace(*m_network);
        m_network->forward(&workspace, buffer.input.data(), buffer.output.data(), buffer.skyOutput.data(),
                           config.surfaceOutput ? buffer.surfaceOutput.data() : nullptr,
                           config.objectOutput ? buffer.objectOutput.data() : nullptr);
        convertImage(index);
        // the buffers of the image are not needed anymore
        buffer = Buffers();
    });
    return result;
}
