#include "vision_transformer.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include "global_log.h"
#include "pixel_label.h"

// Adam optimizer constants (Kingma & Ba 2014)
#define ADAM_BETA1          0.9f
#define ADAM_BETA2          0.999f
#define ADAM_EPSILON        1e-8f
#define LAYER_NORM_EPSILON  1e-5f
// standard deviation of the starting position embeddings
#define POSITION_INIT_SCALE 0.02f
// GELU tanh approximation: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
#define GELU_SCALE          0.7978845608f
#define GELU_CUBIC          0.044715f
// model file
#define MODEL_FILE_MAGIC    "VITM"
// version 2 has the sky output flag, version 1 files have no sky output,
// version 3 has the pixel head size (pixelHidden, pixelWindow), version 4 has the surface output flag,
// version 5 has the refine step size (refineHidden, refineWindow), version 6 has the two distances flag,
// version 7 has the cameras after the weight tensors
#define MODEL_FILE_VERSION  7
// a model file cannot have more cameras than this
#define MODEL_MAX_CAMERAS   1000

static float gelu(float x)
{
    return 0.5f*x*(1.0f + std::tanh(GELU_SCALE*(x + GELU_CUBIC*x*x*x)));
}

static float geluDerivative(float x)
{
    const float t = std::tanh(GELU_SCALE*(x + GELU_CUBIC*x*x*x));
    return 0.5f*(1.0f + t) + 0.5f*x*(1.0f - t*t)*GELU_SCALE*(1.0f + 3.0f*GELU_CUBIC*x*x);
}

/*!
 * \brief VisionTransformer::VisionTransformer
 * \param config image size, patch size and network size
 * \param seed seed of the random starting weights, the same seed gives the same network
 */
VisionTransformer::VisionTransformer(const Config &config, unsigned int seed) :
    m_config(config),
    m_tokenCount((config.width/config.patchSize)*(config.height/config.patchSize)),
    m_patchInputCount(config.patchSize*config.patchSize*3),
    m_patchOutputCount(config.patchSize*config.patchSize),
    m_headOutputCount(config.pixelHidden > 0 ? m_patchOutputCount*config.pixelHidden
                                             : (config.skyOutput ? 2*m_patchOutputCount : m_patchOutputCount)),
    m_pixelInputCount(config.pixelWindow*config.pixelWindow*3),
    m_pixelOutputCount((config.skyOutput ? 2 : 1) + (config.surfaceOutput ? SURFACE_OUTPUT_COUNT : 0)
                       + (config.twoDistances ? 2 : 0)),
    m_refineInputCount(config.refineWindow*config.refineWindow*config.pixelHidden),
    m_pixelWeight(-1), m_pixelBias(-1), m_pixelOutWeight(-1), m_pixelOutBias(-1),
    m_refineWeight(-1), m_refineBias(-1), m_refineOutWeight(-1), m_refineOutBias(-1),
    m_step(0),
    m_huberDelta(0.0f),
    m_edgeWeight(1.0f),
    m_edgeJump(0.0f)
{
    std::mt19937 random(seed);
    const int dimension = config.dimension;
    m_patchWeight = addLinearWeight(m_patchInputCount, dimension, &random);
    m_patchBias = addParameter(dimension, 0.0f);
    // uniform -a..a has standard deviation a/sqrt(3)
    m_position = addRandomParameter(m_tokenCount*dimension, POSITION_INIT_SCALE*std::sqrt(3.0f), &random);
    for (int l=0;l<config.layers;l++) {
        LayerParameters layer;
        layer.norm1Gain = addParameter(dimension, 1.0f);
        layer.norm1Bias = addParameter(dimension, 0.0f);
        layer.qkvWeight = addLinearWeight(dimension, 3*dimension, &random);
        layer.qkvBias = addParameter(3*dimension, 0.0f);
        layer.outWeight = addLinearWeight(dimension, dimension, &random);
        layer.outBias = addParameter(dimension, 0.0f);
        layer.norm2Gain = addParameter(dimension, 1.0f);
        layer.norm2Bias = addParameter(dimension, 0.0f);
        layer.mlp1Weight = addLinearWeight(dimension, config.mlpDimension, &random);
        layer.mlp1Bias = addParameter(config.mlpDimension, 0.0f);
        layer.mlp2Weight = addLinearWeight(config.mlpDimension, dimension, &random);
        layer.mlp2Bias = addParameter(dimension, 0.0f);
        m_layerParameters.push_back(layer);
    }
    m_normFinalGain = addParameter(dimension, 1.0f);
    m_normFinalBias = addParameter(dimension, 0.0f);
    m_headWeight = addLinearWeight(dimension, m_headOutputCount, &random);
    m_headBias = addParameter(m_headOutputCount, 0.0f);
    if (config.pixelHidden > 0) {
        m_pixelWeight = addLinearWeight(m_pixelInputCount, config.pixelHidden, &random);
        m_pixelBias = addParameter(config.pixelHidden, 0.0f);
        m_pixelOutWeight = addLinearWeight(config.pixelHidden, m_pixelOutputCount, &random);
        m_pixelOutBias = addParameter(m_pixelOutputCount, 0.0f);
        if (config.twoDistances) {
            // the far distance starts near the near distance, so the network starts like the one distance network
            m_parameters[static_cast<size_t>(m_pixelOutBias)].value[static_cast<size_t>(m_pixelOutputCount - 2)] = FAR_GAP_INIT;
        }
    }
    if (config.pixelHidden > 0 && config.refineHidden > 0) {
        m_refineWeight = addLinearWeight(m_refineInputCount, config.refineHidden, &random);
        m_refineBias = addParameter(config.refineHidden, 0.0f);
        // the refine step starts as no correction
        m_refineOutWeight = addParameter(config.refineHidden*m_pixelOutputCount, 0.0f);
        m_refineOutBias = addParameter(m_pixelOutputCount, 0.0f);
    }
}

const VisionTransformer::Config &VisionTransformer::getConfig() const
{
    return m_config;
}

int VisionTransformer::getParameterCount() const
{
    size_t count = 0;
    for (const Parameter &parameter : m_parameters) {
        count += parameter.value.size();
    }
    return static_cast<int>(count);
}

int VisionTransformer::getParameterTensorCount() const
{
    return static_cast<int>(m_parameters.size());
}

std::vector<float> &VisionTransformer::getParameter(int parameter)
{
    return m_parameters[static_cast<size_t>(parameter)].value;
}

int VisionTransformer::addParameter(int size, float initValue)
{
    Parameter parameter;
    parameter.value.assign(static_cast<size_t>(size), initValue);
    parameter.moment1.assign(static_cast<size_t>(size), 0.0f);
    parameter.moment2.assign(static_cast<size_t>(size), 0.0f);
    m_parameters.push_back(parameter);
    return static_cast<int>(m_parameters.size()) - 1;
}

int VisionTransformer::addRandomParameter(int size, float limit, std::mt19937 *random)
{
    const int index = addParameter(size, 0.0f);
    std::uniform_real_distribution<float> distribution(-limit, limit);
    for (float &value : m_parameters[static_cast<size_t>(index)].value) {
        value = distribution(*random);
    }
    return index;
}

/*!
 * \brief VisionTransformer::addLinearWeight
 * Xavier/Glorot uniform weights, so the outputs of the layer are neither too small nor too large
 */
int VisionTransformer::addLinearWeight(int inputCount, int outputCount, std::mt19937 *random)
{
    return addRandomParameter(inputCount*outputCount, std::sqrt(6.0f/static_cast<float>(inputCount + outputCount)), random);
}

const float *VisionTransformer::value(int parameter) const
{
    return m_parameters[static_cast<size_t>(parameter)].value.data();
}

/*!
 * \brief VisionTransformer::forward
 * \param workspace keeps the values that backward() needs
 * \param image height x width x 3 colors row by row
 * \param output returns height x width values
 * \param skyOutput if not nullptr and the network has the sky output, returns height x width sky logits
 * \param surfaceOutput if not nullptr and the network has the surface output, returns height x width x SURFACE_OUTPUT_COUNT
 * surface values (pixel_label.h)
 */
void VisionTransformer::forward(Workspace *workspace, const float *image, float *output, float *skyOutput,
                                float *surfaceOutput) const
{
    const int patchSize = m_config.patchSize;
    const int columns = m_config.width/patchSize;
    const int dimension = m_config.dimension;
    const int count = m_tokenCount;

    // tokens: colors of the patches and their positions
    for (int t=0;t<count;t++) {
        float *patch = workspace->m_patches.data() + t*m_patchInputCount;
        const int left = (t%columns)*patchSize;
        const int top = (t/columns)*patchSize;
        for (int py=0;py<patchSize;py++) {
            const float *row = image + (static_cast<size_t>(top + py)*static_cast<size_t>(m_config.width) + static_cast<size_t>(left))*3;
            std::copy(row, row + patchSize*3, patch + py*patchSize*3);
        }
    }
    float *x = workspace->m_layers.empty() ? workspace->m_output.data() : workspace->m_layers[0].input.data();
    linearForward(workspace->m_patches.data(), count, m_patchInputCount, value(m_patchWeight), value(m_patchBias), dimension, x);
    const float *position = value(m_position);
    for (int i=0;i<count*dimension;i++) {
        x[i] += position[i];
    }

    for (size_t l=0;l<m_layerParameters.size();l++) {
        const LayerParameters &parameters = m_layerParameters[l];
        Workspace::Layer &layer = workspace->m_layers[l];
        layerNormForward(layer.input.data(), count, dimension, value(parameters.norm1Gain), value(parameters.norm1Bias),
                         layer.norm1.data(), layer.rstd1.data(), layer.y1.data());
        linearForward(layer.y1.data(), count, dimension, value(parameters.qkvWeight), value(parameters.qkvBias),
                      3*dimension, layer.qkv.data());
        attentionForward(&layer);
        linearForward(layer.attention.data(), count, dimension, value(parameters.outWeight), value(parameters.outBias),
                      dimension, layer.x1.data());
        for (int i=0;i<count*dimension;i++) {
            layer.x1[static_cast<size_t>(i)] += layer.input[static_cast<size_t>(i)];
        }
        layerNormForward(layer.x1.data(), count, dimension, value(parameters.norm2Gain), value(parameters.norm2Bias),
                         layer.norm2.data(), layer.rstd2.data(), layer.y2.data());
        linearForward(layer.y2.data(), count, dimension, value(parameters.mlp1Weight), value(parameters.mlp1Bias),
                      m_config.mlpDimension, layer.hidden.data());
        for (size_t i=0;i<layer.hidden.size();i++) {
            layer.activated[i] = gelu(layer.hidden[i]);
        }
        float *next = l + 1 < m_layerParameters.size() ? workspace->m_layers[l + 1].input.data() : workspace->m_output.data();
        linearForward(layer.activated.data(), count, m_config.mlpDimension, value(parameters.mlp2Weight),
                      value(parameters.mlp2Bias), dimension, next);
        for (int i=0;i<count*dimension;i++) {
            next[i] += layer.x1[static_cast<size_t>(i)];
        }
    }

    layerNormForward(workspace->m_output.data(), count, dimension, value(m_normFinalGain), value(m_normFinalBias),
                     workspace->m_normFinal.data(), workspace->m_rstdFinal.data(), workspace->m_yFinal.data());
    linearForward(workspace->m_yFinal.data(), count, dimension, value(m_headWeight), value(m_headBias),
                  m_headOutputCount, workspace->m_head.data());
    const bool sky = m_config.skyOutput && skyOutput;
    const bool surface = m_config.surfaceOutput && surfaceOutput;
    if (m_config.pixelHidden > 0) {
        // pixel head: hidden = token part + linear(colors around the pixel), output = linear(gelu(hidden))
        const int rows = count*m_patchOutputCount;
        pixelWindows(image, workspace->m_pixelInputs.data());
        linearForward(workspace->m_pixelInputs.data(), rows, m_pixelInputCount, value(m_pixelWeight), value(m_pixelBias),
                      m_config.pixelHidden, workspace->m_pixelHidden.data());
        for (size_t i=0;i<workspace->m_pixelHidden.size();i++) {
            workspace->m_pixelHidden[i] += workspace->m_head[i];
            workspace->m_pixelActivated[i] = gelu(workspace->m_pixelHidden[i]);
        }
        linearForward(workspace->m_pixelActivated.data(), rows, m_config.pixelHidden, value(m_pixelOutWeight),
                      value(m_pixelOutBias), m_pixelOutputCount, workspace->m_pixelOutput.data());
        if (m_refineWeight >= 0) {
            // refine step: output += linear(gelu(linear(activated hidden values of the pixels around the pixel)))
            refineWindows(workspace->m_pixelActivated.data(), workspace->m_refineInputs.data());
            linearForward(workspace->m_refineInputs.data(), rows, m_refineInputCount, value(m_refineWeight),
                          value(m_refineBias), m_config.refineHidden, workspace->m_refineHidden.data());
            for (size_t i=0;i<workspace->m_refineHidden.size();i++) {
                workspace->m_refineActivated[i] = gelu(workspace->m_refineHidden[i]);
            }
            linearForward(workspace->m_refineActivated.data(), rows, m_config.refineHidden, value(m_refineOutWeight),
                          value(m_refineOutBias), m_pixelOutputCount, workspace->m_refineOutput.data());
            for (size_t i=0;i<workspace->m_pixelOutput.size();i++) {
                workspace->m_pixelOutput[i] += workspace->m_refineOutput[i];
            }
        }
    }
    for (int t=0;t<count;t++) {
        const int left = (t%columns)*patchSize;
        const int top = (t/columns)*patchSize;
        for (int k=0;k<m_patchOutputCount;k++) {
            const size_t pixel = static_cast<size_t>(top + k/patchSize)*static_cast<size_t>(m_config.width)
                                 + static_cast<size_t>(left + k%patchSize);
            const float *values = headValues(workspace->m_head, workspace->m_pixelOutput, t, k);
            output[pixel] = m_config.twoDistances ? chosenDistance(values[0], values[m_pixelOutputCount - 2],
                                                                   values[m_pixelOutputCount - 1]) : values[0];
            if (sky) {
                skyOutput[pixel] = values[skyOffset()];
            }
            if (surface) {
                std::copy_n(values + m_surfaceOffset, SURFACE_OUTPUT_COUNT, surfaceOutput + pixel*SURFACE_OUTPUT_COUNT);
            }
        }
    }
}

/*!
 * \brief VisionTransformer::headValues
 * \return distance value of the pixel k of the token t, values[skyOffset()] is its sky value
 */
const float *VisionTransformer::headValues(const std::vector<float> &head, const std::vector<float> &pixelOutput, int t, int k) const
{
    if (m_config.pixelHidden > 0) {
        return pixelOutput.data() + (t*m_patchOutputCount + k)*m_pixelOutputCount;
    }
    return head.data() + t*m_headOutputCount + k;
}

float *VisionTransformer::headValues(std::vector<float> &head, std::vector<float> &pixelOutput, int t, int k) const
{
    return const_cast<float *>(headValues(static_cast<const std::vector<float> &>(head),
                                          static_cast<const std::vector<float> &>(pixelOutput), t, k));
}

// distance of the pixel is values[0] and its sky is values[skyOffset()]
int VisionTransformer::skyOffset() const
{
    return m_config.pixelHidden > 0 ? 1 : m_patchOutputCount;
}

/*!
 * \brief VisionTransformer::pixelWindows
 * colors of the pixelWindow x pixelWindow pixels around every pixel, pixels outside the image are 0
 * \param image height x width x 3 colors
 * \param pixelInputs returns [tokens*patchSize*patchSize x pixelWindow*pixelWindow*3], pixels token by token
 */
void VisionTransformer::pixelWindows(const float *image, float *pixelInputs) const
{
    const int patchSize = m_config.patchSize;
    const int columns = m_config.width/patchSize;
    const int window = m_config.pixelWindow;
    for (int t=0;t<m_tokenCount;t++) {
        for (int k=0;k<m_patchOutputCount;k++) {
            const int x = (t%columns)*patchSize + k%patchSize - window/2;
            const int y = (t/columns)*patchSize + k/patchSize - window/2;
            float *in = pixelInputs + static_cast<size_t>(t*m_patchOutputCount + k)*static_cast<size_t>(m_pixelInputCount);
            for (int wy=0;wy<window;wy++) {
                for (int wx=0;wx<window;wx++) {
                    float *color = in + (wy*window + wx)*3;
                    if (x + wx < 0 || x + wx >= m_config.width || y + wy < 0 || y + wy >= m_config.height) {
                        std::fill(color, color + 3, 0.0f);
                    } else {
                        std::copy_n(image + (static_cast<size_t>(y + wy)*static_cast<size_t>(m_config.width)
                                             + static_cast<size_t>(x + wx))*3, 3, color);
                    }
                }
            }
        }
    }
}

/*!
 * \brief VisionTransformer::refineWindows
 * activated hidden values of the pixel head of the refineWindow x refineWindow pixels around every pixel,
 * pixels outside the image are 0
 * \param activated [rows x pixelHidden] pixels token by token
 * \param refineInputs returns [rows x refineWindow*refineWindow*pixelHidden]
 */
void VisionTransformer::refineWindows(const float *activated, float *refineInputs) const
{
    const int patchSize = m_config.patchSize;
    const int columns = m_config.width/patchSize;
    const int window = m_config.refineWindow;
    const int hidden = m_config.pixelHidden;
    for (int t=0;t<m_tokenCount;t++) {
        for (int k=0;k<m_patchOutputCount;k++) {
            const int x = (t%columns)*patchSize + k%patchSize - window/2;
            const int y = (t/columns)*patchSize + k/patchSize - window/2;
            float *in = refineInputs + static_cast<size_t>(t*m_patchOutputCount + k)*static_cast<size_t>(m_refineInputCount);
            for (int wy=0;wy<window;wy++) {
                for (int wx=0;wx<window;wx++) {
                    float *values = in + (wy*window + wx)*hidden;
                    const int nx = x + wx;
                    const int ny = y + wy;
                    if (nx < 0 || nx >= m_config.width || ny < 0 || ny >= m_config.height) {
                        std::fill(values, values + hidden, 0.0f);
                    } else {
                        const int row = ((ny/patchSize)*columns + nx/patchSize)*m_patchOutputCount
                                        + (ny%patchSize)*patchSize + nx%patchSize;
                        std::copy_n(activated + static_cast<size_t>(row)*static_cast<size_t>(hidden), hidden, values);
                    }
                }
            }
        }
    }
}

/*!
 * \brief VisionTransformer::refineWindowsBackward
 * adds the gradients of refineWindows() inputs to the activated hidden values: every pixel gets the gradients of the
 * windows of the pixels around it
 * \param refineInputGradient [rows x refineWindow*refineWindow*pixelHidden]
 * \param activatedGradient [rows x pixelHidden], gradients are added
 */
void VisionTransformer::refineWindowsBackward(const float *refineInputGradient, float *activatedGradient) const
{
    const int patchSize = m_config.patchSize;
    const int columns = m_config.width/patchSize;
    const int window = m_config.refineWindow;
    const int hidden = m_config.pixelHidden;
    for (int t=0;t<m_tokenCount;t++) {
        for (int k=0;k<m_patchOutputCount;k++) {
            const int x = (t%columns)*patchSize + k%patchSize;
            const int y = (t/columns)*patchSize + k/patchSize;
            float *dActivated = activatedGradient + static_cast<size_t>(t*m_patchOutputCount + k)*static_cast<size_t>(hidden);
            for (int wy=0;wy<window;wy++) {
                for (int wx=0;wx<window;wx++) {
                    // pixel (x, y) is at (wx, wy) of the window of the pixel (nx, ny)
                    const int nx = x - wx + window/2;
                    const int ny = y - wy + window/2;
                    if (nx < 0 || nx >= m_config.width || ny < 0 || ny >= m_config.height) {
                        continue;
                    }
                    const int row = ((ny/patchSize)*columns + nx/patchSize)*m_patchOutputCount
                                    + (ny%patchSize)*patchSize + nx%patchSize;
                    const float *dIn = refineInputGradient + static_cast<size_t>(row)*static_cast<size_t>(m_refineInputCount)
                                       + (wy*window + wx)*hidden;
                    for (int h=0;h<hidden;h++) {
                        dActivated[h] += dIn[h];
                    }
                }
            }
        }
    }
}

/*!
 * \brief VisionTransformer::backward
 * adds the gradients of loss = sum of (output - target)^2 of the PixelDistance pixels into the workspace
 * (with setHuberDelta() the Huber loss of the errors, see distanceLoss(), with twoDistances twoDistanceLoss() of pixel_label.h,
 * the loss of the pixels near the depth jumps is multiplied by the edge weight, see setEdgeWeight()),
 * with the sky output also SKY_LOSS_WEIGHT*binary cross-entropy of the sky output of every pixel that is not ignored,
 * with the surface output and its targets also surfacePixelLoss() of pixel_label.h of every pixel,
 * uses the values of the latest forward() of the workspace
 * \param workspace workspace of the latest forward()
 * \param target height x width correct values
 * \param labels height x width PixelLabel values
 * \param skyLoss if not nullptr, the sky loss is added to it
 * \param normals height x width x 3 correct normals in the camera (ImageSamples::getSurfaceTarget()),
 * nullptr: surface outputs are not learned
 * \param surfaces height x width surface numbers
 * \param surfaceLoss if not nullptr, the surface loss is added to it
 * \return distance loss
 */
double VisionTransformer::backward(Workspace *workspace, const float *target, const uint8_t *labels, double *skyLoss,
                                   const float *normals, const int32_t *surfaces, double *surfaceLoss) const
{
    const int patchSize = m_config.patchSize;
    const int columns = m_config.width/patchSize;
    const int dimension = m_config.dimension;
    const int count = m_tokenCount;
    auto gradient = [workspace](int parameter) {
        return workspace->m_gradients[static_cast<size_t>(parameter)].data();
    };

    const bool pixelHead = m_config.pixelHidden > 0;
    std::vector<float> &dOutput = pixelHead ? workspace->m_dPixelOutput : workspace->m_dHead;
    const int s = skyOffset();
    const bool surface = m_config.surfaceOutput;
    const bool surfaceTarget = surface && normals && surfaces;
    const size_t width = static_cast<size_t>(m_config.width);
    double loss = 0.0;
    for (int t=0;t<count;t++) {
        const int left = (t%columns)*patchSize;
        const int top = (t/columns)*patchSize;
        for (int py=0;py<patchSize;py++) {
            for (int px=0;px<patchSize;px++) {
                const size_t pixel = static_cast<size_t>(top + py)*static_cast<size_t>(m_config.width) + static_cast<size_t>(left + px);
                const int k = py*patchSize + px;
                const float *head = headValues(workspace->m_head, workspace->m_pixelOutput, t, k);
                float *dHead = headValues(dOutput, dOutput, t, k);
                const float weight = labels[pixel] == PixelDistance
                                     ? edgePixelWeight(target, labels, m_config.width, m_config.height, left + px, top + py,
                                                       m_edgeJump, m_edgeWeight) : 1.0f;
                if (m_config.twoDistances) {
                    // gap of the far distance (farDistance()) and the choice logit are the last values of the pixel
                    const int second = m_pixelOutputCount - 2;
                    float gradient[3] = { 0.0f, 0.0f, 0.0f };
                    if (labels[pixel] == PixelDistance) {
                        loss += weight*twoDistanceLoss(head[0], head[second], head[second + 1], target[pixel], m_huberDelta,
                                                       gradient);
                    }
                    dHead[0] = weight*gradient[0];
                    dHead[second] = weight*gradient[1];
                    dHead[second + 1] = weight*gradient[2];
                } else if (labels[pixel] == PixelDistance) {
                    float gradient;
                    loss += weight*distanceLoss(head[0] - target[pixel], m_huberDelta, &gradient);
                    dHead[0] = weight*gradient;
                } else {
                    dHead[0] = 0.0f;
                }
                if (surface) {
                    float *dSurface = dHead + m_surfaceOffset;
                    if (!surfaceTarget) {
                        std::fill(dSurface, dSurface + SURFACE_OUTPUT_COUNT, 0.0f);
                    } else {
                        // neighbours outside the image are not known
                        const int x = left + px;
                        const int y = top + py;
                        const int right = x + 1 < m_config.width
                            ? connectionTarget(labels[pixel], labels[pixel + 1], surfaces[pixel], surfaces[pixel + 1]) : CONNECTION_IGNORE;
                        const int down = y + 1 < m_config.height
                            ? connectionTarget(labels[pixel], labels[pixel + width], surfaces[pixel], surfaces[pixel + width])
                            : CONNECTION_IGNORE;
                        const float pixelLoss = surfacePixelLoss(head + m_surfaceOffset,
                                                            labels[pixel] == PixelDistance ? normals + pixel*3 : nullptr,
                                                            right, down, dSurface);
                        if (surfaceLoss) {
                            *surfaceLoss += static_cast<double>(pixelLoss);
                        }
                    }
                }
                if (!m_config.skyOutput) {
                    continue;
                }
                if (labels[pixel] == PixelIgnore) {
                    dHead[s] = 0.0f;
                    continue;
                }
                // binary cross-entropy of the logit z: loss = log(1 + exp(z)) - y*z, its derivative is sigmoid(z) - y
                const float z = head[s];
                const float y = labels[pixel] == PixelSky ? 1.0f : 0.0f;
                dHead[s] = SKY_LOSS_WEIGHT*(1.0f/(1.0f + std::exp(-z)) - y);
                if (skyLoss) {
                    *skyLoss += static_cast<double>(SKY_LOSS_WEIGHT*(std::max(z, 0.0f) - z*y + std::log1p(std::exp(-std::fabs(z)))));
                }
            }
        }
    }

    float *dx = workspace->m_dx.data();
    float *dy = workspace->m_dy.data();
    float *temp = workspace->m_dAttention.data();
    if (pixelHead) {
        // m_dHead is the gradient of the hidden values of the pixel head, the token part has the same gradient
        const int rows = count*m_patchOutputCount;
        if (m_refineWeight >= 0) {
            linearBackward(workspace->m_refineActivated.data(), workspace->m_dPixelOutput.data(), rows, m_config.refineHidden,
                           m_pixelOutputCount, value(m_refineOutWeight), gradient(m_refineOutWeight),
                           gradient(m_refineOutBias), workspace->m_dRefineHidden.data());
            for (size_t i=0;i<workspace->m_dRefineHidden.size();i++) {
                workspace->m_dRefineHidden[i] *= geluDerivative(workspace->m_refineHidden[i]);
            }
            linearBackward(workspace->m_refineInputs.data(), workspace->m_dRefineHidden.data(), rows, m_refineInputCount,
                           m_config.refineHidden, value(m_refineWeight), gradient(m_refineWeight), gradient(m_refineBias),
                           workspace->m_dRefineInputs.data());
        }
        linearBackward(workspace->m_pixelActivated.data(), workspace->m_dPixelOutput.data(), rows, m_config.pixelHidden,
                       m_pixelOutputCount, value(m_pixelOutWeight), gradient(m_pixelOutWeight), gradient(m_pixelOutBias),
                       workspace->m_dHead.data());
        if (m_refineWeight >= 0) {
            refineWindowsBackward(workspace->m_dRefineInputs.data(), workspace->m_dHead.data());
        }
        for (size_t i=0;i<workspace->m_dHead.size();i++) {
            workspace->m_dHead[i] *= geluDerivative(workspace->m_pixelHidden[i]);
        }
        linearBackward(workspace->m_pixelInputs.data(), workspace->m_dHead.data(), rows, m_pixelInputCount, m_config.pixelHidden,
                       value(m_pixelWeight), gradient(m_pixelWeight), gradient(m_pixelBias), nullptr);
    }
    linearBackward(workspace->m_yFinal.data(), workspace->m_dHead.data(), count, dimension, m_headOutputCount,
                   value(m_headWeight), gradient(m_headWeight), gradient(m_headBias), dy);
    layerNormBackward(dy, workspace->m_normFinal.data(), workspace->m_rstdFinal.data(), count, dimension,
                      value(m_normFinalGain), gradient(m_normFinalGain), gradient(m_normFinalBias), dx);

    for (int l=static_cast<int>(m_layerParameters.size())-1;l>=0;l--) {
        const LayerParameters &parameters = m_layerParameters[static_cast<size_t>(l)];
        const Workspace::Layer &layer = workspace->m_layers[static_cast<size_t>(l)];
        // dx is the gradient of the block output: output = x1 + mlp2(gelu(mlp1(layer norm 2(x1))))
        linearBackward(layer.activated.data(), dx, count, m_config.mlpDimension, dimension, value(parameters.mlp2Weight),
                       gradient(parameters.mlp2Weight), gradient(parameters.mlp2Bias), workspace->m_dHidden.data());
        for (size_t i=0;i<layer.hidden.size();i++) {
            workspace->m_dHidden[i] *= geluDerivative(layer.hidden[i]);
        }
        linearBackward(layer.y2.data(), workspace->m_dHidden.data(), count, dimension, m_config.mlpDimension,
                       value(parameters.mlp1Weight), gradient(parameters.mlp1Weight), gradient(parameters.mlp1Bias), dy);
        layerNormBackward(dy, layer.norm2.data(), layer.rstd2.data(), count, dimension, value(parameters.norm2Gain),
                          gradient(parameters.norm2Gain), gradient(parameters.norm2Bias), temp);
        for (int i=0;i<count*dimension;i++) {
            dx[i] += temp[i];
        }
        // dx is the gradient of x1 = input + out(attention(qkv(layer norm 1(input))))
        linearBackward(layer.attention.data(), dx, count, dimension, dimension, value(parameters.outWeight),
                       gradient(parameters.outWeight), gradient(parameters.outBias), workspace->m_dAttention.data());
        attentionBackward(layer, workspace->m_dAttention.data(), workspace->m_dqkv.data(), workspace->m_dRow.data());
        linearBackward(layer.y1.data(), workspace->m_dqkv.data(), count, dimension, 3*dimension, value(parameters.qkvWeight),
                       gradient(parameters.qkvWeight), gradient(parameters.qkvBias), dy);
        layerNormBackward(dy, layer.norm1.data(), layer.rstd1.data(), count, dimension, value(parameters.norm1Gain),
                          gradient(parameters.norm1Gain), gradient(parameters.norm1Bias), temp);
        for (int i=0;i<count*dimension;i++) {
            dx[i] += temp[i];
        }
    }

    // dx is the gradient of the tokens: linear(patch) + position
    float *positionGradient = gradient(m_position);
    for (int i=0;i<count*dimension;i++) {
        positionGradient[i] += dx[i];
    }
    linearBackward(workspace->m_patches.data(), dx, count, m_patchInputCount, dimension, value(m_patchWeight),
                   gradient(m_patchWeight), gradient(m_patchBias), nullptr);
    return loss;
}

/*!
 * \brief VisionTransformer::setHuberDelta
 * training loss of the distance: 0 is the squared error, > 0 is the Huber loss that is the squared error
 * up to the error huberDelta and grows linearly after it. Edge pixels, whose distance can be the object or the
 * background, are then learned to be one of them (like the median) instead of the mean between them.
 */
void VisionTransformer::setHuberDelta(float huberDelta)
{
    m_huberDelta = huberDelta;
}

float VisionTransformer::getHuberDelta() const
{
    return m_huberDelta;
}

/*!
 * \brief VisionTransformer::setEdgeWeight
 * training loss of the distance of the pixels near the depth jumps (edgePixelWeight() of pixel_label.h) is multiplied
 * by weight, 1 is the same loss as the other pixels
 * \param weight weight of the edge pixels
 * \param jump logarithm of 1 + EDGE_JUMP in the units of the targets (the normalized logarithms of the distances)
 */
void VisionTransformer::setEdgeWeight(float weight, float jump)
{
    m_edgeWeight = weight;
    m_edgeJump = jump;
}

float VisionTransformer::getEdgeWeight() const
{
    return m_edgeWeight;
}

float VisionTransformer::getEdgeJump() const
{
    return m_edgeJump;
}

/*!
 * \brief VisionTransformer::adamStep
 * sums the gradients of the workspaces, changes the weights with Adam and clears the gradients of the workspaces
 * \param workspaces workspaces that have gradients
 * \param learningRate size of the step
 * \param gradientScale gradients are multiplied by this, for example 1/(count of the pixels in the loss)
 */
void VisionTransformer::adamStep(const std::vector<Workspace *> &workspaces, float learningRate, float gradientScale)
{
    m_step++;
    // moments start from 0, so the first steps are corrected to be not too small
    const float correction1 = 1.0f - std::pow(ADAM_BETA1, static_cast<float>(m_step));
    const float correction2 = 1.0f - std::pow(ADAM_BETA2, static_cast<float>(m_step));
    for (size_t p=0;p<m_parameters.size();p++) {
        Parameter &parameter = m_parameters[p];
        for (size_t i=0;i<parameter.value.size();i++) {
            float gradient = 0.0f;
            for (Workspace *workspace : workspaces) {
                gradient += workspace->m_gradients[p][i];
                workspace->m_gradients[p][i] = 0.0f;
            }
            gradient *= gradientScale;
            parameter.moment1[i] = ADAM_BETA1*parameter.moment1[i] + (1.0f - ADAM_BETA1)*gradient;
            parameter.moment2[i] = ADAM_BETA2*parameter.moment2[i] + (1.0f - ADAM_BETA2)*gradient*gradient;
            parameter.value[i] -= learningRate*(parameter.moment1[i]/correction1)
                                  /(std::sqrt(parameter.moment2[i]/correction2) + ADAM_EPSILON);
        }
    }
}

/*!
 * \brief VisionTransformer::linearForward
 * output[n][o] = bias[o] + sum of weight[o][i]*input[n][i]
 */
void VisionTransformer::linearForward(const float *input, int count, int inputCount, const float *weight, const float *bias,
                                      int outputCount, float *output)
{
    for (int n=0;n<count;n++) {
        const float *in = input + n*inputCount;
        float *out = output + n*outputCount;
        for (int o=0;o<outputCount;o++) {
            const float *w = weight + o*inputCount;
            float sum = bias[o];
            for (int i=0;i<inputCount;i++) {
                sum += w[i]*in[i];
            }
            out[o] = sum;
        }
    }
}

/*!
 * \brief VisionTransformer::linearBackward
 * adds the weight and bias gradients and sets the input gradient (if inputGradient is not nullptr)
 */
void VisionTransformer::linearBackward(const float *input, const float *outputGradient, int count, int inputCount, int outputCount,
                                       const float *weight, float *weightGradient, float *biasGradient, float *inputGradient)
{
    if (inputGradient) {
        std::fill(inputGradient, inputGradient + count*inputCount, 0.0f);
    }
    for (int n=0;n<count;n++) {
        const float *in = input + n*inputCount;
        const float *dOut = outputGradient + n*outputCount;
        float *dIn = inputGradient ? inputGradient + n*inputCount : nullptr;
        for (int o=0;o<outputCount;o++) {
            const float d = dOut[o];
            if (d == 0.0f) {
                continue;
            }
            biasGradient[o] += d;
            float *dw = weightGradient + o*inputCount;
            for (int i=0;i<inputCount;i++) {
                dw[i] += d*in[i];
            }
            if (dIn) {
                const float *w = weight + o*inputCount;
                for (int i=0;i<inputCount;i++) {
                    dIn[i] += d*w[i];
                }
            }
        }
    }
}

/*!
 * \brief VisionTransformer::layerNormForward
 * every token is normalized to mean 0 and standard deviation 1, then multiplied by gain and added bias
 */
void VisionTransformer::layerNormForward(const float *input, int count, int dimension, const float *gain, const float *bias,
                                         float *norm, float *rstd, float *output)
{
    for (int n=0;n<count;n++) {
        const float *in = input + n*dimension;
        float mean = 0.0f;
        for (int i=0;i<dimension;i++) {
            mean += in[i];
        }
        mean /= static_cast<float>(dimension);
        float variance = 0.0f;
        for (int i=0;i<dimension;i++) {
            variance += (in[i] - mean)*(in[i] - mean);
        }
        variance /= static_cast<float>(dimension);
        const float r = 1.0f/std::sqrt(variance + LAYER_NORM_EPSILON);
        rstd[n] = r;
        for (int i=0;i<dimension;i++) {
            const float normalized = (in[i] - mean)*r;
            norm[n*dimension + i] = normalized;
            output[n*dimension + i] = normalized*gain[i] + bias[i];
        }
    }
}

/*!
 * \brief VisionTransformer::layerNormBackward
 * adds the gain and bias gradients and sets the input gradient
 */
void VisionTransformer::layerNormBackward(const float *outputGradient, const float *norm, const float *rstd, int count, int dimension,
                                          const float *gain, float *gainGradient, float *biasGradient, float *inputGradient)
{
    for (int n=0;n<count;n++) {
        const float *dOut = outputGradient + n*dimension;
        const float *normalized = norm + n*dimension;
        float sum = 0.0f;
        float sumNormalized = 0.0f;
        for (int i=0;i<dimension;i++) {
            const float dNorm = dOut[i]*gain[i];
            gainGradient[i] += dOut[i]*normalized[i];
            biasGradient[i] += dOut[i];
            sum += dNorm;
            sumNormalized += dNorm*normalized[i];
        }
        const float mean = sum/static_cast<float>(dimension);
        const float meanNormalized = sumNormalized/static_cast<float>(dimension);
        for (int i=0;i<dimension;i++) {
            const float dNorm = dOut[i]*gain[i];
            inputGradient[n*dimension + i] = rstd[n]*(dNorm - mean - normalized[i]*meanNormalized);
        }
    }
}

/*!
 * \brief VisionTransformer::attentionForward
 * multi-head self-attention: every head compares the query of every token with the keys of all tokens,
 * softmax of the scores are the weights, result is the weighted sum of the values of all tokens
 */
void VisionTransformer::attentionForward(Workspace::Layer *layer) const
{
    const int count = m_tokenCount;
    const int dimension = m_config.dimension;
    const int headDimension = dimension/m_config.heads;
    const float scale = 1.0f/std::sqrt(static_cast<float>(headDimension));
    std::fill(layer->attention.begin(), layer->attention.end(), 0.0f);
    for (int h=0;h<m_config.heads;h++) {
        for (int i=0;i<count;i++) {
            const float *query = layer->qkv.data() + i*3*dimension + h*headDimension;
            float *probability = layer->probability.data() + (static_cast<size_t>(h)*count + static_cast<size_t>(i))*count;
            float maxScore = -INFINITY;
            for (int j=0;j<count;j++) {
                const float *key = layer->qkv.data() + j*3*dimension + dimension + h*headDimension;
                float score = 0.0f;
                for (int d=0;d<headDimension;d++) {
                    score += query[d]*key[d];
                }
                score *= scale;
                probability[j] = score;
                maxScore = std::max(maxScore, score);
            }
            float sum = 0.0f;
            for (int j=0;j<count;j++) {
                probability[j] = std::exp(probability[j] - maxScore);
                sum += probability[j];
            }
            float *out = layer->attention.data() + i*dimension + h*headDimension;
            for (int j=0;j<count;j++) {
                probability[j] /= sum;
                const float *valueVector = layer->qkv.data() + j*3*dimension + 2*dimension + h*headDimension;
                for (int d=0;d<headDimension;d++) {
                    out[d] += probability[j]*valueVector[d];
                }
            }
        }
    }
}

/*!
 * \brief VisionTransformer::attentionBackward
 * sets the gradient of queries, keys and values from the gradient of the attention result
 */
void VisionTransformer::attentionBackward(const Workspace::Layer &layer, const float *attentionGradient, float *qkvGradient,
                                          float *rowBuffer) const
{
    const int count = m_tokenCount;
    const int dimension = m_config.dimension;
    const int headDimension = dimension/m_config.heads;
    const float scale = 1.0f/std::sqrt(static_cast<float>(headDimension));
    std::fill(qkvGradient, qkvGradient + count*3*dimension, 0.0f);
    for (int h=0;h<m_config.heads;h++) {
        for (int i=0;i<count;i++) {
            const float *dOut = attentionGradient + i*dimension + h*headDimension;
            const float *probability = layer.probability.data() + (static_cast<size_t>(h)*count + static_cast<size_t>(i))*count;
            // gradient of the weights and of the values
            float weightedSum = 0.0f;
            for (int j=0;j<count;j++) {
                const float *valueVector = layer.qkv.data() + j*3*dimension + 2*dimension + h*headDimension;
                float *dValue = qkvGradient + j*3*dimension + 2*dimension + h*headDimension;
                float dProbability = 0.0f;
                for (int d=0;d<headDimension;d++) {
                    dProbability += dOut[d]*valueVector[d];
                    dValue[d] += probability[j]*dOut[d];
                }
                rowBuffer[j] = dProbability;
                weightedSum += probability[j]*dProbability;
            }
            // softmax: dScore = p*(dp - sum of p*dp)
            const float *query = layer.qkv.data() + i*3*dimension + h*headDimension;
            float *dQuery = qkvGradient + i*3*dimension + h*headDimension;
            for (int j=0;j<count;j++) {
                const float dScore = probability[j]*(rowBuffer[j] - weightedSum)*scale;
                if (dScore == 0.0f) {
                    continue;
                }
                const float *key = layer.qkv.data() + j*3*dimension + dimension + h*headDimension;
                float *dKey = qkvGradient + j*3*dimension + dimension + h*headDimension;
                for (int d=0;d<headDimension;d++) {
                    dQuery[d] += dScore*key[d];
                    dKey[d] += dScore*query[d];
                }
            }
        }
    }
}

/*!
 * \brief VisionTransformer::Workspace::Workspace
 * buffers of one thread for the network
 */
VisionTransformer::Workspace::Workspace(const VisionTransformer &network)
{
    const Config &config = network.m_config;
    const size_t count = static_cast<size_t>(network.m_tokenCount);
    const size_t dimension = static_cast<size_t>(config.dimension);
    const size_t mlp = static_cast<size_t>(config.mlpDimension);
    for (const Parameter &parameter : network.m_parameters) {
        m_gradients.emplace_back(parameter.value.size(), 0.0f);
    }
    m_patches.resize(count*static_cast<size_t>(network.m_patchInputCount));
    m_layers.resize(static_cast<size_t>(config.layers));
    for (Layer &layer : m_layers) {
        layer.input.resize(count*dimension);
        layer.norm1.resize(count*dimension);
        layer.rstd1.resize(count);
        layer.y1.resize(count*dimension);
        layer.qkv.resize(count*3*dimension);
        layer.probability.resize(static_cast<size_t>(config.heads)*count*count);
        layer.attention.resize(count*dimension);
        layer.x1.resize(count*dimension);
        layer.norm2.resize(count*dimension);
        layer.rstd2.resize(count);
        layer.y2.resize(count*dimension);
        layer.hidden.resize(count*mlp);
        layer.activated.resize(count*mlp);
    }
    m_output.resize(count*dimension);
    m_normFinal.resize(count*dimension);
    m_rstdFinal.resize(count);
    m_yFinal.resize(count*dimension);
    m_head.resize(count*static_cast<size_t>(network.m_headOutputCount));
    if (config.pixelHidden > 0) {
        const size_t rows = count*static_cast<size_t>(network.m_patchOutputCount);
        m_pixelInputs.resize(rows*static_cast<size_t>(network.m_pixelInputCount));
        m_pixelHidden.resize(rows*static_cast<size_t>(config.pixelHidden));
        m_pixelActivated.resize(m_pixelHidden.size());
        m_pixelOutput.resize(rows*static_cast<size_t>(network.m_pixelOutputCount));
        m_dPixelOutput.resize(m_pixelOutput.size());
        if (config.refineHidden > 0) {
            m_refineInputs.resize(rows*static_cast<size_t>(network.m_refineInputCount));
            m_refineHidden.resize(rows*static_cast<size_t>(config.refineHidden));
            m_refineActivated.resize(m_refineHidden.size());
            m_refineOutput.resize(m_pixelOutput.size());
            m_dRefineHidden.resize(m_refineHidden.size());
            m_dRefineInputs.resize(m_refineInputs.size());
        }
    }
    m_dx.resize(count*dimension);
    m_dy.resize(count*dimension);
    m_dqkv.resize(count*3*dimension);
    m_dAttention.resize(count*dimension);
    m_dHidden.resize(count*mlp);
    m_dHead.resize(m_head.size());
    m_dRow.resize(count);
}

const std::vector<float> &VisionTransformer::Workspace::getGradient(int parameter) const
{
    return m_gradients[static_cast<size_t>(parameter)];
}

/*!
 * \brief VisionTransformer::setCameras
 * \param cameras cameras of the training images, they are saved to the model file
 */
void VisionTransformer::setCameras(const std::vector<Camera> &cameras)
{
    m_cameras = cameras;
}

/*!
 * \brief VisionTransformer::getCameras
 * \return cameras of the training images, empty if the model file did not have them
 */
const std::vector<VisionTransformer::Camera> &VisionTransformer::getCameras() const
{
    return m_cameras;
}

/*!
 * \brief VisionTransformer::save
 * saves the network to a binary file (native byte order, little-endian on x86):
 * char[4] "VITM", uint32_t version (7), int32_t width, height, patchSize, dimension, heads, layers, mlpDimension,
 * skyOutput (0 or 1, missing in version 1 files), pixelHidden, pixelWindow (missing in version 1 and 2 files),
 * surfaceOutput (0 or 1, missing in version 1-3 files), refineHidden, refineWindow (missing in version 1-4 files),
 * twoDistances (0 or 1, missing in version 1-5 files),
 * float targetMean, targetScale (distance = exp(output*targetScale + targetMean) meters),
 * uint32_t count of the weight tensors, then for every tensor uint32_t count of the values and the float values,
 * uint32_t count of the cameras (missing in version 1-6 files), then for every camera uint32_t number, width, height,
 * distortion model and CAMERA_VALUE_COUNT doubles (Camera).
 * The file is written to filename.part first and then renamed, so a failed save does not leave a broken file
 * (.tmp is a training checkpoint of TrainingCheckpoint).
 * \param filename file of the model
 * \param targetMean, targetScale normalization of the outputs
 * \return true on success
 */
bool VisionTransformer::save(const std::string &filename, float targetMean, float targetScale) const
{
    const std::string temporary = filename + ".part";
    {
        std::ofstream file(temporary, std::ios::binary);
        auto writeUint32 = [&file](uint32_t value) {
            file.write(reinterpret_cast<const char *>(&value), sizeof(value));
        };
        auto writeInt32 = [&file](int32_t value) {
            file.write(reinterpret_cast<const char *>(&value), sizeof(value));
        };
        auto writeFloat = [&file](float value) {
            file.write(reinterpret_cast<const char *>(&value), sizeof(value));
        };
        file.write(MODEL_FILE_MAGIC, 4);
        writeUint32(MODEL_FILE_VERSION);
        writeInt32(m_config.width);
        writeInt32(m_config.height);
        writeInt32(m_config.patchSize);
        writeInt32(m_config.dimension);
        writeInt32(m_config.heads);
        writeInt32(m_config.layers);
        writeInt32(m_config.mlpDimension);
        writeInt32(m_config.skyOutput ? 1 : 0);
        writeInt32(m_config.pixelHidden);
        writeInt32(m_config.pixelWindow);
        writeInt32(m_config.surfaceOutput ? 1 : 0);
        writeInt32(m_config.refineHidden);
        writeInt32(m_config.refineWindow);
        writeInt32(m_config.twoDistances ? 1 : 0);
        writeFloat(targetMean);
        writeFloat(targetScale);
        writeUint32(static_cast<uint32_t>(m_parameters.size()));
        for (const Parameter &parameter : m_parameters) {
            writeUint32(static_cast<uint32_t>(parameter.value.size()));
            file.write(reinterpret_cast<const char *>(parameter.value.data()),
                       static_cast<std::streamsize>(parameter.value.size()*sizeof(float)));
        }
        writeUint32(static_cast<uint32_t>(m_cameras.size()));
        for (const Camera &camera : m_cameras) {
            writeUint32(camera.number);
            writeUint32(camera.width);
            writeUint32(camera.height);
            writeUint32(camera.distortionModel);
            file.write(reinterpret_cast<const char *>(camera.values), sizeof(camera.values));
        }
        file.close();
        if (!file) {
            TG_ERROR_LOG("Cannot save", temporary);
            std::remove(temporary.c_str());
            return false;
        }
    }
    if (std::rename(temporary.c_str(), filename.c_str()) != 0) {
        TG_ERROR_LOG("Cannot rename", temporary, "to", filename);
        std::remove(temporary.c_str());
        return false;
    }
    return true;
}

/*!
 * \brief VisionTransformer::load
 * loads a network that save() saved
 * \param filename file of the model
 * \param targetMean, targetScale return the normalization of the outputs
 * \return network, nullptr on failure
 */
std::unique_ptr<VisionTransformer> VisionTransformer::load(const std::string &filename, float *targetMean, float *targetScale)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        TG_ERROR_LOG("Cannot open", filename);
        return nullptr;
    }
    char magic[4];
    uint32_t version = 0, tensorCount = 0;
    int32_t values[7];
    file.read(magic, sizeof(magic));
    file.read(reinterpret_cast<char *>(&version), sizeof(version));
    if (!file || memcmp(magic, MODEL_FILE_MAGIC, sizeof(magic)) != 0) {
        TG_ERROR_LOG(filename, "is not a model file");
        return nullptr;
    }
    if (version < 1 || version > MODEL_FILE_VERSION) {
        TG_ERROR_LOG(filename + ":", "model file version is", std::to_string(version) + ",", "but 1 -", MODEL_FILE_VERSION,
                     "are supported");
        return nullptr;
    }
    file.read(reinterpret_cast<char *>(values), sizeof(values));
    Config config = { values[0], values[1], values[2], values[3], values[4], values[5], values[6] };
    // version 1 has no sky output flag after the network size
    if (version >= 2) {
        int32_t skyOutput = 0;
        file.read(reinterpret_cast<char *>(&skyOutput), sizeof(skyOutput));
        config.skyOutput = skyOutput != 0;
    }
    // versions 1 and 2 have the linear head
    if (version >= 3) {
        int32_t pixelHead[2] = { 0, 0 };
        file.read(reinterpret_cast<char *>(pixelHead), sizeof(pixelHead));
        config.pixelHidden = pixelHead[0];
        config.pixelWindow = pixelHead[1];
    }
    // versions 1-3 have no surface output
    if (version >= 4) {
        int32_t surfaceOutput = 0;
        file.read(reinterpret_cast<char *>(&surfaceOutput), sizeof(surfaceOutput));
        config.surfaceOutput = surfaceOutput != 0;
    }
    // versions 1-4 have no refine step
    if (version >= 5) {
        int32_t refine[2] = { 0, 0 };
        file.read(reinterpret_cast<char *>(refine), sizeof(refine));
        config.refineHidden = refine[0];
        config.refineWindow = refine[1];
    }
    // versions 1-5 have one distance
    if (version >= 6) {
        int32_t twoDistances = 0;
        file.read(reinterpret_cast<char *>(&twoDistances), sizeof(twoDistances));
        config.twoDistances = twoDistances != 0;
    }
    file.read(reinterpret_cast<char *>(targetMean), sizeof(*targetMean));
    file.read(reinterpret_cast<char *>(targetScale), sizeof(*targetScale));
    file.read(reinterpret_cast<char *>(&tensorCount), sizeof(tensorCount));
    if (!file) {
        TG_ERROR_LOG(filename, "is too short");
        return nullptr;
    }
    if (config.patchSize <= 0 || config.width <= 0 || config.height <= 0 || config.width%config.patchSize != 0
        || config.height%config.patchSize != 0 || config.heads <= 0 || config.dimension%config.heads != 0
        || config.layers < 0 || config.mlpDimension <= 0 || config.pixelHidden < 0
        || (config.pixelHidden > 0 && (config.pixelWindow <= 0 || config.pixelWindow%2 == 0))
        || (config.surfaceOutput && (config.pixelHidden <= 0 || !config.skyOutput)) || config.refineHidden < 0
        || (config.refineHidden > 0 && (config.pixelHidden <= 0 || config.refineWindow <= 0 || config.refineWindow%2 == 0))
        || (config.twoDistances && (config.pixelHidden <= 0 || !config.skyOutput))) {
        TG_ERROR_LOG(filename, "has an invalid network size");
        return nullptr;
    }
    std::unique_ptr<VisionTransformer> network = std::make_unique<VisionTransformer>(config, 0);
    if (tensorCount != network->m_parameters.size()) {
        TG_ERROR_LOG(filename, "has", tensorCount, "weight tensors, but the network has", network->m_parameters.size());
        return nullptr;
    }
    for (Parameter &parameter : network->m_parameters) {
        uint32_t size;
        file.read(reinterpret_cast<char *>(&size), sizeof(size));
        if (!file) {
            TG_ERROR_LOG(filename, "is too short");
            return nullptr;
        }
        if (size != parameter.value.size()) {
            TG_ERROR_LOG(filename, "has a weight tensor of a wrong size");
            return nullptr;
        }
        file.read(reinterpret_cast<char *>(parameter.value.data()), static_cast<std::streamsize>(size*sizeof(float)));
    }
    // versions 1-6 have no cameras
    if (version >= 7) {
        uint32_t cameraCount = 0;
        file.read(reinterpret_cast<char *>(&cameraCount), sizeof(cameraCount));
        if (cameraCount > MODEL_MAX_CAMERAS) {
            TG_ERROR_LOG(filename, "has", cameraCount, "cameras");
            return nullptr;
        }
        network->m_cameras.resize(cameraCount);
        for (Camera &camera : network->m_cameras) {
            uint32_t values[4] = { 0, 0, 0, 0 };
            file.read(reinterpret_cast<char *>(values), sizeof(values));
            camera.number = values[0];
            camera.width = values[1];
            camera.height = values[2];
            camera.distortionModel = values[3];
            file.read(reinterpret_cast<char *>(camera.values), sizeof(camera.values));
        }
    }
    if (!file) {
        TG_ERROR_LOG(filename, "is too short");
        return nullptr;
    }
    return network;
}
