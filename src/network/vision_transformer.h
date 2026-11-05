#ifndef VISION_TRANSFORMER_H
#define VISION_TRANSFORMER_H

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

/*!
 * \brief VisionTransformer
 * Small Vision Transformer (ViT) that predicts a value for every pixel of an image.
 * The image is cut into patchSize x patchSize patches, every patch is one token (like a word of an LLM):
 * token = linear(colors of the patch) + learned position embedding of the patch.
 * Every transformer block: x = x + attention(layer norm(x)), x = x + mlp(layer norm(x)),
 * attention lets every token compare itself to all other tokens (multi-head self-attention).
 * At the end a linear head turns every token into the values of its patchSize x patchSize pixels,
 * with skyOutput also into a sky value of every pixel (logit of the probability that the pixel is sky).
 * With pixelHidden > 0 the head is a small MLP of every pixel instead (pixel head): the token gives pixelHidden
 * values to each of its pixels, the colors of the pixelWindow x pixelWindow pixels around the pixel are added
 * through their own weights, then GELU and a linear output of the distance (and sky). Colors of the pixel and
 * its neighbours let the edges of the objects follow the edges of the colors inside a token.
 * With surfaceOutput the pixel head gives also the surface of every pixel (SURFACE_OUTPUT_COUNT values of pixel_label.h):
 * normal of the surface in the camera and whether the pixel is on the same surface as its right and lower neighbour.
 * With refineHidden > 0 a refine step follows the pixel head: every pixel sees the GELU hidden values of the pixel head
 * of the refineWindow x refineWindow pixels around it, also of the pixels of the neighbouring tokens, and a small MLP
 * (linear, GELU, linear) adds a correction to all outputs of the pixel. Pixels at the edges of the tokens then follow
 * their neighbours instead of making 4x4 steps, and an occlusion edge can be sharp across the tokens.
 * The output weights of the refine step start from 0, so the refine step starts as no correction.
 * With twoDistances the pixel head gives a near and a far distance and a choice logit, the output distance is
 * the chosen one (chosenDistance() of pixel_label.h), so an edge pixel is the near or the far surface instead of their mean.
 *
 * The model file has also the cameras of the training images (position, direction and lens of every camera,
 * all images of a camera are from the same place), so a viewer needs only the images and the model file.
 *
 * Training: forward() and backward() use a Workspace, so many threads can run them at the same time
 * with the same weights, every Workspace adds its own gradients. adamStep() sums the gradients of
 * the workspaces and changes the weights (Adam optimizer).
 */
class VisionTransformer
{
public:
    struct Config {
        int width;          // image width in pixels, divisible by patchSize
        int height;         // image height in pixels, divisible by patchSize
        int patchSize;
        int dimension;      // size of a token vector
        int heads;          // attention heads, dimension must be divisible by heads
        int layers;         // transformer blocks
        int mlpDimension;   // hidden size of the MLP of a block
        bool skyOutput = false; // true: every pixel has also a sky output (logit, sky if > 0)
        int pixelHidden = 0;    // hidden size of the pixel head, 0: linear head
        int pixelWindow = 0;    // odd width of the color window around the pixel of the pixel head
        bool surfaceOutput = false; // true: every pixel has also the surface outputs, needs the pixel head and skyOutput
        int refineHidden = 0;   // hidden size of the refine step, 0: no refine step, needs the pixel head
        int refineWindow = 0;   // odd width of the window of the pixel head hidden values that the refine step sees
        bool twoDistances = false;  // true: the pixel head gives two distances and the choice between them (pixel_label.h),
                                    // needs the pixel head and skyOutput
    };

    // values of a camera: the camera information block of the data file of prj-camera-image-creator
    static constexpr int CAMERA_VALUE_COUNT = 36;
    struct Camera {
        uint32_t number;            // camera number of the image filenames (1, 2, ...)
        uint32_t width;             // image size in pixels
        uint32_t height;
        uint32_t distortionModel;   // 0: pinhole (k1-k6, p1, p2), 1: fisheye (k1-k4)
        // position x/y/z (mm), direction x/y/z, up x/y/z, ..., fx, fy, cx, cy (15-18), k1-k6 (19-24), p1, p2 (25, 26),
        // fisheye k1-k4 (27-30), like the data file
        double values[CAMERA_VALUE_COUNT];
    };

    class Workspace
    {
    public:
        explicit Workspace(const VisionTransformer &network);
        const std::vector<float> &getGradient(int parameter) const;

    private:
        friend class VisionTransformer;
        struct Layer {
            std::vector<float> input;       // [tokens x dimension]
            std::vector<float> norm1;       // normalized input before gain and bias
            std::vector<float> rstd1;       // [tokens] 1/standard deviation
            std::vector<float> y1;          // layer norm 1 output
            std::vector<float> qkv;         // [tokens x 3*dimension] queries, keys, values
            std::vector<float> probability; // [heads x tokens x tokens] attention weights
            std::vector<float> attention;   // [tokens x dimension] attention result of the heads
            std::vector<float> x1;          // input + attention output
            std::vector<float> norm2;
            std::vector<float> rstd2;
            std::vector<float> y2;
            std::vector<float> hidden;      // [tokens x mlpDimension] before GELU
            std::vector<float> activated;   // after GELU
        };
        std::vector<std::vector<float>> m_gradients;
        std::vector<float> m_patches;       // [tokens x patchSize*patchSize*3]
        std::vector<Layer> m_layers;
        std::vector<float> m_output;        // [tokens x dimension] output of the last block
        std::vector<float> m_normFinal;
        std::vector<float> m_rstdFinal;
        std::vector<float> m_yFinal;
        std::vector<float> m_head;          // [tokens x headOutputCount], pixel head: [tokens*patchSize*patchSize x pixelHidden]
        // pixel head, rows are the pixels token by token: row = token*patchSize*patchSize + py*patchSize + px
        std::vector<float> m_pixelInputs;   // [rows x pixelWindow*pixelWindow*3] colors around the pixel
        std::vector<float> m_pixelHidden;   // [rows x pixelHidden] before GELU
        std::vector<float> m_pixelActivated;
        std::vector<float> m_pixelOutput;   // [rows x pixelOutputCount] distance, then sky, then surface
                                            // (with the refine step its correction is added)
        // refine step, rows like the pixel head
        std::vector<float> m_refineInputs;  // [rows x refineWindow*refineWindow*pixelHidden] activated hidden values around
        std::vector<float> m_refineHidden;  // [rows x refineHidden] before GELU
        std::vector<float> m_refineActivated;
        std::vector<float> m_refineOutput;  // [rows x pixelOutputCount] correction
        std::vector<float> m_dRefineHidden;
        std::vector<float> m_dRefineInputs;
        std::vector<float> m_dPixelOutput;
        // backward buffers
        std::vector<float> m_dx;
        std::vector<float> m_dy;
        std::vector<float> m_dqkv;
        std::vector<float> m_dAttention;
        std::vector<float> m_dHidden;
        std::vector<float> m_dHead;
        std::vector<float> m_dRow;
    };

    VisionTransformer(const Config &config, unsigned int seed);

    const Config &getConfig() const;
    int getParameterCount() const;
    int getParameterTensorCount() const;
    std::vector<float> &getParameter(int parameter);
    void forward(Workspace *workspace, const float *image, float *output, float *skyOutput = nullptr,
                 float *surfaceOutput = nullptr) const;
    double backward(Workspace *workspace, const float *target, const uint8_t *labels, double *skyLoss = nullptr,
                    const float *normals = nullptr, const int32_t *surfaces = nullptr, double *surfaceLoss = nullptr) const;
    void adamStep(const std::vector<Workspace *> &workspaces, float learningRate, float gradientScale);
    void setHuberDelta(float huberDelta);
    float getHuberDelta() const;
    void setEdgeWeight(float weight, float jump);
    float getEdgeWeight() const;
    float getEdgeJump() const;
    void setCameras(const std::vector<Camera> &cameras);
    const std::vector<Camera> &getCameras() const;
    bool save(const std::string &filename, float targetMean, float targetScale) const;
    static std::unique_ptr<VisionTransformer> load(const std::string &filename, float *targetMean, float *targetScale);

private:
    struct Parameter {
        std::vector<float> value;
        std::vector<float> moment1;     // Adam: average of the gradients
        std::vector<float> moment2;     // Adam: average of the squared gradients
    };
    struct LayerParameters {
        int norm1Gain, norm1Bias, qkvWeight, qkvBias, outWeight, outBias;
        int norm2Gain, norm2Bias, mlp1Weight, mlp1Bias, mlp2Weight, mlp2Bias;
    };
    Config m_config;
    int m_tokenCount;
    int m_patchInputCount;      // patchSize*patchSize*3
    int m_patchOutputCount;     // patchSize*patchSize
    int m_headOutputCount;      // m_patchOutputCount, with skyOutput 2*m_patchOutputCount (distances, then sky),
                                // pixel head: m_patchOutputCount*pixelHidden
    int m_pixelInputCount;      // pixelWindow*pixelWindow*3
    int m_pixelOutputCount;     // pixel head: 1, with skyOutput 2, with surfaceOutput 2 + SURFACE_OUTPUT_COUNT,
                                // with twoDistances 2 more (the gap of the far distance and the choice logit are the last ones)
    int m_refineInputCount;     // refineWindow*refineWindow*pixelHidden
    std::vector<Parameter> m_parameters;
    int m_patchWeight, m_patchBias, m_position;
    std::vector<LayerParameters> m_layerParameters;
    int m_normFinalGain, m_normFinalBias, m_headWeight, m_headBias;
    int m_pixelWeight, m_pixelBias, m_pixelOutWeight, m_pixelOutBias;  // pixel head
    int m_refineWeight, m_refineBias, m_refineOutWeight, m_refineOutBias;  // refine step
    int m_step;
    float m_huberDelta;         // training only, 0: squared error of the distance
    float m_edgeWeight;         // training only, weight of the distance loss of the edge pixels (edgePixelWeight())
    float m_edgeJump;           // training only, logarithm of 1 + EDGE_JUMP in the units of the targets
    std::vector<Camera> m_cameras;  // cameras of the training images, empty if not known

    int addParameter(int size, float initValue);
    int addRandomParameter(int size, float limit, std::mt19937 *random);
    int addLinearWeight(int inputCount, int outputCount, std::mt19937 *random);
    const float *value(int parameter) const;

    static void linearForward(const float *input, int count, int inputCount, const float *weight, const float *bias,
                              int outputCount, float *output);
    static void linearBackward(const float *input, const float *outputGradient, int count, int inputCount, int outputCount,
                               const float *weight, float *weightGradient, float *biasGradient, float *inputGradient);
    static void layerNormForward(const float *input, int count, int dimension, const float *gain, const float *bias,
                                 float *norm, float *rstd, float *output);
    static void layerNormBackward(const float *outputGradient, const float *norm, const float *rstd, int count, int dimension,
                                  const float *gain, float *gainGradient, float *biasGradient, float *inputGradient);
    void attentionForward(Workspace::Layer *layer) const;
    void attentionBackward(const Workspace::Layer &layer, const float *attentionGradient, float *qkvGradient,
                           float *rowBuffer) const;
    void pixelWindows(const float *image, float *pixelInputs) const;
    void refineWindows(const float *activated, float *refineInputs) const;
    void refineWindowsBackward(const float *refineInputGradient, float *activatedGradient) const;
    const float *headValues(const std::vector<float> &head, const std::vector<float> &pixelOutput, int t, int k) const;
    float *headValues(std::vector<float> &head, std::vector<float> &pixelOutput, int t, int k) const;
    int skyOffset() const;
    static const int m_surfaceOffset = 2;   // surface outputs of the pixel head are after the distance and the sky
};

#endif // VISION_TRANSFORMER_H
