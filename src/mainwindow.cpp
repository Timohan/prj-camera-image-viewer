#include "mainwindow.h"

// labels at the top left corner: the picked pixel, the shown distances, filling of the black areas
#define LABEL_MARGIN                10
#define LABEL_HEIGHT                36
#define LABEL_FONT_SIZE             20
#define PIXEL_LABEL_WIDTH           500
#define DISTANCES_LABEL_WIDTH       260
#define FILL_LABEL_WIDTH            220
#define LABEL_BACKGROUND_COLOR      30, 80, 200, 255
#define LABEL_TEXT_COLOR            255, 255, 255
#define PIXEL_LABEL_START_TEXT      "Click a pixel"

MainWindow::MainWindow(int width, int height, std::unique_ptr<ViewerScene> viewerScene)
    : TgMainWindow(width, height, "prj-camera-image-viewer"),
    m_viewerScene(std::move(viewerScene)),
    m_viewKeyListener(this),
    m_mouseCapture(this),
    m_pixelBackground(this, LABEL_MARGIN, LABEL_MARGIN, PIXEL_LABEL_WIDTH, LABEL_HEIGHT, LABEL_BACKGROUND_COLOR),
    m_pixelText(&m_pixelBackground, PIXEL_LABEL_START_TEXT, "", LABEL_FONT_SIZE, LABEL_TEXT_COLOR),
    m_distancesBackground(this, 2*LABEL_MARGIN + PIXEL_LABEL_WIDTH, LABEL_MARGIN, DISTANCES_LABEL_WIDTH, LABEL_HEIGHT,
                          LABEL_BACKGROUND_COLOR),
    m_distancesText(&m_distancesBackground, "", "", LABEL_FONT_SIZE, LABEL_TEXT_COLOR),
    m_fillBackground(this, 3*LABEL_MARGIN + PIXEL_LABEL_WIDTH + DISTANCES_LABEL_WIDTH, LABEL_MARGIN, FILL_LABEL_WIDTH,
                     LABEL_HEIGHT, LABEL_BACKGROUND_COLOR),
    m_fillText(&m_fillBackground, "", "", LABEL_FONT_SIZE, LABEL_TEXT_COLOR)
{
    m_pixelText.setHorizontalAlign(TgTextfieldHorizontalAlign::AlignCenterH);
    m_pixelText.setVerticalAlign(TgTextfieldVerticalAlign::AlignCenterV);
    m_distancesText.setHorizontalAlign(TgTextfieldHorizontalAlign::AlignCenterH);
    m_distancesText.setVerticalAlign(TgTextfieldVerticalAlign::AlignCenterV);
    m_fillText.setHorizontalAlign(TgTextfieldHorizontalAlign::AlignCenterH);
    m_fillText.setVerticalAlign(TgTextfieldVerticalAlign::AlignCenterV);
    updateDistancesText();
    updateFillText();
    m_viewKeyListener.onCameraSelected = [this](int index) {
        m_viewerScene->moveToCamera(index);
    };
    m_viewKeyListener.onToggleDistances = [this]() {
        m_viewerScene->toggleDistances();
        updateDistancesText();
        // the picked point is picked again on the mesh of the new distances
        requestRender();
    };
    m_viewKeyListener.onToggleFillGaps = [this]() {
        m_viewerScene->toggleFillGaps();
        updateFillText();
    };
    m_viewKeyListener.onTurnStart = [this](int direction) {
        m_viewerScene->startTurn(direction);
    };
    m_viewKeyListener.onTurnStop = [this](int direction) {
        m_viewerScene->stopTurn(direction);
    };
    m_viewKeyListener.onPitchStart = [this](int direction) {
        m_viewerScene->startPitch(direction);
    };
    m_viewKeyListener.onPitchStop = [this](int direction) {
        m_viewerScene->stopPitch(direction);
    };
    m_mouseCapture.connectOnMouseClicked([this](TgMouseType button, float x, float y, const void *) {
        if (button == TgMouseType::ButtonLeft) {
            m_viewerScene->pick(x, y);
            requestRender();
        }
    });
    m_mouseCapture.connectOnMouseScrollMove([this](double, double y) {
        // wheel forward is positive, many wheel steps can be summed into one event
        if (y != 0.0) {
            m_viewerScene->move(static_cast<float>(y));
        }
    });
}

int MainWindow::initCustomShader()
{
    return m_viewerScene->initGl();
}

void MainWindow::customBeforeRender()
{
    if (m_viewerScene->update()) {
        // renders continuously while turning, otherwise next frame waits for events
        requestRender();
    }
}

void MainWindow::customRender()
{
    m_viewerScene->render(getWidth(), getHeight());
    if (m_viewerScene->takePicked()) {
        m_pixelText.setText(m_viewerScene->getPickedText().c_str());
        requestRender();
    }
}

void MainWindow::updateDistancesText()
{
    m_distancesText.setText(m_viewerScene->getCorrectDistances() ? "Correct distances" : "Predicted distances");
}

void MainWindow::updateFillText()
{
    m_fillText.setText(m_viewerScene->getFillGaps() ? "Fill gaps: on" : "Fill gaps: off");
}
