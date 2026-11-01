#include "mainwindow.h"

// label of the shown distances at the top left corner
#define DISTANCES_LABEL_MARGIN      10
#define DISTANCES_LABEL_WIDTH       260
#define DISTANCES_LABEL_HEIGHT      36
#define DISTANCES_LABEL_FONT_SIZE   20

MainWindow::MainWindow(int width, int height, std::unique_ptr<ViewerScene> viewerScene)
    : TgMainWindow(width, height, "prj-camera-image-viewer"),
    m_viewerScene(std::move(viewerScene)),
    m_viewKeyListener(this),
    m_distancesBackground(this, DISTANCES_LABEL_MARGIN, DISTANCES_LABEL_MARGIN, DISTANCES_LABEL_WIDTH,
                          DISTANCES_LABEL_HEIGHT, 30, 80, 200, 255),
    m_distancesText(&m_distancesBackground, "", "", DISTANCES_LABEL_FONT_SIZE, 255, 255, 255)
{
    m_distancesText.setHorizontalAlign(TgTextfieldHorizontalAlign::AlignCenterH);
    m_distancesText.setVerticalAlign(TgTextfieldVerticalAlign::AlignCenterV);
    updateDistancesText();
    m_viewKeyListener.onCameraSelected = [this](int index) {
        m_viewerScene->moveToCamera(index);
    };
    m_viewKeyListener.onToggleDistances = [this]() {
        m_viewerScene->toggleDistances();
        updateDistancesText();
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
    m_viewKeyListener.onMove = [this](float steps) {
        m_viewerScene->move(steps);
    };
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
}

void MainWindow::updateDistancesText()
{
    m_distancesText.setText(m_viewerScene->getCorrectDistances() ? "Correct distances" : "Predicted distances");
}
