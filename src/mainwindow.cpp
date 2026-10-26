#include "mainwindow.h"

MainWindow::MainWindow(int width, int height, std::unique_ptr<ViewerScene> viewerScene)
    : TgMainWindow(width, height, "prj-camera-image-viewer"),
    m_viewerScene(std::move(viewerScene)),
    m_viewKeyListener(this)
{
    m_viewKeyListener.onCameraSelected = [this](int index) {
        m_viewerScene->moveToCamera(index);
    };
    m_viewKeyListener.onToggleDistances = [this]() {
        m_viewerScene->toggleDistances();
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
