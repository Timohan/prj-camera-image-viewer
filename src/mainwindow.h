#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <memory>
#include <window/tg_mainwindow.h>
#include "scene/viewer_scene.h"
#include "scene/view_key_listener.h"

class MainWindow : public TgMainWindow
{
public:
    // viewerScene must be loaded, the window is created only after it, so a failed load does not need a window
    MainWindow(int width, int height, std::unique_ptr<ViewerScene> viewerScene);

protected:
    virtual int initCustomShader() override;
    virtual void customBeforeRender() override;
    virtual void customRender() override;

private:
    std::unique_ptr<ViewerScene> m_viewerScene;
    ViewKeyListener m_viewKeyListener;
};

#endif
