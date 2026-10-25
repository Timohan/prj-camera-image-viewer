#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <string>
#include <window/tg_mainwindow.h>
#include "scene/viewer_scene.h"
#include "scene/view_key_listener.h"

class MainWindow : public TgMainWindow
{
public:
    MainWindow(int width, int height);

    bool load(const std::string &imageDirectory, const std::string &scene, const std::string &modelFile);

protected:
    virtual int initCustomShader() override;
    virtual void customBeforeRender() override;
    virtual void customRender() override;

private:
    ViewerScene m_viewerScene;
    ViewKeyListener m_viewKeyListener;
};

#endif
