#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <memory>
#include <window/tg_mainwindow.h>
#include <item2d/tg_mouse_capture.h>
#include <item2d/tg_rectangle.h>
#include <item2d/tg_textfield.h>
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
    // mouse on the whole window: left click picks a pixel, wheel moves forward or backward
    TgMouseCapture m_mouseCapture;
    // picked pixel and its distance from its camera, white text on a blue background
    TgRectangle m_pixelBackground;
    TgTextfield m_pixelText;
    // shows if the distances are predicted or correct, white text on a blue background
    TgRectangle m_distancesBackground;
    TgTextfield m_distancesText;
    // shows if the black areas are filled
    TgRectangle m_fillBackground;
    TgTextfield m_fillText;
    // shows if the images are tinted by the classes of the objects (only with the objects)
    TgRectangle m_objectsBackground;
    TgTextfield m_objectsText;
    // object of the picked pixel, below the picked pixel (only with the objects, after the first pick)
    TgRectangle m_objectBackground;
    TgTextfield m_objectText;

    void updateDistancesText();
    void updateFillText();
    void updateObjectsText();
};

#endif
