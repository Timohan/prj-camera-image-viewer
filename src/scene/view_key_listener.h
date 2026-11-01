#ifndef VIEW_KEY_LISTENER_H
#define VIEW_KEY_LISTENER_H

#include <functional>
#include <item2d/tg_item2d.h>

/*!
 * \brief ViewKeyListener
 * Keys of the viewer: 0 moves to the middle of the cameras, 1-9 move to camera 1-9, t changes between
 * the predicted and the correct distances, f fills the black areas or shows them. Left and right arrow keys turn the view while they are down
 * (direction -1 is left and 1 is right), up and down arrow keys turn it up or down (direction -1 is down
 * and 1 is up). The mouse is handled by the TgMouseCapture of the main window.
 * TgMainWindow sends the events only to its children, so this must be a child of the main window.
 */
class ViewKeyListener : public TgItem2d
{
public:
    explicit ViewKeyListener(TgItem2d *parent);

    std::function<void(int index)> onCameraSelected;
    std::function<void()> onToggleDistances;
    std::function<void()> onToggleFillGaps;
    std::function<void(int direction)> onTurnStart;
    std::function<void(int direction)> onTurnStop;
    std::function<void(int direction)> onPitchStart;
    std::function<void(int direction)> onPitchStop;

protected:
    virtual TgEventResult handleEvent(TgEventData *eventData, const TgWindowInfo *windowInfo) override;
};

#endif // VIEW_KEY_LISTENER_H
