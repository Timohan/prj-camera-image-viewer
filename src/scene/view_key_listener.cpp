#include "view_key_listener.h"

ViewKeyListener::ViewKeyListener(TgItem2d *parent) :
    TgItem2d(parent)
{
}

TgEventResult ViewKeyListener::handleEvent(TgEventData *eventData, const TgWindowInfo *windowInfo)
{
    if (eventData->m_type == TgEventType::EventTypeKeyPress || eventData->m_type == TgEventType::EventTypeKeyRelease) {
        const bool press = eventData->m_type == TgEventType::EventTypeKeyPress;
        const TgPressReleaseKey key = eventData->m_event.m_keyEvent.m_pressReleaseKey;
        if (key == TgPressReleaseKey::PressReleaseKey_Key_Left || key == TgPressReleaseKey::PressReleaseKey_Key_Right) {
            const int direction = key == TgPressReleaseKey::PressReleaseKey_Key_Left ? -1 : 1;
            const auto &callback = press ? onTurnStart : onTurnStop;
            if (callback) {
                callback(direction);
            }
            return TgEventResult::EventResultCompleted;
        }
        if (key == TgPressReleaseKey::PressReleaseKey_Key_Up || key == TgPressReleaseKey::PressReleaseKey_Key_Down) {
            const int direction = key == TgPressReleaseKey::PressReleaseKey_Key_Down ? -1 : 1;
            const auto &callback = press ? onPitchStart : onPitchStop;
            if (callback) {
                callback(direction);
            }
            return TgEventResult::EventResultCompleted;
        }
    }
    if (eventData->m_type == TgEventType::EventTypeCharacterCallback
        && eventData->m_event.m_keyEvent.m_pressReleaseKey == TgPressReleaseKey::PressReleaseKey_NormalKey) {
        const char key = static_cast<char>(eventData->m_event.m_keyEvent.m_key);
        if (key >= '0' && key <= '9' && onCameraSelected) {
            onCameraSelected(key - '0');
            return TgEventResult::EventResultCompleted;
        }
        if (key == 't' && onToggleDistances) {
            onToggleDistances();
            return TgEventResult::EventResultCompleted;
        }
    }
    return TgItem2d::handleEvent(eventData, windowInfo);
}
