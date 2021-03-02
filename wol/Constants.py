class UserActions(int):
    Edit = 1
    Activate = 2
    Unselect = 3
    Grab = 4
    Release = 5
    Invoke_Console = 6
    Save = 6
    Snap_To_90 = 7
    Move_Forward = 8
    Move_Back = 9
    Strafe_Left = 10
    Strafe_Right = 11
    Move_Up = 12
    Move_Down = 13
    Copy = 14
    Cut = 15
    Paste = 16


class MappingTypes(int):
    KeyDown = 1
    KeyPressed = 2
    MouseButtonClicked = 3


class Events(int):
    TextChanged = 1
    Clicked = 2  #TODO
    GotFocus = 3
    LostFocus = 4
    AnimationFinished = 5
