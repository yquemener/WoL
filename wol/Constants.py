class UserActions(int):
    Edit = 1
    Activate = 2
    Unselect = 3
    Grab = 4
    Release = 5
    Invoke_Console = 6
    Save = 6
    Load = 7
    Snap_To_90 = 8
    Move_Forward = 9
    Move_Back = 10
    Strafe_Left = 11
    Strafe_Right = 12
    Move_Up = 13
    Move_Down = 14
    Copy = 15
    Cut = 16
    Paste = 17
    Change_Cursor_Mode = 18


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
