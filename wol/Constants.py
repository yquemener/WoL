class UserActions(int):
    Edit = 1
    Activate = 2
    Unselect = 3
    Grab = 4
    Release = 5
    Invoke_Console = 6
    Save = 7
    Load = 8
    Snap_To_90 = 9
    Move_Forward = 10
    Move_Back = 11
    Strafe_Left = 12
    Strafe_Right = 13
    Move_Up = 14
    Move_Down = 15
    Copy = 16
    Cut = 17
    Paste = 18
    Change_Cursor_Mode = 19
    Execute = 20


class Events(int):
    TextChanged = 1001
    Clicked = 1002  #TODO
    GotFocus = 1003
    LostFocus = 1004
    AnimationFinished = 1005
    AppClose = 1006
    Ungrabbed = 1007
    GeometryChanged = 1008


class MappingTypes(int):
    KeyDown = 1
    KeyPressed = 2
    MouseButtonClicked = 3


