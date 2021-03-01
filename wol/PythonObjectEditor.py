from wol.SceneNode import SceneNode


class PythonObjectEditorNode(SceneNode):
    def __init__(self, parent, target):
        SceneNode.__init__(self, parent=parent)
        self.target = target
