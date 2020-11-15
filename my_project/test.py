import random
import time
import sys

print("coucou")

print(self)
print(self.context)
print(self.context.scene)

self.context.scene.lock.acquire()
CodeRunnerEditorNode(parent=self.context.scene, filename="my_project/server.py")
self.context.scene.lock.release()