import importlib
import importlib.util
import os

class Context:
    def __init__(self):
        self.hp = 10

import tokenize
file_path = tokenize.__file__
module_name = tokenize.__name__

print(file_path)
print(module_name)

for m in os.listdir("modules/"):
    if not m.endswith(".py"):
        continue
    print(m)
    spec = importlib.util.spec_from_file_location("A", "/home/yves/Projects/musing/WoL/modules/"+m)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    c=Context()
    print(c.hp)
    module.A().update(c)
    print(c.hp)
