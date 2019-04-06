import os

class Context:
    def __init__(self):
        self.hp = 10


for m in os.listdir("pieces/"):
    code_str = "def func(ctxt):\n\t"
    for line in open("pieces/"+m).read().split("\n"):
        code_str += "\t"+line
    exec(code_str)
c=Context()
print(c.hp)
func(c)
print(c.hp)
