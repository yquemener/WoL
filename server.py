import os

class Server:
    def __init__(self):
        self.world=list()
        self.programs=list()
        for m in os.listdir("pieces/"):
            print(m)
            code_str = "def func(ctxt):\n\t"
            for line in open("pieces/"+m).read().split("\n"):
                code_str += "\t"+line
            exec(code_str)
            self.programs.append(eval("func"))


    def list(self):
        print("Programs:")
        for p in self.programs:
            print(p)
        print("Objects:")
        for o in self.world:
            print(o)
