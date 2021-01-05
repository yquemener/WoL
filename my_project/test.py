import random
import time
import sys                                                 

b="coucou"
print(b)

global a
a=1

l=list(locals().keys())
for ll in l:
    print(ll)
for i in range(100):
    a=str(i)
    print(a)
    time.sleep(0.5)