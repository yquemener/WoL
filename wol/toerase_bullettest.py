import pybullet as p
import os

import math

useGui = True

if (useGui):
  p.connect(p.GUI)
else:
  p.connect(p.DIRECT)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setAdditionalSearchPath(os.getcwd()+"/../urdf/")
print(os.getcwd()+"/urdf/")

p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
#p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)

#p.loadURDF("samurai.urdf")
p.loadURDF("sphere.urdf", [3, 3, 1])

rayFrom = []
rayTo = []
rayIds = []

numRays = 128

rayLen = 13

rayHitColor = [1, 0, 0]
rayMissColor = [0, 1, 0]

replaceLines = True

for i in range(numRays):
  rayFrom.append([0, 0, 1])
  rayTo.append([
      rayLen * math.sin(2. * math.pi * float(i) / numRays),
      rayLen * math.cos(2. * math.pi * float(i) / numRays), 1
  ])
  if (replaceLines):
    rayIds.append(p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor))
  else:
    rayIds.append(-1)


numSteps = 327680


for i in range(numSteps):
  p.stepSimulation()
  results = p.rayTestBatch(rayFrom, rayTo)

  if (useGui):
    if (not replaceLines):
      p.removeAllUserDebugItems()

    for i in range(numRays):
      hitObjectUid = results[i][0]

      if (hitObjectUid < 0):
        hitPosition = [0, 0, 0]
        p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor, replaceItemUniqueId=rayIds[i])
      else:
        hitPosition = results[i][3]
        p.addUserDebugLine(rayFrom[i], hitPosition, rayHitColor, replaceItemUniqueId=rayIds[i])

  #time.sleep(1./240.)

