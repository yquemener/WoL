import wol
import PyQt5


obj_0 = wol.GeomNodes.Grid(parent=context.scene,name="Grid")
obj_0.position = PyQt5.QtGui.QVector3D(0.0, 0.0, 0.0)
obj_0.orientation = PyQt5.QtGui.QQuaternion(0.7071067690849304, 0.0, 0.0, 0.7071067690849304)
obj_0.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_0.properties = {}
obj_0.color = PyQt5.QtGui.QVector4D(0.20000000298023224, 0.20000000298023224, 0.6000000238418579, 1.0)
obj_0.visible = True

obj_1 = wol.GeomNodes.Avatar(parent=context.scene,name="avatar_tata")
obj_1.position = PyQt5.QtGui.QVector3D(0.0, 0.0, 0.0)
obj_1.orientation = PyQt5.QtGui.QQuaternion(0.9928369522094727, 0.030225396156311035, -0.11553698778152466, 0.0035172980278730392)
obj_1.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_1.properties = {}
obj_1.visible = True

obj_2 = wol.Notebook.NotebookNode(parent=context.scene,name="Notebook9")
obj_2.position = PyQt5.QtGui.QVector3D(-5.704586029052734, 6.46835994720459, 0.6592416763305664)
obj_2.orientation = PyQt5.QtGui.QQuaternion(0.9857195615768433, 0.025659019127488136, 0.16638614237308502, -0.00376404938288033)
obj_2.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_2.properties = {}
obj_2.visible = True
cell = obj_2.add_cell(0)
cell.set_text('import torch\nimport torch.nn as nn\nimport torch.optim as optim\nimport torch.nn.functional as F\nfrom torch.utils.data import DataLoader\nimport torchvision.transforms as transforms\nimport torchvision.datasets as datasets\nimport time')
cell = obj_2.add_cell(1)
cell.set_text('BATCH_SIZE = 64\n\nclass MNISTPerceptron(nn.Module):\n    def __init__(self):\n        super(MNISTPerceptron, self).__init__()\n        self.fc1 = nn.Linear(28*28, 128)\n        self.fc2 = nn.Linear(128, 64)\n        self.fc3 = nn.Linear(64, 10)\n        self.probe = torch.zeros(28, 28, dtype=torch.float32)\n        \n    def forward(self, x):\n        self.probe[:,:] = x[0,:,:]\n        x = x.view(-1, 28*28)\n        x = F.relu(self.fc1(x))\n        x = F.relu(self.fc2(x))\n        x = self.fc3(x)\n        return x')
cell = obj_2.add_cell(2)
cell.set_text('def test_epoch(model, device, test_loader, criterion):\n    model.eval()\n    test_loss = 0\n    correct = 0\n    with torch.no_grad():\n        for data, target in test_loader:\n            data, target = data.to(device), target.to(device)\n            output = model(data)\n            test_loss += criterion(output, target).item()\n            pred = output.argmax(dim=1, keepdim=True)\n            correct += pred.eq(target.view_as(pred)).sum().item()\n    \n    test_loss /= len(test_loader)\n    accuracy = 100. * correct / len(test_loader.dataset)\n    return test_loss, accuracy')
cell = obj_2.add_cell(3)
cell.set_text("device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nprint(f'Using device: {device}')\n    \ntransform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])\n    \ntrain_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)\ntest_dataset = datasets.MNIST('data', train=False, transform=transform)\n    \ntrain_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)\ntest_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)\n \nmodel = MNISTPerceptron().to(device)\noptimizer = optim.Adam(model.parameters(), lr=0.001)\ncriterion = nn.CrossEntropyLoss()\n\ncontext.model=model")
cell = obj_2.add_cell(4)
cell.set_text("print('Starting training...')\nfor epoch in range(10):\n    start = time.time()\n    last_frame = time.time()\n    model.train()\n    total_loss = 0\n    correct = 0\n    for batch_idx, (data, target) in enumerate(train_loader):\n        data, target = data.to(device), target.to(device)\n        optimizer.zero_grad()\n        output = model(data)\n        loss = criterion(output, target)\n        loss.backward()\n        optimizer.step()\n        \n        total_loss += loss.item()\n        pred = output.argmax(dim=1, keepdim=True)\n        correct += pred.eq(target.view_as(pred)).sum().item()\n\n        if batch_idx % 100 == 0:\n            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')\n\n\n    end = time.time()\n    print(f'Epoch {epoch+1}/10: Time: {end - start:.2f}s')\n    accuracy = 100. * correct / len(train_loader.dataset)\n    avg_loss = total_loss / len(train_loader)\n\n    test_loss, test_acc = test_epoch(model, device, test_loader, criterion)\n    \n    print(f'Epoch {epoch+1}/10:')\n    print(f'  Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.2f}%')\n    print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')\n    print('-' * 50)")
cell = obj_2.add_cell(5)
cell.set_text('\n\nimport wol\n\ncm=wol.CudaMemoryNode.CudaMemoryNode(parent=scene)')
cell = obj_2.add_cell(6)
cell.set_text('scene.add_child(cm)')
cell = obj_2.add_cell(7)
cell.set_text('cm.associate_tensor(model.probe)\n#cm.associate_tensor(model.fc3.weight)')
cell = obj_2.add_cell(8)
cell.set_text('cm.position = QVector3D(0, 2, 0)')
cell = obj_2.add_cell(9)
cell.set_text('print(12)')
cell = obj_2.add_cell(10)
cell.set_text('for model. ')
cell = obj_2.add_cell(11)
cell.set_text('')
cell = obj_2.layout()

obj_3 = wol.Notebook.NotebookNode(parent=context.scene,name="Notebook22")
obj_3.position = PyQt5.QtGui.QVector3D(-7.1454620361328125, 4.555408000946045, 2.5535812377929688)
obj_3.orientation = PyQt5.QtGui.QQuaternion(0.9738142490386963, -0.03325330466032028, 0.22487853467464447, 0.003103331895545125)
obj_3.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_3.properties = {}
obj_3.visible = True
cell = obj_3.add_cell(0)
cell.set_text('import wol.CudaMemoryNode as cmn')
cell = obj_3.add_cell(1)
cell.set_text('a=None\nfor i,p in enumerate(context.model.parameters()):\n\tprint(p.shape)\n\tif(len(p.shape))==1:\n\t\tcontinue\n\tcm=cmn.CudaMemoryNode(parent=scene)\n\tscene.add_child(cm)\n\tcm.associate_tensor(p)\n\tcm.value_amplifier = 20.0\n\tcm.position=QVector3D(3*i+3, 3, 0)\n\tif i==0:\n\t\ta=cm\n\t\tcm.reshape=(28,28,128)\n\t\tcm.update(1)')
cell = obj_3.add_cell(2)
cell.set_text('for c in scene.children:\n\tprint(c.name, type(c), getattr(c,"value_amplifier",0))')
cell = obj_3.add_cell(3)
cell.set_text('print(a.reshape)')
cell = obj_3.add_cell(4)
cell.set_text('print(a.scale[1]/0.05)')
cell = obj_3.add_cell(5)
cell.set_text('a.reshape=(28,28,128)')
cell = obj_3.add_cell(6)
cell.set_text('a.update(1)')
cell = obj_3.add_cell(7)
cell.set_text('print(a.tensor.shape)')
cell = obj_3.add_cell(8)
cell.set_text('import torch')
cell = obj_3.add_cell(9)
cell.set_text('torch.save(a.tensor, "layer1.pt")')
cell = obj_3.add_cell(10)
cell.set_text('')
cell = obj_3.layout()








