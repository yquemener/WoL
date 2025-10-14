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

obj_2 = wol.Notebook.NotebookNode(parent=context.scene,name="Notebook5")
obj_2.position = PyQt5.QtGui.QVector3D(-1.4135291576385498, 5.406454563140869, 1.3666977882385254)
obj_2.orientation = PyQt5.QtGui.QQuaternion(0.9890741109848022, -0.0030229396652430296, -0.14738093316555023, -0.0014766992535442114)
obj_2.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_2.properties = {}
obj_2.visible = True
cell = obj_2.add_cell(0)
cell.set_text("print('Hi!')")
cell = obj_2.add_cell(1)
cell.set_text('a=12')
cell = obj_2.add_cell(2)
cell.set_text('print(a)')
cell = obj_2.add_cell(3)
cell.set_text('')
cell = obj_2.layout()

obj_3 = wol.Notebook.NotebookNode(parent=context.scene,name="Notebook8")
obj_3.position = PyQt5.QtGui.QVector3D(0.17542731761932373, 5.30002498626709, 2.951037645339966)
obj_3.orientation = PyQt5.QtGui.QQuaternion(0.9768542647361755, -0.03740271180868149, -0.2104564756155014, -0.008058151230216026)
obj_3.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_3.properties = {}
obj_3.visible = True
cell = obj_3.add_cell(0)
cell.set_text("print('Hi!')")
cell = obj_3.add_cell(1)
cell.set_text('')
cell = obj_3.layout()

obj_4 = wol.Notebook.NotebookNode(parent=context.scene,name="Notebook8")
obj_4.position = PyQt5.QtGui.QVector3D(-5.197747230529785, 4.4042463302612305, 0.562042236328125)
obj_4.orientation = PyQt5.QtGui.QQuaternion(0.9993668794631958, -0.027593040838837624, -0.01574418880045414, 0.016017453745007515)
obj_4.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_4.properties = {}
obj_4.visible = True
cell = obj_4.add_cell(0)
cell.set_text("print('Hi!')")
cell = obj_4.add_cell(1)
cell.set_text('print(context)')
cell = obj_4.add_cell(2)
cell.set_text('print(scene)')
cell = obj_4.add_cell(3)
cell.set_text('for c in scene.children:\n\tprint(c.name, type(c))')
cell = obj_4.add_cell(4)
cell.set_text('s')
cell = obj_4.add_cell(5)
cell.set_text('')
cell = obj_4.layout()

obj_5 = wol.CudaMemoryNode.CudaMemoryNode(parent=context.scene,name="Card")
obj_5.position = PyQt5.QtGui.QVector3D(0.0, 0.0, 0.0)
obj_5.orientation = PyQt5.QtGui.QQuaternion(1.0, 0.0, 0.0, 0.0)
obj_5.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_5.properties = {}
obj_5.visible = True

obj_6 = wol.CudaMemoryNode.CudaMemoryNode(parent=context.scene,name="Card")
obj_6.position = PyQt5.QtGui.QVector3D(0.0, 0.0, 0.0)
obj_6.orientation = PyQt5.QtGui.QQuaternion(1.0, 0.0, 0.0, 0.0)
obj_6.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_6.properties = {}
obj_6.visible = True

obj_7 = wol.CudaMemoryNode.CudaMemoryNode(parent=context.scene,name="Card")
obj_7.position = PyQt5.QtGui.QVector3D(0.0, 0.0, 0.0)
obj_7.orientation = PyQt5.QtGui.QQuaternion(1.0, 0.0, 0.0, 0.0)
obj_7.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_7.properties = {}
obj_7.visible = True

obj_8 = wol.Notebook.NotebookNode(parent=context.scene,name="Notebook10")
obj_8.position = PyQt5.QtGui.QVector3D(-7.2228312492370605, 2.6541788578033447, 10.151665687561035)
obj_8.orientation = PyQt5.QtGui.QQuaternion(0.7969168424606323, 0.01666729338467121, 0.6019994020462036, -0.04735400527715683)
obj_8.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_8.properties = {}
obj_8.visible = True
cell = obj_8.add_cell(0)
cell.set_text('scene.add_child(wol.CudaMemoryNode.CudaMemoryNode())')
cell = obj_8.add_cell(1)
cell.set_text('import wol\n')
cell = obj_8.add_cell(2)
cell.set_text('print()')
cell = obj_8.add_cell(3)
cell.set_text('')
cell = obj_8.layout()

obj_9 = wol.CudaMemoryNode.CudaMemoryNode(parent=context.scene,name="Card")
obj_9.position = PyQt5.QtGui.QVector3D(0.0, 0.0, 0.0)
obj_9.orientation = PyQt5.QtGui.QQuaternion(1.0, 0.0, 0.0, 0.0)
obj_9.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_9.properties = {}
obj_9.visible = True

obj_10 = wol.CudaMemoryNode.CudaMemoryNode(parent=context.scene,name="Card")
obj_10.position = PyQt5.QtGui.QVector3D(0.0, 2.0, 0.0)
obj_10.orientation = PyQt5.QtGui.QQuaternion(1.0, 0.0, 0.0, 0.0)
obj_10.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_10.properties = {}
obj_10.visible = True

obj_11 = wol.CudaMemoryNode.CudaMemoryNode(parent=context.scene,name="Card")
obj_11.position = PyQt5.QtGui.QVector3D(0.0, 2.0, 0.0)
obj_11.orientation = PyQt5.QtGui.QQuaternion(1.0, 0.0, 0.0, 0.0)
obj_11.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_11.properties = {}
obj_11.visible = True

obj_12 = wol.CudaMemoryNode.CudaMemoryNode(parent=context.scene,name="Card")
obj_12.position = PyQt5.QtGui.QVector3D(0.0, 2.0, 0.0)
obj_12.orientation = PyQt5.QtGui.QQuaternion(1.0, 0.0, 0.0, 0.0)
obj_12.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_12.properties = {}
obj_12.visible = True

obj_13 = wol.Notebook.NotebookNode(parent=context.scene,name="Notebook9")
obj_13.position = PyQt5.QtGui.QVector3D(-5.871818542480469, 5.716098308563232, 5.717964172363281)
obj_13.orientation = PyQt5.QtGui.QQuaternion(0.9591037034988403, 0.04384626820683479, 0.2796023190021515, -0.004491106607019901)
obj_13.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_13.properties = {}
obj_13.visible = True
cell = obj_13.add_cell(0)
cell.set_text('import torch\nimport torch.nn as nn\nimport torch.optim as optim\nimport torch.nn.functional as F\nfrom torch.utils.data import DataLoader\nimport torchvision.transforms as transforms\nimport torchvision.datasets as datasets\nimport time')
cell = obj_13.add_cell(1)
cell.set_text('BATCH_SIZE = 64\n\nclass MNISTPerceptron(nn.Module):\n    def __init__(self):\n        super(MNISTPerceptron, self).__init__()\n        self.fc1 = nn.Linear(28*28, 128)\n        self.fc2 = nn.Linear(128, 64)\n        self.fc3 = nn.Linear(64, 10)\n        self.probe = torch.zeros(BATCH_SIZE, 128, dtype=torch.float32)\n        \n    def forward(self, x):\n        x = x.view(-1, 28*28)\n        x = F.relu(self.fc1(x))\n        if self.probe.shape[0] == x.shape[0]:\n            self.probe[:,:] = x[:,:]\n        x = F.relu(self.fc2(x))\n        x = self.fc3(x)\n        return x')
cell = obj_13.add_cell(2)
cell.set_text('def test_epoch(model, device, test_loader, criterion):\n    model.eval()\n    test_loss = 0\n    correct = 0\n    with torch.no_grad():\n        for data, target in test_loader:\n            data, target = data.to(device), target.to(device)\n            output = model(data)\n            test_loss += criterion(output, target).item()\n            pred = output.argmax(dim=1, keepdim=True)\n            correct += pred.eq(target.view_as(pred)).sum().item()\n    \n    test_loss /= len(test_loader)\n    accuracy = 100. * correct / len(test_loader.dataset)\n    return test_loss, accuracy')
cell = obj_13.add_cell(3)
cell.set_text("device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\nprint(f'Using device: {device}')\n    \ntransform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])\n    \ntrain_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)\ntest_dataset = datasets.MNIST('data', train=False, transform=transform)\n    \ntrain_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)\ntest_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)\n \nmodel = MNISTPerceptron().to(device)\noptimizer = optim.Adam(model.parameters(), lr=0.001)\ncriterion = nn.CrossEntropyLoss()")
cell = obj_13.add_cell(4)
cell.set_text("print('Starting training...')\nfor epoch in range(10):\n    start = time.time()\n    last_frame = time.time()\n    model.train()\n    total_loss = 0\n    correct = 0\n    for batch_idx, (data, target) in enumerate(train_loader):\n        data, target = data.to(device), target.to(device)\n        optimizer.zero_grad()\n        output = model(data)\n        loss = criterion(output, target)\n        loss.backward()\n        optimizer.step()\n        \n        total_loss += loss.item()\n        pred = output.argmax(dim=1, keepdim=True)\n        correct += pred.eq(target.view_as(pred)).sum().item()\n\n        if batch_idx % 100 == 0:\n            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')\n\n\n    end = time.time()\n    print(f'Epoch {epoch+1}/10: Time: {end - start:.2f}s')\n    accuracy = 100. * correct / len(train_loader.dataset)\n    avg_loss = total_loss / len(train_loader)\n\n    test_loss, test_acc = test_epoch(model, device, test_loader, criterion)\n    \n    print(f'Epoch {epoch+1}/10:')\n    print(f'  Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.2f}%')\n    print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')\n    print('-' * 50)")
cell = obj_13.add_cell(5)
cell.set_text('import wol\n\ncm=wol.CudaMemoryNode.CudaMemoryNode()')
cell = obj_13.add_cell(6)
cell.set_text('scene.add_child(cm)')
cell = obj_13.add_cell(7)
cell.set_text('cm.associate_tensor(model.probe)')
cell = obj_13.add_cell(8)
cell.set_text('cm.position = QVector3D(0, 2, 0)')
cell = obj_13.add_cell(9)
cell.set_text('')
cell = obj_13.layout()

obj_14 = wol.CudaMemoryNode.CudaMemoryNode(parent=context.scene,name="Card")
obj_14.position = PyQt5.QtGui.QVector3D(0.0, 2.0, 0.0)
obj_14.orientation = PyQt5.QtGui.QQuaternion(1.0, 0.0, 0.0, 0.0)
obj_14.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_14.properties = {}
obj_14.visible = True

obj_15 = wol.CudaMemoryNode.CudaMemoryNode(parent=context.scene,name="Card")
obj_15.position = PyQt5.QtGui.QVector3D(0.0, 2.0, 0.0)
obj_15.orientation = PyQt5.QtGui.QQuaternion(1.0, 0.0, 0.0, 0.0)
obj_15.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_15.properties = {}
obj_15.visible = True

obj_16 = wol.CudaMemoryNode.CudaMemoryNode(parent=context.scene,name="Card")
obj_16.position = PyQt5.QtGui.QVector3D(0.0, 2.0, 0.0)
obj_16.orientation = PyQt5.QtGui.QQuaternion(1.0, 0.0, 0.0, 0.0)
obj_16.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_16.properties = {}
obj_16.visible = True



obj_19 = wol.CudaMemoryNode.CudaMemoryNode(parent=context.scene,name="Card")
obj_19.position = PyQt5.QtGui.QVector3D(0.0, 2.0, 0.0)
obj_19.orientation = PyQt5.QtGui.QQuaternion(1.0, 0.0, 0.0, 0.0)
obj_19.scale = PyQt5.QtGui.QVector3D(1.0, 1.0, 1.0)
obj_19.properties = {}
obj_19.visible = True
