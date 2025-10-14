#!/usr/bin/env python3

import time 
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from cuda import cudart
import cupy as cp
import numpy as np
import glfw
from OpenGL.GL import *


VERT = """
#version 430
in vec2 pos;
out vec2 uv;
void main() { gl_Position = vec4(pos, 0, 1); uv = pos * 0.5 + 0.5; }
"""

FRAG = """
#version 430
in vec2 uv;
out vec4 col;
layout(std430, binding=0) readonly buffer Buf { float data[]; };
uniform int w;
void main() { 
    int x = int(uv.x * w);
    int y = int(uv.y * w);
    int i = (y * w  + x);
    col = vec4(data[i], data[i], data[i], 1.0); 
}
"""

BATCH_SIZE = 64

class MNISTPerceptron(nn.Module):
    def __init__(self):
        super(MNISTPerceptron, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.probe = torch.zeros(BATCH_SIZE, 128, dtype=torch.float32)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        if self.probe.shape[0] == x.shape[0]:
            self.probe[:,:] = x[:,:]
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def test_epoch(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

quad = np.array([-1,-1, 1,-1, -1,1, 1,1], dtype=np.float32)


def main():
    if not glfw.init():
        exit()

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    window = glfw.create_window(512, 512, "Minimal Zero-Copy", None, None)
    glfw.make_context_current(window)
    shader = glCreateProgram()
    vs = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vs, VERT)
    glCompileShader(vs)
    fs = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fs, FRAG)
    glCompileShader(fs)
    glAttachShader(shader, vs)
    glAttachShader(shader, fs)
    glLinkProgram(shader)
    glUseProgram(shader)

    
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 2, GL_FLOAT, False, 8, None)
    glEnableVertexAttribArray(0)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    model = MNISTPerceptron().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()


    # Store model.fc1.weight in ssbo with zero copy
    ssbo = glGenBuffers(1)
    print(f'SSBO: {ssbo}')
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
    glBufferData(GL_SHADER_STORAGE_BUFFER, model.fc1.weight.numel()*4, None, GL_DYNAMIC_DRAW)
    flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard
    gres = cudart.cudaGraphicsGLRegisterBuffer(ssbo, flags)[1]

    cudart.cudaGraphicsMapResources(1, gres, None)
    err, ptr, size = cudart.cudaGraphicsResourceGetMappedPointer(gres)
    print(f'SSBO pointer: {ptr}, error: {err}')
    mem = cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(ptr, size, None), 0)
    img = cp.ndarray(model.fc1.weight.shape, dtype=cp.float32, memptr=mem)
    # model.probe.data = torch.as_tensor(img, device='cuda')
    model.fc1.weight.data = torch.as_tensor(img, device='cuda')



    
    print('Starting training...')
    for epoch in range(10):
        start = time.time()
        last_frame = time.time()
        model.train()
        total_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            cudart.cudaDeviceSynchronize()
            # cudart.cudaGraphicsUnmapResources(1, gres, None)

            now = time.time()
            if now - last_frame > 1.0/60:
                glClear(GL_COLOR_BUFFER_BIT)
                glUniform1i(glGetUniformLocation(shader, "w"), BATCH_SIZE)
                glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo)
                glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
                last_frame = now

                glfw.swap_buffers(window)
                glfw.poll_events()
                if glfw.window_should_close(window):
                    sys.exit()


            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')


        end = time.time()
        print(f'Epoch {epoch+1}/10: Time: {end - start:.2f}s')
        accuracy = 100. * correct / len(train_loader.dataset)
        avg_loss = total_loss / len(train_loader)

        test_loss, test_acc = test_epoch(model, device, test_loader, criterion)
        
        print(f'Epoch {epoch+1}/10:')
        print(f'  Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.2f}%')
        print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        print('-' * 50)


    
    print('Training completed!')

if __name__ == '__main__':
    main()

