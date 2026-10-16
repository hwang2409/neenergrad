import numpy as np
from engine.tensor import Tensor

class SGD:
    def __init__(self, lr: float = 0.1, params: List[Tensor] = None):
        self.lr = lr
        self.params = params if params else []
    
    def step(self):
        for tensor in self.params:
            tensor.data -= tensor.grad * self.lr

    def zero_grad(self):
        for tensor in self.params:
            tensor.grad = np.zeros(tensor.shape)

class Momentum:
    def __init__(self, lr: float = 0.1, mm: float = 0.9, params: List[Tensor] = None):
        self.lr = lr
        self.momentum = mm
        self.params = params if params else []
        self.state = []

        for tensor in self.params:
            self.state.append(np.zeros(tensor.shape))

    def step(self):
        for idx, tensor in enumerate(self.params):
            b = self.state[idx] * self.momentum + tensor.grad
            tensor.data -= self.lr * b
            self.state[idx] = b

    def zero_grad(self):
        for tensor in self.params:
            tensor.grad = np.zeros(tensor.shape)

class Adam:
    def __init__(self):
        pass
