from engine.scalar import Value
from typing import List
from collections.abc import Callable
import random

class Neuron:
    def __init__(self, 
                 wlength: int,
                 nonlinearity: Callable[[Value], Value] = None,
                 lower: float = -1.0,
                 upper: float = +1.0):
        self.weights = []
        self.bias = Value(0)
        self.nonlinearity = nonlinearity

        for x in range(wlength):
            self.weights.append(Value(random.uniform(lower, upper)))

    def __call__(self, inputs: List[Value]) -> Value: 
        assert(len(inputs) == len(self.weights))

        L = len(inputs)
        result = Value(0)
        for idx in range(L):
            result += self.weights[idx] * inputs[idx]
        result += self.bias

        return result if self.nonlinearity is None else self.nonlinearity(result)

    def parameters(self) -> List[Value]:
        return self.weights + [self.bias]
