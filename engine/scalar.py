from collections import defaultdict
from typing import List
import math

class Value:
    def __init__(self, data: float, children: dict[Value, float] = {}):
        self.data = data
        self.children = children
        self.grad = 0

    def __mul__(self, other: Value) -> Value:
        if self is other:
            return Value(self.data * self.data, { self: 2*self.data })
        if not isinstance(other, Value):
            other = Value(other)
        return Value(self.data * other.data, { self: other.data, other: self.data })

    def __add__(self, other: Value) -> Value:
        if self is other:
            return Value(2*self.data, { self: 2 })
        if not isinstance(other, Value):
            other = Value(other)
        return Value(self.data + other.data, { self: 1.0, other: 1.0 })

    def __sub__(self, other: Value) -> Value:
        if self is other:
            return Value(0)
        if not isinstance(other, Value):
            other = Value(other)
        return Value(self.data - other.data, { self: 1.0, other: -1.0 })

    def __pow__(self, other: int) -> Value:
        if other == 0:
            return Value(1)

        result = self.data ** other
        return Value(result, { self: other * (self.data ** (other-1)) })

    def __truediv__(self, other: Value) -> Value:
        if not isinstance(other, Value):
            other = Value(other)
        assert(other.data != 0)
        if self is other:
            return Value(1)
        return Value(self.data / other.data, { self: 1/other.data, other: -self.data / (other.data*other.data) })

    def exp(self) -> Value:
        x = math.exp(self.data)
        return Value(x, { self: x })

    def log(self) -> Value:
        assert(self.data > 0)
        return Value(math.log(self.data), { self: 1/self.data })

    def tanh(self) -> Value:
        tan = math.tanh(self.data)
        return Value(tan, { self: 1 - tan*tan })

    def relu(self) -> Value:
        return Value(max(self.data, 0), { self: 1 if self.data > 0 else 0 })

    def __radd__(self, other) -> Value:
        return self + other

    def __rsub__(self, other) -> Value:
        return Value(other) - self

    def __rmul__(self, other) -> Value:
        return self * other

    def __rtruediv__(self, other) -> Value:
        return Value(other) / self

    def __neg__(self):
        return self.__mul__(-1)

    def __repr__(self) -> str:
        return f"Value({self.data}, grad={self.grad})"

    def topological(self) -> List[Value]:
        visited = defaultdict(bool)
        ordered = []

        def dfs(node: Value):
            if visited[node]:
                return

            for c in node.children:
                dfs(c)
            ordered.append(node)
            visited[node] = True
        dfs(self)
        return ordered[::-1]

    def backwards(self) -> None:
        ll = self.topological()

        for x in ll:
            if x is self:
                x.grad = 1

            for c in x.children:
                c.grad += x.grad * x.children[c]
