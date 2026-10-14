from collections import defaultdict
from typing import List
import math

class Value:
    def __init__(self, data: float, children: dict[Value, float] = {}):
        self.data = data
        self.children = children
        self.grad = 0

    def __mul__(self, other: Value) -> Value:
        other = other if isinstance(other, Value) else Value(other)

        return (Value(self.data * other.data, { self: other.data, other: self.data }) if self is not other 
                else Value(self.data * self.data, { self: 2 * self.data }))

    def __add__(self, other: Value) -> Value:
        other = other if isinstance(other, Value) else Value(other)

        return (Value(self.data + other.data, { self: 1, other: 1 }) if self is not other
                else Value(2*self.data, { self: 2 }))

    def __sub__(self, other: Value) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return (Value(self.data - other.data, { self: 1, other: -1 }) if self is not other
                else Value(0))

    def __pow__(self, other: int) -> Value:
        return (Value(self.data ** other, { self: other * (self.data ** (other-1)) }) if other != 0
                else Value(1))

    def __truediv__(self, other: Value) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return (Value(1) if self is other 
                else Value(self.data / other.data, { self: 1/other.data, other: -self.data / (other.data*other.data) }))

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
