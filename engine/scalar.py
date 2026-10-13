from collections import defaultdict

class Value:
    def __init__(self, data: float, children: map[Value, float] = {}):
        self.data = data
        self.children = children
        self.grad = 0

    def __mul__(self, other: Value) -> Value:
        if self is other:
            return Value(self.data * self.data, { self: 2*self.data })
        return Value(self.data * other.data, { self: other.data, other: self.data })

    def __add__(self, other: Value) -> Value:
        if self is other:
            return Value(2*self.data, { self: 2 })
        return Value(self.data + other.data, { self: 1.0, other: 1.0 })

    def __sub__(self, other: Value) -> Value:
        if self is other:
            return Value(0)
        return Value(self.data - other.data, { self: 1.0, other: -1.0 })

    def __pow__(self, other: int) -> Value:
        assert(other > 1)

        return Value(self.data ** other, { self: other * (self.data**(other-1)) })

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

    def backwards(self) -> float:
        ll = self.topological()

        for x in ll:
            if x is self:
                x.grad = 1

            for c in x.children:
                c.grad += x.grad * x.children[c]
