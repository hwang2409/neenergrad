class Value:
    def __init__(self, data: float, children: map[Value, float] = {}):
        self.data = data
        self.children = children
        self.grad = None
    
    def __mul__(self, other: Value) -> Value:
        return Value(
                        self.data * other.data, 
                        {
                            self: other.data,
                            other: self.data
                        }
                    )

    def __add__(self, other: Value) -> Value:
        return Value(
                        self.data + other.data, 
                        {
                            self: 1.0, 
                            other: 1.0
                        }
                    )

    def __sub__(self, other: Value) -> Value:
        return Value(
                        self.data - other.data,
                        {
                            self: 1.0, 
                            other: -1.0
                        }
                    )

    def __pow__(self, other: int) -> Value:
        assert(other > 1)

        return Value(
                        self.data ** other,
                        {
                            self: other * (self.data**(other-1))
                        }
                    )

    def __repr__(self) -> str:
        return f"Value({self.data}, grad={self.grad})"

    def backwards(self) -> float:
        for c in self.children:
            if c.grad is None:
                c.grad = 0
            c.grad += self.grad * self.children[c]

        for c in self.children:
            c.backwards()
