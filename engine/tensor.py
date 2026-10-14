import numpy as np
import numpy.typing as npt
from collections import defaultdict

class Tensor:
    def __init__(self, data: npt.ArrayLike, children: dict[Tensor, npt.ArrayLike] = None):
        self.data = np.asarray(data, dtype=np.float64)
        self.children = children if children is not None else {}
        self.shape = self.data.shape
        self.grad = np.zeros(self.shape)

    def __add__(self, other: Tensor) -> Tensor:
        assert(self.shape == other.shape)

        if self is other:
            return Tensor(2 * self.data, { self: np.full(self.shape, 2) })

        return Tensor(self.data + other.data,{ self: np.full(self.shape, 1), other: np.full(self.shape, 1) })

    def __mul__(self, other: Tensor) -> Tensor:
        assert(self.shape == other.shape)
    
        if self is other:
            return self.__pow__(2)

        return Tensor(self.data * other.data, { self: other.data.copy(), other: self.data.copy() })

    def __sub__(self, other: Tensor) -> Tensor:
        assert(self.shape == other.shape)

        if self is other:
            return Tensor.fill(self.shape, 0)

        return Tensor(self.data - other.data, { self: np.full(self.shape, 1), other: np.full(self.shape, -1) })

    def __pow__(self, exp: int) -> Tensor:
        if exp == 0:
            return Tensor.fill(self.shape, 1)

        return Tensor(np.power(self.data, exp), { self: exp * np.power(self.data, exp-1) })

    def __truediv__(self, other: Tensor) -> Tensor:
        assert(self.shape == other.shape)

        if self is other:
            return Tensor.fill(self.shape, 1)

        return Tensor(self.data / other.data, { self: np.full(self.shape, 1) / other.data, other: -self.data / (other.data * other.data) })

    def exp(self) -> Tensor:
        x = np.exp(self.data)
        return Tensor(x, { self: x })

    def log(self) -> Tensor:
        assert(np.all(self.data > 0))
        return Tensor(np.log(self.data), { self: 1 / self.data })

    def tanh(self) -> Tensor:
        x = np.tanh(self.data)

        return Tensor(x, { self: np.full(self.shape, 1) - x*x })

    def relu(self) -> Tensor:
        return Tensor(np.maximum(0, self.data), { self: (self.data > 0).astype(float) })

    def __neg__(self):
        return self.__mul__(Tensor.fill(self.shape, -1))

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, data={self.data})"

    def sum(self) -> Tensor:
        return Tensor(np.sum(self.data), { self: np.full(self.shape, 1) })

    def topological(self) -> List[Tensor]:
        visited = defaultdict(bool)
        ll = []

        def dfs(node: Tensor):
            if visited[node]:
                return
            
            for child in node.children:
                dfs(child)
            ll.append(node)
            visited[node] = True
        dfs(self)
        return ll[::-1]

    def backwards(self) -> None:
        ll = self.topological()

        for x in ll:
            if x is self:
                x.grad = np.full(self.shape, 1)

            for child in x.children:
                child.grad += x.grad * x.children[child]

    @classmethod
    def zeros(cls, shape: tuple[int, ...]) -> Tensor:
        return Tensor(np.zeros(shape))

    @classmethod
    def fill(cls, shape: tuple[int, ...], value: float) -> Tensor:
        return Tensor(np.full(shape, value))

    @classmethod
    def random(cls, shape: tuple[int, ...], lower: float = -1.0, upper: float = +1.0) -> Tensor:
        rng = np.random.default_rng()
        return Tensor(rng.uniform(lower, upper, size=shape))
