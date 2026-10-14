from engine.scalar import Value
from engine.mlp import MLP
from data.dataset import Dataset
from visualization.plot import Plot
from typing import List

def check_accuracy(mlp: MLP, moons: List[tuple[float, float, int]]):
    cnt = 0
    for x, y, label in moons:
        predicted = 1 if (mlp([Value(x),Value(y)])[0].data > 0) else -1
        cnt += (predicted == label)
    return float(cnt) / len(moons)

def f():
    N = 100
    plot = Plot(6,6)
    moons = Dataset.make_moons(N)
    xs, ys, labels = zip(*moons)
    #plot.scatter(xs, ys, color=labels, cmap="coolwarm")
    #plot.show()

    # two moons
    # MLP: 2 -> 16 -> 16 -> 1
    # 2 inputs: (x,y)
    # 1 output: predicted label
    mlp = MLP(
                [2,16,16,1],
                -1.0,
                +1.0,
                [Value.tanh, Value.tanh, None]
             )
    
    lr = 0.1
    steps = 100
    for k in range(steps):
        error = Value(0)
        for j in range(N):
            x, y, label = moons[j]
            output = mlp([Value(x), Value(y)])[0]
            error += (output-label) * (output-label)
        MSE = error / (N)
        MSE.backwards()
        params = mlp.parameters()
        for x in params:
            x.data -= x.grad * lr
            x.grad = 0

    print(check_accuracy(mlp, moons))
    
if __name__ == "__main__":
    f()
