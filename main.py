from engine.scalar import Value
from data.dataset import Dataset
from visualization.plot import Plot

def f():
    plot = Plot(6,6)

    moons = Dataset.make_moons(100)
    xs, ys, labels = zip(*moons)

    plot.scatter(xs, ys, color=labels, cmap="coolwarm")
    plot.show()

if __name__ == "__main__":
    f()
