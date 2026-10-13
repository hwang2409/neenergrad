from engine.scalar import Value

def f():
    x = Value(1)
    y = Value(2)
    z = Value(3)

    a = x+y
    b = a*z
    c = y**2
    f = b+c

    f.grad = 1
    f.backwards()

    print(x, y, z)

if __name__ == "__main__":
    f()
