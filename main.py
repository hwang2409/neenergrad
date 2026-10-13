from engine.scalar import Value

def f():
    a = Value(2)
    b = Value(1)
    c = a+b
    d = b+1
    e = c*d

    e.grad = 1
    e.backwards()

    print(a,b,c)

if __name__ == "__main__":
    f()
