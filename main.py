from engine.scalar import Value

def g():
    a = Value(2)
    b = Value(-3)
    c = Value(5)

    d = a*b
    e = b+c
    f = d*e
    g = c*e
    L = f+g

    L.backwards()

    print(a,b,c)

if __name__ == "__main__":
    g()
