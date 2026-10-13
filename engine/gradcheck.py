from collections.abc import Callable
from engine.scalar import Value
from typing import List

def gradcheck(f: Callable[..., Value], params: List[Value]):
    h = 1e-5
    out = f(*params)
    out.backwards()

    grads = {}
    numeric_grads = {}
    for param in params:
        grads[param] = param.grad

    for idx in range(len(params)):
        value = params[idx]
        prev = value.data
        p0, p1 = prev+h, prev-h

        params[idx] = Value(p0)
        o0 = f(*params).data
        params[idx] = Value(p1)
        o1 = f(*params).data
        params[idx] = value

        numeric = (o0 - o1) / (2*h)
        numeric_grads[value] = numeric

    for p in params:
        grad, numeric = grads[p], numeric_grads[p]
        check = abs(grad - numeric) / max(1, max(abs(grad), abs(numeric)))

        if check >= 1e-6:
            return False
    return True
