from engine.layer import Layer

class MLP:
    # define the last dimension as the 
    # number of outputs of the MLP
    def __init__(self,
                 dimensions: List[int],
                 lower: float = -1.0,
                 upper: float = +1.0,
                 nonlins: List[Callable[[Value], Value]] = None):
        self.layers = []
        
        for x in range(0, len(dimensions) - 1):
            d0, d1 = dimensions[x], dimensions[x+1]
            f = nonlins[x] if nonlins is not None and x < len(nonlins) else None
            ll = Layer(d0, d1, lower, upper, f)
            self.layers.append(ll)

    def __call__(self, inputs: List[Value]) -> List[Value]:
        args = inputs
        for layer in self.layers:
            args = layer(args)
        return args

    def parameters(self) -> List[Value]:
        ll = []

        for layer in self.layers:
            ll.extend(layer.parameters())
        return ll
