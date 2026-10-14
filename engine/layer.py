from engine.neuron import Neuron

class Layer:
    def __init__(self,
                 n_inputs: int,
                 n_outputs: int,
                 lower: float = -1.0,
                 upper: float = +1.0,
                 nonlinearity: Callable[[Value], Value] = None):
        self.neurons = []

        for x in range(n_outputs):
            nn = Neuron(n_inputs, nonlinearity, lower, upper)
            self.neurons.append(nn)

    def __call__(self, inputs: List[Value]) -> List[Value]:
        outputs = []
        for neuron in self.neurons:
            result = neuron(inputs)
            outputs.append(result)
        return outputs

    def parameters(self) -> List[Value]:
        ll = []

        for neuron in self.neurons:
            ll.extend(neuron.parameters())
        return ll
