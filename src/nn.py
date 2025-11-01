import random

from .engine import Value


class Neuron:
    def __init__(self, n_in: int) -> None:
        self.w: list[Value] = [Value(random.uniform(-1,1)) for _ in range(n_in)]
        self.b: Value = Value(random.uniform(-1,1))

    def __call__(self, x: list[Value]) -> Value:
        "w * x + b"
        act = sum([wi*xi for wi, xi in zip(self.w, x)], self.b)
        return act.tanh()

    def __repr___(self) -> str:
        return f"Neuron({self.w}, {self.b})"


class Layer:
    def __init__(self, n_in: int, n_out: int) -> None:
        self.neurons: list[Neuron] = [Neuron(n_in) for _ in range(n_out)]
        
    def __call__(self, x: list[Value]) -> list[Value]:
        return [n(x) for n in self.neurons]


class MLP:
    def __init__(self, n_in: int, n_outs: list[int]) -> None:
        sz = [n_in] + n_outs
        self.layers: list[Layer] = [Layer(sz[i], sz[i+1]) for i in range(len(n_outs))]

    def __call__(self, x: list[Value]) -> list[Value] | Value:
        for layer in self.layers:
            x = layer(x)
        return x
