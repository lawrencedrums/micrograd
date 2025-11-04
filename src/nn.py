import random

from micrograd.engine import Value


class Module:
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0

    def parameters(self) -> list[Value]:
        return []


class Neuron(Module):
    def __init__(self, n_in: int) -> None:
        self.w: list[Value] = [Value(random.uniform(-1,1)) for _ in range(n_in)]
        self.b: Value = Value(random.uniform(-1,1))

    def parameters(self) -> list[Value]:
        return self.w + [self.b]

    def __call__(self, x: list[Value]) -> Value:
        "w * x + b"
        act = sum([wi*xi for wi, xi in zip(self.w, x)], self.b)
        return act.tanh()

    def __repr___(self) -> str:
        return f"Neuron({self.w}, {self.b})"


class Layer(Module):
    def __init__(self, n_in: int, n_out: int) -> None:
        self.neurons: list[Neuron] = [Neuron(n_in) for _ in range(n_out)]

    def parameters(self) -> list[Value]:
        return [p for neuron in self.neurons for p in neuron.parameters()]

    def __call__(self, x: list[Value]) -> list[Value]:
        return [n(x) for n in self.neurons]


class MLP(Module):
    def __init__(self, n_in: int, n_outs: list[int]) -> None:
        sz = [n_in] + n_outs
        self.layers: list[Layer] = [Layer(sz[i], sz[i+1]) for i in range(len(n_outs))]

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def __call__(self, x: list[Value]) -> list[Value] | Value:
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x
