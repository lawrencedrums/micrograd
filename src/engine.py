from collections.abc import Callable

import math

class Value:
    def __init__(
        self,
        data: float,
        grad: float = 0.0,
        label: str = "",
        _backward: Callable = lambda : None,
        _op: str = "",
        _children: tuple[Value, ...] = (),
    ) -> None:
        self.data = data
        self.grad = grad
        self.label = label
        self._backward = _backward
        self._op = _op
        self._children = set(_children)

    def backward(self) -> None:
        topo = []
        visited = set()

        def build_topo(node: Value) -> None:
            if node not in visited:
                visited.add(node)
                for child in node._children:
                    build_topo(child)
                topo.append(node)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def tanh(self) -> Value:
        n = self.data
        data = (math.exp(2*n) - 1)/(math.exp(2*n) + 1)

        def _backward():
            self.grad += (1 - out.data**2) * out.grad

        out = Value(
            data=data,
            _backward=_backward,
            _children=(self,),
            _op="tanh",
        )
        return out

    def exp(self) -> Value:
        def _backward():
            self.grad += out.data * out.grad

        out = Value(
            data=math.exp(self.data),
            _backward=_backward,
            _children=(self,),
            _op="exp",
        )
        return out

    def __repr__(self) -> str:
        return f"Value({self.data})"

    def __add__(self, other: Value | int | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out = Value(
            data=(self.data + other.data),
            _backward=_backward,
            _children=(self, other),
            _op='+',
        )
        return out

    def __radd__(self, other: Value | int | float) -> Value:  # other + self
        return self + other

    def __mul__(self, other: Value | int | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out = Value(
            data=(self.data * other.data),
            _backward = _backward,
            _children=(self, other),
            _op='*',
        )
        return out

    def __rmul__(self, other: Value | int | float) -> Value:  # other * self
        return self * other

    def __pow__(self, other: Value | int | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
    
        def _backward():
            self.grad += other.data * self.data ** (other.data - 1) * out.grad

        out = Value(
            data=self.data**other.data,
            _backward=_backward,
            _children=(self,),
            _op=f"**{other}"
        )
        return out

    def __truediv__(self, other: Value | int | float) -> Value:
        return self * other**-1

    def __neg__(self) -> Value:
        return self * -1

    def __sub__(self, other: Value | int | float) -> Value:
        return self + (-other)

