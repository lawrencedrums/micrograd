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

    def tanh(self) -> Value:
        n = self.data
        data = (math.exp(2*n) - 1)/(math.exp(2*n) + 1)

        def _backward():
            self.grad = (1 - out.data**2) * out.grad

        out = Value(
            data=data,
            _backward=_backward,
            _children=(self,),
            _op="tanh",
        )
        return out

    def __repr__(self) -> str:
        return f"Value({self.data})"

    def __add__(self, other: Value | int | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)

        def _backward():
            self.grad = 1.0 * out.grad
            other.grad = 1.0 * out.grad

        out = Value(
            data=(self.data + other.data),
            _backward=_backward,
            _children=(self, other),
            _op='+',
        )
        return out

    def __mul__(self, other: Value | int | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)

        def _backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad

        out = Value(
            data=(self.data * other.data),
            _backward = _backward,
            _children=(self, other),
            _op='*',
        )
        return out

    def __rmul__(self, other: Value | int | float) -> Value:  # other * self
        return self * other
