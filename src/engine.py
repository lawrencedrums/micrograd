class Value:
    def __init__(
        self,
        data: float,
        grad: float = 0.0,
        label: str = "",
        _op: str = "",
        _children: tuple[Value, ...] = (),
    ) -> None:
        self.data = data
        self.grad = grad
        self.label = label
        self._backward = callable
        self._op = _op
        self._children = set(_children)

    def __repr__(self) -> str:
        return f"Value({self.data})"

    def __add__(self, other: Value | int | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return Value(
            data=(self.data + other.data),
            _op='+',
            _children=(self, other),
        )

    def __mul__(self, other: Value | int | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return Value(
            data=(self.data * other.data),
            _op='*',
            _children=(self, other),
        )

    def __rmul__(self, other: Value | int | float) -> Value:  # other * self
        return self * other
