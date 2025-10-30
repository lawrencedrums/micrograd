class Value:
    def __init__(self, value: float) -> None:
        self.value = value

    def __add__(self, other: Value) -> Value:
        if not isinstance(other, Value):
            raise ValueError("Input type invalid")
        return Value(self.value + other.value)

    def __mul__(self, other: Value) -> Value:
        if not isinstance(other, Value):
            raise ValueError("Input type invalid")
        return Value(self.value - other.value)
