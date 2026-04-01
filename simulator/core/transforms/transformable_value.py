import numpy as np



class ValueView:
    def __init__(self, parent, to_internal=lambda x: x, from_internal=lambda x: x):
        self._parent = parent
        self._to_internal = to_internal
        self._from_internal = from_internal

    def set(self, value):
        self._parent._data = self._to_internal(value)

    def get(self):
        return self._from_internal(self._parent._data.copy())


class TransformableValue:
    def __init__(self, shape, scale=1.0):
        self._shape = shape
        self._data = np.zeros(shape)
        self._scale = scale

        self._internal_view = ValueView(self)
        self._real_view = ValueView(
            self,
            to_internal=lambda x: x / self._scale,
            from_internal=lambda x: x * self._scale
        )

    @property
    def internal(self):
        return self._internal_view
    
    @property
    def real(self):
        return self._real_view
        


