import numpy as np
from typing import TypeVar, Generic, Type

class BufferView:
    def __init__(self, parent, to_internal=lambda x: x, from_internal=lambda x: x):
        self._parent = parent
        self._to_internal = to_internal
        self._from_internal = from_internal

    def push(self, value):
        value = self._to_internal(value)
        self._parent._insert(value)

    def set(self, value):
        value = self._to_internal(value)
        self._parent._buffer = value

    def get(self):
        return self._from_internal(self._parent._buffer)
    
    def get_index(self, n=0):
        return self._from_internal(self._parent._buffer[n])
    
    def with_transform(self, to_internal, from_internal):
        return BufferView(
            self._parent,
            to_internal=lambda x: self._to_internal(to_internal(x)),
            from_internal=lambda x: from_internal(self._from_internal(x))
        )
    

V = TypeVar("V", bound="BufferView")

class TransformableBuffer(Generic[V]):
    def __init__(self, size, scale=1.0, view_cls: Type[V] = BufferView):
        self._size = size
        self._scale = scale
        self._buffer = np.zeros(self._size)


        self._internal_view = view_cls(self)
        self._real_view = view_cls(
            self,
            to_internal=lambda x: x / self._scale,
            from_internal=lambda x: x * self._scale
        )

    def size(self):
        return self._size

    def _insert(self, value):
        self._buffer = np.roll(self._buffer, -1, axis=0)
        self._buffer[-1] = value

    def clear(self):
        self._buffer = np.zeros(self._size)

    @property
    def internal(self):
        return self._internal_view
    
    @property
    def real(self):
        return self._real_view
    