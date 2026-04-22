import numpy as np


def _unbroadcast(grad, shape):
    """Sum gradient along axes that were broadcast during forward pass."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, (g, s) in enumerate(zip(grad.shape, shape)):
        if s == 1 and g != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        if isinstance(data, np.ndarray):
            self.data = data if data.dtype == np.float32 else data.astype(np.float32)
        else:
            self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._children = set(_children)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def T(self):
        return self.transpose()

    def __repr__(self):
        return f"Tensor(shape={self.shape}, grad={self.grad is not None})"

    def __len__(self):
        return len(self.data)

    # ---- Arithmetic ----

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other), _op='+',
        )

        def _backward():
            if self.requires_grad:
                g = _unbroadcast(out.grad, self.shape)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = _unbroadcast(out.grad, other.shape)
                other.grad = g if other.grad is None else other.grad + g

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other), _op='*',
        )

        def _backward():
            if self.requires_grad:
                g = _unbroadcast(out.grad * other.data, self.shape)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = _unbroadcast(out.grad * self.data, other.shape)
                other.grad = g if other.grad is None else other.grad + g

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data / other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other), _op='/',
        )

        def _backward():
            if self.requires_grad:
                g = _unbroadcast(out.grad / other.data, self.shape)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = _unbroadcast(-out.grad * self.data / (other.data ** 2), other.shape)
                other.grad = g if other.grad is None else other.grad + g

        out._backward = _backward
        return out

    def __rtruediv__(self, other):
        return Tensor(other) / self

    def __pow__(self, exp):
        assert isinstance(exp, (int, float))
        out = Tensor(
            self.data ** exp,
            requires_grad=self.requires_grad,
            _children=(self,), _op=f'**{exp}',
        )

        def _backward():
            if self.requires_grad:
                g = out.grad * exp * self.data ** (exp - 1)
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    # ---- Matrix ops ----

    def matmul(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other), _op='matmul',
        )

        def _backward():
            if self.requires_grad:
                g = out.grad @ np.swapaxes(other.data, -1, -2)
                self.grad = g if self.grad is None else self.grad + g
            if other.requires_grad:
                g = np.swapaxes(self.data, -1, -2) @ out.grad
                self.grad_other = g
                other.grad = g if other.grad is None else other.grad + g

        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def transpose(self, *axes):
        if not axes:
            out_data = self.data.T
            inv_axes = None
        else:
            out_data = np.transpose(self.data, axes)
            inv_axes = [0] * len(axes)
            for i, a in enumerate(axes):
                inv_axes[a] = i
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op='T')

        def _backward():
            if self.requires_grad:
                g = out.grad.T if inv_axes is None else np.transpose(out.grad, inv_axes)
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    # ---- Activations ----

    def relu(self):
        out = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad, _children=(self,), _op='relu')

        def _backward():
            if self.requires_grad:
                g = out.grad * (self.data > 0).astype(np.float32)
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    def silu(self):
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -500, 500)))
        out = Tensor(self.data * s, requires_grad=self.requires_grad, _children=(self,), _op='silu')

        def _backward():
            if self.requires_grad:
                g = out.grad * (s + self.data * s * (1 - s))
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -500, 500)))
        out = Tensor(s, requires_grad=self.requires_grad, _children=(self,), _op='sigmoid')

        def _backward():
            if self.requires_grad:
                g = out.grad * s * (1 - s)
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, requires_grad=self.requires_grad, _children=(self,), _op='tanh')

        def _backward():
            if self.requires_grad:
                g = out.grad * (1 - t ** 2)
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    # ---- Reductions ----

    def sum(self, axis=None, keepdims=False):
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad, _children=(self,), _op='sum',
        )

        def _backward():
            if self.requires_grad:
                g = out.grad
                if axis is not None and not keepdims:
                    g = np.expand_dims(g, axis=axis)
                g = np.broadcast_to(g, self.shape).copy()
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        n = np.prod([self.shape[a] for a in (range(self.ndim) if axis is None else (axis if isinstance(axis, tuple) else (axis,)))])
        out = Tensor(
            self.data.mean(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad, _children=(self,), _op='mean',
        )

        def _backward():
            if self.requires_grad:
                g = out.grad
                if axis is not None and not keepdims:
                    g = np.expand_dims(g, axis=axis)
                g = np.broadcast_to(g, self.shape).copy() / n
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    def max(self, axis=None, keepdims=False):
        out_data = self.data.max(axis=axis, keepdims=keepdims)
        out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op='max')

        def _backward():
            if self.requires_grad:
                max_vals = self.data.max(axis=axis, keepdims=True)
                mask = (self.data == max_vals).astype(np.float32)
                mask /= mask.sum(axis=axis, keepdims=True)
                g = out.grad
                if axis is not None and not keepdims:
                    g = np.expand_dims(g, axis=axis)
                g = np.broadcast_to(g, self.shape).copy() * mask
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    # ---- Element-wise math ----

    def exp(self):
        e = np.exp(np.clip(self.data, -500, 500))
        out = Tensor(e, requires_grad=self.requires_grad, _children=(self,), _op='exp')

        def _backward():
            if self.requires_grad:
                g = out.grad * e
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    def log(self):
        safe = np.clip(self.data, 1e-8, None)
        out = Tensor(np.log(safe), requires_grad=self.requires_grad, _children=(self,), _op='log')

        def _backward():
            if self.requires_grad:
                g = out.grad / safe
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    def abs(self):
        out = Tensor(np.abs(self.data), requires_grad=self.requires_grad, _children=(self,), _op='abs')

        def _backward():
            if self.requires_grad:
                g = out.grad * np.sign(self.data)
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    def clamp(self, min_val=None, max_val=None):
        out = Tensor(
            np.clip(self.data, min_val, max_val),
            requires_grad=self.requires_grad, _children=(self,), _op='clamp',
        )

        def _backward():
            if self.requires_grad:
                mask = np.ones_like(self.data)
                if min_val is not None:
                    mask *= (self.data >= min_val)
                if max_val is not None:
                    mask *= (self.data <= max_val)
                g = out.grad * mask
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    # ---- Shape ops ----

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad, _children=(self,), _op='reshape')

        def _backward():
            if self.requires_grad:
                g = out.grad.reshape(self.shape)
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    def flatten(self, start_dim=0):
        new_shape = self.shape[:start_dim] + (-1,)
        return self.reshape(*new_shape)

    # ---- Probabilistic / DreamerV3-specific ----

    def softmax(self, axis=-1):
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(shifted)
        s = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(s, requires_grad=self.requires_grad, _children=(self,), _op='softmax')

        def _backward():
            if self.requires_grad:
                g = out.grad * s - s * (out.grad * s).sum(axis=axis, keepdims=True)
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    def log_softmax(self, axis=-1):
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        lse = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
        ls = shifted - lse
        out = Tensor(ls, requires_grad=self.requires_grad, _children=(self,), _op='log_softmax')

        def _backward():
            if self.requires_grad:
                s = np.exp(ls)
                g = out.grad - s * out.grad.sum(axis=axis, keepdims=True)
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    def straight_through(self, hard):
        """Forward uses hard values, backward passes gradient to self (the soft logits)."""
        hard_data = hard.data if isinstance(hard, Tensor) else hard
        out = Tensor(hard_data, requires_grad=self.requires_grad, _children=(self,), _op='ST')

        def _backward():
            if self.requires_grad:
                self.grad = out.grad.copy() if self.grad is None else self.grad + out.grad

        out._backward = _backward
        return out

    def symlog(self):
        out = Tensor(
            np.sign(self.data) * np.log1p(np.abs(self.data)),
            requires_grad=self.requires_grad, _children=(self,), _op='symlog',
        )

        def _backward():
            if self.requires_grad:
                g = out.grad / (np.abs(self.data) + 1)
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    def symexp(self):
        out = Tensor(
            np.sign(self.data) * (np.exp(np.abs(self.data)) - 1),
            requires_grad=self.requires_grad, _children=(self,), _op='symexp',
        )

        def _backward():
            if self.requires_grad:
                g = out.grad * np.exp(np.abs(self.data))
                self.grad = g if self.grad is None else self.grad + g

        out._backward = _backward
        return out

    # ---- Gradient control ----

    def detach(self):
        """Returns a copy with no graph connection (stop gradient)."""
        return Tensor(self.data.copy())

    def zero_grad(self):
        self.grad = None

    # ---- Backward ----

    def backward(self):
        topo = []
        visited = set()

        def _build_topo(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._children:
                    _build_topo(child)
                topo.append(v)

        _build_topo(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()


# ---- Free functions ----

def cat(tensors, axis=0):
    data = np.concatenate([t.data for t in tensors], axis=axis)
    out = Tensor(
        data,
        requires_grad=any(t.requires_grad for t in tensors),
        _children=tuple(tensors), _op='cat',
    )

    def _backward():
        sections = np.cumsum([t.shape[axis] for t in tensors[:-1]])
        grads = np.split(out.grad, sections, axis=axis)
        for t, g in zip(tensors, grads):
            if t.requires_grad:
                t.grad = g.copy() if t.grad is None else t.grad + g

    out._backward = _backward
    return out


def one_hot(indices, num_classes):
    """One-hot encode integer indices. No gradient (used for targets/sampling)."""
    flat = indices.flatten().astype(int)
    oh = np.zeros((flat.size, num_classes), dtype=np.float32)
    oh[np.arange(flat.size), flat] = 1.0
    return Tensor(oh.reshape(indices.shape + (num_classes,)))
