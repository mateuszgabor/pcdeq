import torch
import torch.nn.functional as F
from torch import autograd
from torch import nn
from layers import (
    PcDEQ1LinearLayer,
    PcDEQ2LinearLayer,
    PcDEQ1ConvLayer,
    PcDEQ2ConvLayer,
)
from solvers import fixed_point_iteration


class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.iter_forward = 0
        self.iter_backward = 0
        self.kwargs = kwargs
        self.res_forward = 0
        self.res_backward = 0

    def forward(self, x):
        with torch.no_grad():
            z, self.res_forward, self.iter_forward = self.solver(
                lambda z: self.f(z, x), torch.zeros_like(x), **self.kwargs
            )
        z = self.f(z, x)

        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, x)

        def backward_hook(grad):
            g, self.res_backward, self.iter_backward = self.solver(
                lambda y: autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                grad,
                **self.kwargs
            )
            return g

        z.register_hook(backward_hook)
        return z


class LinearPcDEQ1Block(nn.Module):
    def __init__(self, ch, act, **kwargs):
        super().__init__()
        self.deq = DEQFixedPoint(
            PcDEQ1LinearLayer(ch, act), fixed_point_iteration, **kwargs
        )

    def forward(self, x):
        x = F.softplus(x, beta=5)
        x = self.deq(x)
        return x

    def clamp(self):
        self.deq.f.W.weight_v.data.clamp_(min=0)
        self.deq.f.W.weight_g.data.clamp_(min=0)


class LinearPcDEQ2Block(nn.Module):
    def __init__(self, ch, act, **kwargs):
        super().__init__()
        self.deq = DEQFixedPoint(
            PcDEQ2LinearLayer(ch, act), fixed_point_iteration, **kwargs
        )

    def forward(self, x):
        x = F.relu(x)
        x = self.deq(x)
        return x

    def clamp(self):
        self.deq.f.W.weight_v.data.clamp_(min=0)
        self.deq.f.W.weight_g.data.clamp_(min=0)


class ConvPcDEQ1Block(nn.Module):
    def __init__(self, ch, act, **kwargs):
        super().__init__()
        self.deq = DEQFixedPoint(
            PcDEQ1ConvLayer(ch, act), fixed_point_iteration, **kwargs
        )

    def forward(self, x):
        x = F.softplus(x, beta=5)
        x = self.deq(x)
        return x

    def clamp(self):
        self.deq.f.W.weight_v.data.clamp_(min=0)
        self.deq.f.W.weight_g.data.clamp_(min=0)


class ConvPcDEQ2Block(nn.Module):
    def __init__(self, ch, act, **kwargs):
        super().__init__()
        self.deq = DEQFixedPoint(
            PcDEQ2ConvLayer(ch, act), fixed_point_iteration, **kwargs
        )

    def forward(self, x):
        x = F.relu(x)
        x = self.deq(x)
        return x

    def clamp(self):
        self.deq.f.W.weight_v.data.clamp_(min=0)
        self.deq.f.W.weight_g.data.clamp_(min=0)
