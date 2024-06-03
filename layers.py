from abc import abstractmethod
from torch import nn
from torch.nn.utils import weight_norm
from utils import initialize_nn_weights


class _PcDEQLinearLayer(nn.Module):
    def __init__(self, out_features, act):
        super().__init__()
        self.W = nn.Linear(out_features, out_features, bias=False)
        self.W.weight = nn.Parameter(initialize_nn_weights(self.W.weight.data, 1e-4))
        self.W = weight_norm(self.W)
        self.activation = self._get_activation(act)

    def forward(self, z, x):
        z = self.activation(self.W(z) + x)
        return z

    @abstractmethod
    def _get_activation(self, act):
        pass


class PcDEQ1LinearLayer(_PcDEQLinearLayer):
    def _get_activation(self, act):
        match act:
            case "tanh":
                return nn.Tanh()
            case "softsign":
                return nn.Softsign()
            case "relu6":
                return nn.ReLU6()
            case _:
                raise NotImplementedError(
                    f"Activation function '{act}' currently is not supported"
                )


class PcDEQ2LinearLayer(_PcDEQLinearLayer):
    def _get_activation(self, act):
        if act == "sigmoid":
            return nn.Sigmoid()

        raise NotImplementedError(
            f"Activation function '{act}' currently is not supported"
        )


class _PcDEQConvLayer(nn.Module):
    def __init__(self, out_features, act):
        super().__init__()
        self.W = nn.Conv2d(out_features, out_features, 3, padding=1, bias=False)
        self.W.weight = nn.Parameter(initialize_nn_weights(self.W.weight.data, 1e-4))
        self.W = weight_norm(self.W)
        self.activation = self._get_activation(act)

    def forward(self, z, x):
        z = self.activation(self.W(z) + x)
        return z

    @abstractmethod
    def _get_activation(self, act):
        pass


class PcDEQ1ConvLayer(_PcDEQConvLayer):
    def _get_activation(self, act):
        match act:
            case "tanh":
                return nn.Tanh()
            case "softsign":
                return nn.Softsign()
            case "relu6":
                return nn.ReLU6()
            case _:
                raise NotImplementedError(
                    f"Activation function '{act}' currently is not supported"
                )


class PcDEQ2ConvLayer(_PcDEQConvLayer):
    def _get_activation(self, act):
        if act == "sigmoid":
            return nn.Sigmoid()

        raise NotImplementedError(
            f"Activation function '{act}' currently is not supported"
        )
