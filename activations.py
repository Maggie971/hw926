import numpy as np

class ActivationFunction:
    def forward(self, z):
        raise NotImplementedError

    def backward(self, dz, z):
        raise NotImplementedError

class Linear(ActivationFunction):
    def forward(self, z):
        return z

    def backward(self, dz, z):
        return dz

class ReLU:
    def forward(self, z):
        return np.maximum(0, z)

    def backward(self, dz, z):
        dz = np.array(dz, copy=True)  # 确保不修改输入的dz
        dz[z <= 0] = 0
        return dz


class Softmax:
    def forward(self, z):
        # 数值稳定的 softmax
        z_shifted = z - np.max(z, axis=1, keepdims=True)  # 防止溢出
        exps = np.exp(z_shifted)
        return exps / np.sum(exps, axis=1, keepdims=True)


class Sigmoid:
    def forward(self, z):
        return 1 / (1 + np.exp(-z))

    def backward(self, dz, z):
        sig = self.forward(z)
        return dz * sig * (1 - sig)


class Tanh:
    def forward(self, z):
        return np.tanh(z)

    def backward(self, dz, z):
        return dz * (1 - np.tanh(z) ** 2)
