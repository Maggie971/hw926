import numpy as np

class DeepNeuralNetwork:
    def __init__(self, layer_sizes, activations):
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.params = self._initialize_weights()

    # 初始化权重
    def _initialize_weights(self):
        """
        Initialize weights and biases for each layer. He initialization is used for weights.
        """
        params = {}
        L = len(self.layer_sizes) - 1  # Number of layers excluding input

        # Initialize weights and biases for each layer
        for l in range(1, L + 1):
            params[f"W{l}"] = np.random.randn(self.layer_sizes[l], self.layer_sizes[l-1]) * np.sqrt(2 / self.layer_sizes[l-1])
            params[f"b{l}"] = np.zeros((self.layer_sizes[l], 1))

        return params

    # 前向传播
    def forward(self, X):
        cache = {"A0": X}
        A = X
        L = len(self.layer_sizes) - 1  # 层数

        for l in range(1, L + 1):
            Z = np.dot(self.params[f"W{l}"], A) + self.params[f"b{l}"]
            A = self.activations[l - 1].forward(Z)
            cache[f"Z{l}"] = Z
            cache[f"A{l}"] = A

        return A, cache  # 返回最后的输出和中间缓存

    def backward(self, Y, cache):
        grads = {}
        L = len(self.layer_sizes) - 1
        m = Y.shape[1]
        AL = cache[f"A{L}"]

        # 输出层的梯度
        dZL = AL - Y
        grads[f"dW{L}"] = np.dot(dZL, cache[f"A{L-1}"].T) / m
        grads[f"db{L}"] = np.sum(dZL, axis=1, keepdims=True) / m

        # 反向传播到前一层
        for l in reversed(range(1, L)):
            dA = np.dot(self.params[f"W{l+1}"].T, dZL)
            dZ = self.activations[l-1].backward(dA, cache[f"Z{l}"])

            # 梯度裁剪：限制梯度的最大值，防止梯度爆炸
            dZ = np.clip(dZ, -0.5, 0.5)

            grads[f"dW{l}"] = np.dot(dZ, cache[f"A{l-1}"].T) / m
            grads[f"db{l}"] = np.sum(dZ, axis=1, keepdims=True) / m
            dZL = dZ

        return grads

    # 参数更新
    def update_params(self, grads, learning_rate):
        """
        使用梯度下降法更新参数。
        grads: 反向传播计算出的梯度
        learning_rate: 学习率
        """
        L = len(self.layer_sizes) - 1  # 总层数
        for l in range(1, L + 1):
            self.params[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
            self.params[f"b{l}"] -= learning_rate * grads[f"db{l}"]

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        L2_reg = 0
        lambd = 0.05  # 正则化系数
        for l in range(1, len(self.layer_sizes)):
            L2_reg += np.sum(np.square(self.params[f"W{l}"]))  # L2 正则化
        cost = -np.sum(Y * np.log(AL + 1e-8)) / m + (lambd / (2 * m)) * L2_reg
        return cost

