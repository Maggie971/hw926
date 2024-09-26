import numpy as np
from deep_neural_network import DeepNeuralNetwork
from activations import ReLU, Softmax

# 模拟数据
X_train = np.random.randn(784, 100)  # 确保 X_train 是 (num_features, num_samples) 的形状
Y_train = np.eye(10)[np.random.choice(10, 100)].T  # 转置 Y_train 确保形状正确

# 定义神经网络层大小及其激活函数
layer_sizes = [784, 128, 64, 10]  # 输入层 784，两个隐藏层，输出层 10 个类别
activations = [ReLU(), ReLU(), Softmax()]

# 初始化神经网络
nn = DeepNeuralNetwork(layer_sizes, activations)

# 定义训练参数
learning_rate = 0.0005
num_iterations = 1000

# 训练模型
for i in range(num_iterations):
    AL, cache = nn.forward(X_train)
    cost = nn.compute_cost(AL, Y_train)
    grads = nn.backward(Y_train, cache)
    nn.update_params(grads, learning_rate)

    if i % 100 == 0:
        print(f"第 {i} 次迭代，损失值：{cost}")
