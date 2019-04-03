# 1.1.1 vector and matrix
import numpy as np

x = np.array([1, 2, 3])
x.__class__ # 클래스 이름 표시
# <class 'numpy.ndarray'>

x.shape
# (3,)

x.ndim
# 1

W = np.array([[1, 2, 3], [4, 5, 6]])
W.shape
# (2, 3)

W.ndim
# 2

# 1.1.2 element-wise of matrix
W = np.array([[1, 2, 3], [4, 5, 6]])
X = np.array([[0, 1, 2], [3, 4, 5]])
W + X
# array([[ 1,  3,  5],
#        [ 7,  9, 11]])
W * X
# array([[ 0,  2,  6],
#        [12, 20, 30]])

# 1.1.3 broadcast
A = np.array([[1, 2], [3, 4]])
A * 10
# array([[10, 20],
#        [30, 40]])

A = np.array([[1, 2], [3, 4]])
b = np.array([10, 20])
A * b
# array([[10, 40],
#        [30, 80]])

# 1.1.4
# 벡터의 내적
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
np.dot(a, b)
# 32

# 행렬의 곱
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
np.dot(A, B)
# array([[19, 22],
#        [43, 50]])

# 1.1.5

# 1.2

# 1.2.1
import numpy as np

W1 = np.random.randn(2, 4) # 가중치
b1 = np.random.randn(4) # 편향
x = np.random.randn(10, 2) # 입력
h = np.dot(x, W1) + b1

# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

a = sigmoid(h)

# summary
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.random.randn(10, 2)
W1 = np.random.randn(2, 4)
b1 = np.random.randn(4)
W2 = np.random.randn(4, 3)
b2 = np.random.randn(3)

h = np.dot(x, W1) + b1
a = sigmoid(h)
s = np.dot(a, W2) + b2

# 1.2.2

# coding: utf-8
import numpy as np


class Sigmoid:
    def __init__(self):
        self.params = []

    def forward(self, x):
        return 1 / (1 + np.exp(-x))


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        return out


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # 가중치와 편향 초기화
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        # 계층 생성
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        # 모든 가중치를 리스트에 모은다.
        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x


x = np.random.randn(10, 2)
model = TwoLayerNet(2, 4, 3)
s = model.predict(x)
print(s)

# 1.3

# 1.3.1 Loss Function

# 1.3.2 Differential and Gradient

# 1.3.3 Chain Rule

# 1.3.4

# Repeat Node
import numpy as np

D, N = 8, 7
x = np.random.randn(1, D) # 입력
y = np.repeat(x, N, axis=0) # 순전파

dy = np.random.randn(N, D) # 무작위 기울기
dx = np.sum(dy, axis=0, keepdims=True) # 역전파

# Sum Node
import numpy as np

D, N = 8, 7
x = np.random.randn(N, D) # 입력
y = np.sum(x, axis=0, keepdims=True) # 순전파

dy = np.random.randn(1, D) # 무작위 기울기
dx = np.repeat(dy, N, axis=0) # 역전파

# MatMul
class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None
    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out
    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx

# 1.3.5