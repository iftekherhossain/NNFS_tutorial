import numpy as np 
import nnfs
from nnfs.datasets import spiral_data
from matplotlib import pyplot as plt

nnfs.init()
class Layer_Dense:
    def __init__(self,n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1,n_neurons))
    
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2,3) # 2 for 2 input feature and 3 for 3 class
dense1.forward(X)
print(dense1.output[:5])
# a = np.random.randn(2,3)
# print("input_shape",X.shape)
# # plt.scatter(X[:,0],X[:,1],c=y,cmap='brg')
# # plt.show()

# #print(a)
# weights = np.random.randn(2,3)
# print("weight_shape",weights.shape)

# res = np.dot(X,weights)
# print("result",res)
# print("res_shape",res.shape)