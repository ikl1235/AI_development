#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""network.py"""

import random
import numpy as np
import copy


class Network(object):

    #example sizes = [2,1,2,3] > 4계층, 각 레이어마다 2개, 1개, 2개, 3개의 노드
    #np.random.randn(x,y) : 평균0, 표준편차1 랜덤 x*y차원의 배열 생성
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

        
    #노드에서 노드로 옮겨가면서 이전 노드의 출력을 다음 노드의 입력으로 변환
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    
    #eta = 학습 률
    #100개 데이터 배치사이즈가 20이면 epoch=5
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        test_data1 = copy.deepcopy(test_data)
        list_test = list(test_data)
        list_training = list(training_data)
        if list_test : n_test = len(list_test)
        n = len(list_training)
        for j in range(epochs):
            random.shuffle(list_training)
            mini_batches = [ list_training[k:k+mini_batch_size] for k in range(0, n, mini_batch_size) ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if list_test :
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(copy.deepcopy(test_data1)), n_test))
            else:
                print("Epoch {0} complete".format(j))

                
    #가중치와 바이어스 업데이트 언제??한번 그라디언트 계산할때마다??
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

        
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) *             sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))




