#layer.py
import numpy as np
import util
import sys

class Sequential:
    def __init__(self, layers = []):
        self.layers = layers
    def addlayer(self, layer):
        self.layers.append(layer)
    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def backward(self, dout):
        for l in reversed(self.layers):
            dout = l.backward(dout)
        return dout

    def update(self):
        for l in self.layers:
            l.update()

    def zerograd(self):
        for l in self.layers:
            l.zerograd()

class Classifier:
    def __init__(self, model):
        self.model = model

    def predict(self, x, t):
        y = self.model.forward(x)
        #pred = np.argmax(y, axis=1) #pred represents the prediction of the number for each image 
        #acc = 1.0 * np.where(pred == t)[0].size / y.shape[0]
        loss = util.special_error(y,t)
        #return loss, acc
        return loss

    def update(self, x, t):
        self.model.zerograd()
        y = self.model.forward(x)
        #pred = np.argmax(y, axis=1)
        #acc = 1.0 * np.where(pred == 0)[0].size / y.shape[0]
        #acc = 1.0 * np.where(pred == t)[0].size / y.shape[0] #???
        #prob = util.softmax(y)#change output to probability (normalization)
        #loss = util.cross_entropy(prob, t)
        loss = util.special_error(y,t)
        #dout = prob
        dout = y-t
        #dout = prob - t
        #dout[np.arange(dout.shape[0]), t] -= 1 #???
        self.model.backward(dout / dout.shape[0])#calculate partial differentiations by each parameters of each layer to use in next update() function to update parameters. 
        self.model.update()#update parameters based on the partial differntials
        #return loss, acc
        return loss

class Layer(object):
    def __init__(self, lr=0.001, momentum=0.9, weight_decay_rate=5e-4):
        self.params = {}
        self.grads = {} #partial differntials of current time
        self.v = None #the velue which reflects not only effect of current time partial differentials but also effect of past partial differentials
        self.momentum = momentum #the coefficient of inertia term
        self.lr = lr #learning rate
        self.weight_decay_rate = weight_decay_rate #something like attenuation rate for the nurm of each parameter

    def update(self):
        if self.v == None:
            self.v = {}
            for k in self.params.keys():
                self.v[k] = np.zeros(shape = self.params[k].shape, dtype = self.params[k].dtype)
        for k in self.params.keys():
            self.v[k] = self.v[k] * self.momentum - self.lr * self.grads[k]
            self.params[k] = (1 - self.lr * self.weight_decay_rate) * self.params[k] + self.v[k]

    def zerograd(self):
        for k in self.params.keys():
            self.grads[k] = np.zeros(shape = self.params[k].shape, dtype = self.params[k].dtype)


class LinearLayer(Layer):
    def __init__(self, input_dim, output_dim):
        super(LinearLayer, self).__init__()
        self.params['W'] = np.random.normal(scale=np.sqrt(1.0/input_dim), size=(input_dim, output_dim)).astype(np.float32) #use normal distribution for initializing 'W' 
        self.params['b'] = np.zeros(shape = (1, output_dim), dtype=np.float32)

    def forward(self, x):
        self.x = x
        #print(self.params['W'])
        return np.dot(x, self.params['W']) + self.params['b']

    def backward(self, dout):
        self.grads['W'] = np.dot(self.x.T,dout)
        self.grads['b'] = np.sum(dout,axis = 0,keepdims=True)
        return np.dot(dout,self.params['W'].T)

class ReLULayer(Layer):
    def __init__(self):
        super(ReLULayer, self).__init__()

    def forward(self, x):
        out = np.maximum(x, 0)
        self.mask = np.sign(out)
        return out

    def backward(self, dout):
        return self.mask * dout
    
class FlattenLayer(Layer):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        self.original_shape = x.shape
        return np.reshape(x, (x.shape[0], x.size // x.shape[0]))

    def backward(self, dout):
        return np.reshape(dout, self.original_shape)


