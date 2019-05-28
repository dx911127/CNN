import numpy as np
import pandas as pd
from scipy import linalg
from abc import ABCMeta, abstractmethod
import copy
def tanh(x):
    return np.tanh(x)
def tanh_derivative(x):
    return 1.0 - x * x
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
def relu(x):
	return np.maximum(x, 0)
def relu_derivative(x):
	t = copy.copy(x)
	#for i in range(len(t)):
	#	if t[i] <= (1e-12):
	#		t[i] = 0
	#	else:
	#		t[i] = 1
	t[t > 0] = 1
	return t

class ActivationFunc:
	def __init__(self):
		self.tdict = dict()
		self.tdict['tanh'] = np.tanh
		self.tdict['sigmoid'] = lambda x: 1 / (1 + np.exp(-x.clip(-40,40)))
		self.tdict['relu'] = relu
		self.tdict['softmax'] = lambda x: np.exp(x.clip(-40, 40))
		self.ddict = dict()
		self.ddict['tanh'] = tanh_derivative
		self.ddict['sigmoid'] = sigmoid_derivative
		self.ddict['relu'] = relu_derivative
		self.ddict['softmax'] = lambda x: np.exp(x.clip(-40, 40))

	def getActivation(self, activation):
		if activation in self.tdict:
			return self.tdict[activation]
		else:
			return lambda x: x

	def getDActivation(self, activation):
		if activation in self.ddict:
			return self.ddict[activation]
		else:
			return lambda x: np.ones(x.shape)

class BaseLayer:
	def __init__(self):
		self.mintput = None
		self.moutput = None
		self.para = None
		self.bstep = 0
		self.grad = None
		self.activationFac = ActivationFunc()

	@abstractmethod
	def mcompile(self):
		raise NotImplementedError

	@abstractmethod
	def forward(self):
		raise NotImplementedError

	@abstractmethod
	def backward(self):
		raise NotImplementedError

	
	def getlen(self, vec):
		return np.sqrt(np.dot(vec, vec))

	def myschmitt(self, minput):
		for i in range(len(minput)):
			orglen = self.getlen(minput[i])
			for j in range(i):
				minput[i] -= np.dot(minput[j], minput[i])/np.dot(minput[i], minput[i]) * minput[j]
			minput[i] *= (orglen / self.getlen(minput[i]))
		return minput

	def step(self, lr = 0.001, bcnt = 1, maxdiv = 1):
		a = 'doingnothing'


#---- can be splitted into 2 files----

class Conv2D(BaseLayer):
	def __init__(self, activation = 'relu', msize = 3, filters = 1, padding = 'same', strides = 1):
		BaseLayer.__init__(self)
		self.msize = msize
		self.activation = self.activationFac.getActivation(activation)
		self.dactivation = self.activationFac.getDActivation(activation)
		
		self.minput = None
		self.moutput = None
		self.stride = int(strides)
		self.outputshape = None
		self.inputshape = None
		self.padding = padding
		
		self.para = None
		self.grad = None
		self.backout = None
		self.mbias = None
		self.gbias = None
		self.filternum = filters
		self.bstep = 0
		self.outputfunc = False
		self.validshape = None

	def mcompile(self, val = None, inputshape = (1,), isoutput = False):
		self.validshape = inputshape
		if self.padding == 'same':
			self.inputshape = (inputshape[0], (inputshape[1] + self.msize - 1) // self.stride * self.stride, (inputshape[2] + self.msize - 1) // self.stride * self.stride)
		else:
			self.inputshape = self.validshape
		if val == None:
			val = np.sqrt(6 / (self.msize * self.msize))
		self.para = 2 * val * (np.random.rand(self.filternum, inputshape[0] * self.msize * self.msize) - 0.5)
		if self.para.shape[0] <= self.para.shape[1]:
			self.para = self.myschmitt(self.para).reshape(self.filternum, inputshape[0], self.msize, self.msize)
		else:
			self.para = self.para.reshape(self.filternum, inputshape[0], self.msize, self.msize)
		#self.para *= val
		self.grad = np.zeros((self.filternum, inputshape[0], self.msize, self.msize))
		self.mbias = (2 * val * (np.random.rand(self.filternum) - 0.5))
		self.gbias = np.zeros(self.filternum)
		self.minput = np.zeros(self.inputshape)
	
		self.outputshape = (self.filternum, (self.inputshape[1] - self.msize)//self.stride + 1, (self.inputshape[2] - self.msize)//self.stride + 1)
		self.moutput = np.zeros(self.outputshape)
		self.backout = np.zeros(self.inputshape)
		return self.outputshape

	def forward(self, minput):
		self.minput[:, :minput.shape[1], :minput.shape[2]] = minput
		for i in range(self.filternum):
			for j in range(0, (self.inputshape[1] - self.msize)//self.stride + 1):
				xstart = j * self.stride
				for k in range(0, (self.inputshape[2] - self.msize)//self.stride + 1):
					ystart = k * self.stride
					self.moutput[i][j][k] = self.mbias[i]
					for i1 in range(0, self.inputshape[0]):
						self.moutput[i][j][k] += np.multiply(self.minput[i1][xstart:xstart + self.msize, ystart:ystart + self.msize], self.para[i][i1]).sum()
		self.moutput = self.activation(self.moutput)
		return self.moutput

	def backward(self, mloss):
		#print(self.moutput.shape, mloss.shape)
		xloss = self.dactivation(self.moutput) * mloss
		hend = ((self.inputshape[1] - self.msize)//self.stride + 1) * self.stride
		vend = ((self.inputshape[2] - self.msize)//self.stride + 1) * self.stride

		self.backout.fill(0)
		# without consider mloss[i] is full of zero
		'''
		oshape1 = self.outputshape[1]
		oshape2 = self.outputshape[2]
 		for i in range(self.filternum):
			for i1 in range(self.inputshape[0]):
				for j in range(self.msize):
					for k in range(self.msize):
						self.grad[i][i1][j][k] += np.multiply(xloss[i], self.minput[i1][j: j+oshape1: self.stride, k: k+oshape2: self.stride]).sum()
						self.backout[i1][j: j+oshape1: self.stride, k: k+oshape2: self.stride] += xloss[i] * self.para[i][i1][j][k] 
		'''
		for i in range(self.filternum):
			for j in range(self.outputshape[1]):
				for k in range(self.outputshape[2]):
					if np.abs(xloss[i][j][k]) > 1e-12:
						xstart = j * self.stride
						ystart = k * self.stride
						tloss = xloss[i][j][k]
						self.gbias[i] += tloss
						for i1 in range(self.inputshape[0]):
							self.grad[i][i1] += tloss * self.minput[i1][xstart: xstart + self.msize, ystart: ystart + self.msize]
							self.backout[i1][xstart:xstart + self.msize, ystart: ystart + self.msize] += tloss * self.para[i][i1]
		
		return copy.copy(self.backout[:, :self.validshape[1], :self.validshape[2]])

		def step(self, lr = 0.001, bcnt = 1, maxdiv = 1):
			self.para -= lr * bcnt * self.grad / maxdiv
			self.mbias -= lr * bcnt * self.gbias / maxdiv
			self.grad.fill(0)
			self.gbias.fill(0)

class Dense(BaseLayer):
	def __init__(self, activation = 'relu', msize = 1):
		BaseLayer.__init__(self)
		self.outputsize = msize
		self.activation = self.activationFac.getActivation(activation)
		self.dactivation = self.activationFac.getDActivation(activation)
		self.activationname = activation

		self.para = None
		self.grad = None
		self.minput = None
		self.moutput = None
		self.mbias = None
		self.gbias = None
		self.isoutput = False

	def mcompile(self, val = None, inputshape = (1,), isoutput = False):
		self.inputsize = inputshape[0]
		if val == None:
			val = np.sqrt(6/(self.inputsize + self.outputsize))
		self.para = 2 * val * (np.random.rand(self.inputsize, self.outputsize)- 0.5)
		if self.para.shape[0] <= self.para.shape[1]:
			self.para = self.myschmitt(self.para)
		#self.para *= val
		self.grad = np.zeros((self.inputsize, self.outputsize))
		self.mbias = 2 * val * (np.random.rand(self.outputsize) - 0.5)
		self.gbias = np.zeros(self.outputsize)
		self.moutput = np.zeros(self.outputsize)
		self.isoutput = isoutput
		return self.moutput.shape

	def forward(self, minput):
		self.minput = np.atleast_2d(minput)
		#print(self.minput.shape,self.para.shape,self.mbias.shape)
		self.moutput = self.activation(np.matmul(self.minput, self.para) + self.mbias) 
		return self.moutput
		
	def backward(self, mloss):
		if not self.isoutput:
			tloss = mloss * self.dactivation(self.moutput)
		else:
			tloss = mloss
		self.grad += np.matmul(self.minput.T, tloss)
		self.backout = np.matmul(tloss, self.para.T)
		self.gbias += np.squeeze(tloss, axis = 0)
		return self.backout

	def step(self, lr = 0.001, bcnt = 1, maxdiv = 1):
		self.para -= lr * bcnt * self.grad / maxdiv
		self.mbias -= lr * bcnt * self.gbias / maxdiv
		self.grad.fill(0)
		self.gbias.fill(0)

class MaxPooling(BaseLayer):
	def __init__(self, msize = 2):
		BaseLayer.__init__(self)
		self.msize = int(msize)
		self.minput = None
		self.moutput = None
		self.backout = None
		self.maxid = None

	def mcompile(self, inputshape = None, isoutput = False):
		self.inputsize = inputshape
		tmp = (self.inputsize[1] + self.msize - 1) // self.msize
		self.outputsize = (self.inputsize[0], tmp, tmp) 
		self.moutput = np.zeros(self.outputsize)
		self.maxid = np.zeros(self.outputsize)
		self.backout = np.zeros(self.inputsize)
		return self.outputsize

	def forward(self, minput):
		self.minput = minput
		self.moutput.fill(0)
		for i in range(self.outputsize[0]): 
			for j in range(self.outputsize[1]):
				for k in range(self.outputsize[2]):
					tmax = 0
					tmaxid = 0
					for j1 in range(self.msize):
						if j * self.msize + j1 == self.inputsize[1]:
							break
						for k1 in range(self.msize):
							if k * self.msize + k1 == self.inputsize[2]:
								break
							if self.minput[i][j * self.msize + j1][k * self.msize + k1] > tmax:
								tmax = self.minput[i][j * self.msize + j1][k * self.msize + k1]
								tmaxid = j1 * self.msize + k1
					self.maxid[i][j][k] = tmaxid 
					self.moutput[i][j][k] = tmax
		return self.moutput
	
	def backward(self, mloss):
		self.backout.fill(0)
		for i in range(self.outputsize[0]):
			for j in range(self.outputsize[1]):
				for k in range(self.outputsize[2]):
					tloss = mloss[i][j][k]
					if np.abs(tloss) > 1e-12:
						xid = int(self.maxid[i][j][k]) // self.msize
						yid = int(self.maxid[i][j][k]) % self.msize
						#print(j * self.msize + xid, k * self.msize + yid)
						self.backout[i][j * self.msize + xid][k * self.msize + yid] = tloss
		return self.backout


class GlobalAveragePooling(BaseLayer):
	def	__init__(self):
		BaseLayer.__init__(self)
		self.minput = None
		self.moutput = None
		self.backout = None
		self.inputsize = None
		self.outputsize = None

	def mcompile(self, inputshape = None, isoutput = False):
		self.inputsize = inputshape
		self.outputsize = (self.inputsize[0], )
		self.moutput = np.zeros(self.outputsize)
		self.backout = np.zeros(self.inputsize)
		return self.outputsize

	def forward(self, minput):
		self.minput = minput
		for i in range(self.outputsize[0]):
			self.moutput[i] = minput[i].mean()
		return self.moutput
	
	def backward(self, mloss):
		for i in range(self.outputsize[0]):
			self.backout[i] = self.minput[i] * mloss[0][i] / (self.inputsize[1] * self.inputsize[2])
		return self.backout

class Flatten(BaseLayer):
	def	__init__(self):
		BaseLayer.__init__(self)
		self.minput = None
		self.moutput = None
		self.backout = None
		self.inputsize = None
		self.outputsize = None

	def mcompile(self, inputshape = None, isoutput = False):
		self.inputsize = inputshape
		self.outputsize = (self.inputsize[0]*self.inputsize[1] * self.inputsize[2], )
		self.moutput = np.zeros(self.outputsize)
		self.backout = np.zeros(self.inputsize)
		return self.outputsize

	def forward(self, minput):
		self.minput = minput
		#for i in range(self.outputsize[0]):
		self.moutput = minput.flatten()
		return self.moutput
	
	def backward(self, mloss):
		self.backout = mloss.reshape(self.inputsize)
		return self.backout

class CNetwork:
	def __init__(self, inputsize):
		self.layerlist = []
		self.moutput = None
		self.inputsize = inputsize
		self.outputfunc = None
		self.bstep = 0
		self.lr = 0.001
	
	def mcompile(self, lr = 0.001):
		nowinputshape = self.inputsize
		#print(nowinputshape)
		for layer in self.layerlist:
			flag = layer is self.layerlist[-1]
			nowinputshape = layer.mcompile(inputshape = nowinputshape, isoutput = flag)
			#print(nowinputshape)
		self.outputfunc = self.layerlist[-1].activationname
		self.lr = lr
		
	def add(self, nowlayer):
		self.layerlist.append(nowlayer)
		

	def forward(self, minput):
		for eachlayer in self.layerlist:
			minput = eachlayer.forward(minput)
		return copy.copy(minput)
			
	def backward(self, y, y_label):
		self.maxnum = 0.001
		self.bstep += 1
		loss = copy.copy(y)
		if self.outputfunc == 'softmax':
			tsumy = sum(y)
			loss[y_label] -= tsumy
			loss /= max(tsumy, 1e-4)
		elif self.outputfunc == 'sigmoid':
			if y_label == 1:
				loss -= 1
		loss = np.atleast_2d(loss)
		#print(loss)
		for layer in reversed(self.layerlist):
			loss = layer.backward(loss)


	def step(self):
		mdiv = 0
		for layer in self.layerlist:
			if layer.grad is not None:
				mdiv = max(mdiv, np.abs(layer.grad).max())
				mdiv = max(mdiv, layer.gbias.max())
		for layer in self.layerlist:
			layer.step(lr = self.lr, bcnt = self.bstep, maxdiv = max(mdiv // 1000, 1))
		self.bstep = 0

	def predict(self, minput):
		predictions = self.forward(minput)
		res = np.argmax(predictions[0])
		return res

if __name__ == "__main__":
	
	#model = CNetwork(inputsize = (784,))
	model = CNetwork(inputsize = (1, 28, 28))
	model.add(Conv2D(filters = 6, msize = 5))
	model.add(MaxPooling(msize = 2))
	model.add(Conv2D(filters = 16, msize = 5))
	model.add(MaxPooling(msize = 2))
	#model.add(GlobalAveragePooling())
	model.add(Flatten())
	#model.add(Dense(msize = 256))
	model.add(Dense(msize = 64))
	model.add(Dense(msize = 10, activation = 'softmax'))
	model.mcompile(lr = 0.0001)
	x_train = np.load('mnist/x_train.npy') / 255
	y_train = np.load('mnist/y_train.npy')
	x_test = np.load('mnist/x_test.npy') / 255
	y_test = np.load('mnist/y_test.npy')
	
	
	#print(x_train.shape)
	#print(x_test.shape)
	'''
	epochs = 4
	for e in range(epochs):
		for i in range(len(x_train)):
			if y_train[i] > 1:
				continue
			moutput = model.forward(x_train[i].reshape(784, ))
			#print(moutput, y_train[i])
			model.backward(moutput, y_train[i])
			if i % 10 == 9:
				model.step()
		tcnt = 0
		tot = 0
		for i in range(len(x_test)):
			if y_test[i] < 2:
				tot += 1
				tmp = model.forward(x_test[i].reshape(784,))
				if int(tmp > 0.5) == y_test[i]:
					tcnt += 1
		print('epoch {},Accuracy {}%'.format(e+1,tcnt / tot * 100))
	'''
	epochs = 1
	for e in range(epochs):
		tot = 0
		for i in range(len(x_train)):
		#	if y_train[i] >= 5:
		#		continue
			moutput = model.forward(np.expand_dims(x_train[i], axis = 0))
			print("case {}:".format(tot + 1), moutput, y_train[i])

			model.backward(np.squeeze(moutput, axis = 0), y_train[i])
			#if i > len(x_train)//2 - 20:
				#for i in range(2):
				#	print(model.layerlist[i].backout)
				#print()
			#if i % 10 == 9:
			if tot % 5 == 4:
				model.step()
			tot += 1
		tcnt = 0
		tot = 0
		val_loss = 0
		for i in range(len(x_test)):
			#if y_test[i] < 5:
			tot += 1
			tmp = model.forward(np.expand_dims(x_test[i],axis = 0))
			tx = np.argmax(tmp[0])
			val_loss += min(-np.log(tmp[0][y_test[i]]/tmp[0].sum()), 100)
			if tx == y_test[i]:
				tcnt += 1
		val_loss /= tot
		print('epoch {},Accuracy {}%,val_loss {}'.format(e + 1, tcnt / tot * 100, val_loss))
	

	'''
	model = CNetwork(inputsize = (2,))
	model.add(Dense(msize = 16))
	model.add(Dense(msize = 8))
	model.add(Dense(msize = 1,activation = 'sigmoid'))
	model.mcompile(lr = 0.001)
	print(model.outputfunc)
	X = np.array([[0,0],[0,1],[1,0],[1,1]])
	y = np.array([0, 1, 1, 0])
	for i in range(10000):
		for j in range(4):
			moutput = model.forward(np.expand_dims(X[j], axis = 0))
			#print(moutput)
			model.backward(np.squeeze(moutput, axis = 0), y[j])
		model.step()
	for j in range(4):
		print(model.forward(X[j]))
	'''
	
	

