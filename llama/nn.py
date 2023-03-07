import os
import numpy as np

elsize = 2 #float16

new_tensor_id = 0

offload_path = os.path.join(os.path.dirname(__file__), 'offload')

dt = np.float16

os.makedirs(offload_path, exist_ok=True)

def prod(x):
	acc = 0
	for i in x:
		acc *= i
	return acc

class Tensor:
	
	def __init__(self, shape):
		global new_tensor_id

		self.is_param = False
		self.shape = shape
		self.tensor_id = new_tensor_id
		new_tensor_id += 1

		self.offloaded = False


	def size(self):
		return elsize * prod(self.shape)

	def get_offload_path(self):
		return os.path.join(offload_path, '{}.tensor'.format(self.new_tensor_id))

	def load(self):
		if self.offloaded:
			x = open(self.get_offload_path(), 'rb').read()
			self.data = np.fromfile(self.get_offload_path(), dtype=dt)
			self.offloaded = False

	def offload(self):
		if not self.offloaded:
			self.data.tofile(self.get_offload_path())
			del self.data
			self.offloaded = True


	def set_data(self, data):
		self.load()
		self.data = data


	def pow(n):
		pass

	def mean():
		pass

	def __add__(x):
		pass

	def __mul__(x):
		pass

	# do we need this??
	#
	# this commentary explains need for this:
	#
	# view_as_complex() is only supported for tensors with torch.dtype torch.float64 and torch.float32.
	# The input is expected to have the last dimension of size 2.
	# In addition, the tensor must have a stride of 1 for its last dimension.
	# The strides of all other dimensions must be even numbers.
	
	def float():
		pass

	def reshape(new_shape):
		pass

	def view_as_complex():
		pass

	def view_as_real():
		pass

	def type_as():
		pass

	def flatten():
		pass

	def view():
		pass


def zeros(shape):
	x = Tensor(shape)
	x.set_data(np.zeros(shape, dtype=dt))
	return x

def make_tensor(data):
	arr = np.array(data, dtype=dt)
	x = Tensor(arr.shape)
	x.set_data(arr)
	return x


# def ones(shape):
# 	return Tensor()


def Param(x):
	x.is_param = True
	return x


class Module:
	def __init__(self):
		self.is_param = True
		pass

class ModuleList(list):
	def __getattribute__(self, name: str):
		if name == 'is_param':
			return True
		return super().__getattribute__(name)

class Linear(Module):
	def __init__(self, in_features, out_features, bias=True):
		assert bias == False
		pass

class Embedding:
	def __init__(self, vocab_size, dim):
		pass


def ones(shape):
	pass

def rqsrt(x):
	pass

def arange(start = 0, stop, step=1):
	nums = []
	for i in range(start, stop, step):
		nums.append(i)
	return make_tensor(nums)

def outer(x, y):
	pass

def polar(abs: Tensor, angle: Tensor):
	pass

def ones_like(x):
	pass

def full(x):
	pass

def triu(x):
	pass

def F_silu(x):
	pass

def F_softmax(x):
	pass

def matmul(x, y):
	pass