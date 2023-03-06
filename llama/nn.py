
class Tensor:
	def __init__(self):
		self.is_param = False
		pass

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
	return Tensor()

def ones(shape):
	return Tensor()


def Param(x):
	x.is_param = True
	return x


class Module:
	def __init__(self):
		pass


class Linear(Module):
	def __init__(self) -> None:
		pass



def ones(shape):
	pass

def rqsrt(x):
	pass

def arange():
	pass

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