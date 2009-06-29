import math
from numpy import sqrt, dot, shape, take, array
from numpy.random import shuffle
from memoized import memoized

def normalize(v):
	return v/sqrt(dot(v, v))

def log_factory(base):
	def f(x):
		return math.log(x, base)
	return f

log2 = log_factory(2)
logn = log_factory(math.e)

def shuffle_matrix(m):
	"""Shuffles the rows of a matrix."""
	r, c = shape(m)
	idx = range(r)
	shuffle(idx)
	return take(m, idx, 0)

def crc16str(s):
	"""Returns the 16-bit CRC value of a string. Equivalent to icrc1 in Press."""
	crc = 0
	for index1 in range(len(s)):
		crc = crc ^ (ord(s[index1]) << 8)
		for index2 in range(1, 9):
			if crc & 0x8000 != 0:
				crc = ((crc << 1) ^ 0x1021)
			else:
				crc = crc << 1
	return crc & 0xFFFF

@memoized
def fmt_integer(b):
	if b == 8:
		return 'B'
	elif b == 16:
		return 'H'
	elif b == 32:
		return 'I'
	else:
		print "Cannot handle integers of bit size %d" % b
		return None

@memoized
def mask_integer(b, ub):
	if b == 8:
		return (0xFF >> (b-ub))
	elif b == 16:
		return (0xFFFF >> (b-ub))
	elif b == 32:
		return (0xFFFFFFFF >> (b-ub))
	else:
		print "Cannot handle integers of bit size %d" % b
		return None

if __name__ == '__main__':
	x = array([[1,1,2],
		   [2,2,3],
		   [3,3,4],
		   [4,4,5],
		   [5,5,6]])
	print shuffle_matrix(x)
	print log2(1024)
	print logn(1024)
	print crc16str("skedaddle")
	print crc16str("gobbledygook")

