#!/usr/bin/env python

from ctypes import *
from ctypes.util import find_library
from os import path
import sys

__all__ = ['liblinear', 'feature_node', 'gen_feature_nodearray', 'problem',
           'parameter', 'model', 'toPyModel', 'L2R_LR', 'L2R_L2LOSS_SVC_DUAL',
           'L2R_L2LOSS_SVC', 'L2R_L1LOSS_SVC_DUAL', 'MCSVM_CS', 
           'L1R_L2LOSS_SVC', 'L1R_LR', 'L2R_LR_DUAL', 'L2R_L2LOSS_SVR', 
           'L2R_L2LOSS_SVR_DUAL', 'L2R_L1LOSS_SVR_DUAL', 'print_null']

try:
	dirname = path.dirname(path.abspath(__file__))
	if sys.platform == 'win32':
		liblinear = CDLL(path.join(dirname, r'..\windows\liblinear.dll'))
	else:
		liblinear = CDLL(path.join(dirname, '../liblinear.so.3'))
except:
# For unix the prefix 'lib' is not considered.
	if find_library('linear'):
		liblinear = CDLL(find_library('linear'))
	elif find_library('liblinear'):
		liblinear = CDLL(find_library('liblinear'))
	else:
		raise Exception('LIBLINEAR library not found.')

L2R_LR = 0
L2R_L2LOSS_SVC_DUAL = 1 
L2R_L2LOSS_SVC = 2 
L2R_L1LOSS_SVC_DUAL = 3
MCSVM_CS = 4 
L1R_L2LOSS_SVC = 5 
L1R_LR = 6 
L2R_LR_DUAL = 7  
L2R_L2LOSS_SVR = 11
L2R_L2LOSS_SVR_DUAL = 12
L2R_L1LOSS_SVR_DUAL = 13

PRINT_STRING_FUN = CFUNCTYPE(None, c_char_p)
def print_null(s): 
	return 

def genFields(names, types): 
	return list(zip(names, types))

def fillprototype(f, restype, argtypes): 
	f.restype = restype
	f.argtypes = argtypes

class feature_node(Structure):
	_names = ["index", "value"]
	_types = [c_int, c_double]
	_fields_ = genFields(_names, _types)

	def __str__(self):
		return '%d:%g' % (self.index, self.value)

def gen_feature_nodearray(xi, feature_max=None, issparse=True):
	if isinstance(xi, dict):
		index_range = list(xi.keys())
	elif isinstance(xi, (list, tuple)):
		xi = [0] + xi  # idx should start from 1
		index_range = list(range(1, len(xi)))
	else:
		raise TypeError('xi should be a dictionary, list or tuple')

	if feature_max:
		assert(isinstance(feature_max, int))
		index_range = [j for j in index_range if j <= feature_max]
	if issparse: 
		index_range = [j for j in index_range if xi[j] != 0]

	index_range = sorted(index_range)
	ret = (feature_node * (len(index_range)+2))()
	ret[-1].index = -1 # for bias term
	ret[-2].index = -1
	for idx, j in enumerate(index_range):
		ret[idx].index = j
		ret[idx].value = xi[j]
	max_idx = 0
	if index_range : 
		max_idx = index_range[-1]
	return ret, max_idx

class problem(Structure):
	_names = ["l", "n", "y", "x", "bias"]
	_types = [c_int, c_int, POINTER(c_double), POINTER(POINTER(feature_node)), c_double]
	_fields_ = genFields(_names, _types)

	def __init__(self, y, x, bias = -1):
		if len(y) != len(x) :
			raise ValueError("len(y) != len(x)")
		self.l = l = len(y)
		self.bias = -1

		max_idx = 0
		x_space = self.x_space = []
		for i, xi in enumerate(x):
			tmp_xi, tmp_idx = gen_feature_nodearray(xi)
			x_space += [tmp_xi]
			max_idx = max(max_idx, tmp_idx)
		self.n = max_idx

		self.y = (c_double * l)()
		for i, yi in enumerate(y): self.y[i] = y[i]

		self.x = (POINTER(feature_node) * l)() 
		for i, xi in enumerate(self.x_space): self.x[i] = xi

		self.set_bias(bias)

	def set_bias(self, bias):
		if self.bias == bias:
			return 
		if bias >= 0 and self.bias < 0: 
			self.n += 1
			node = feature_node(self.n, bias)
		if bias < 0 and self.bias >= 0: 
			self.n -= 1
			node = feature_node(-1, bias)

		for xi in self.x_space:
			xi[-2] = node
		self.bias = bias


class parameter(Structure):
	_names = ["solver_type", "eps", "C", "nr_thread", "nr_weight", "weight_label", "weight", "p", "init_sol"]
	_types = [c_int, c_double, c_double, c_int, c_int, POINTER(c_int), POINTER(c_double), c_double, POINTER(c_double)]
	_fields_ = genFields(_names, _types)

	def __init__(self, options = None):
		if options == None:
			options = ''
		self.parse_options(options)

	def __str__(self):
		s = ''
		attrs = parameter._names + list(self.__dict__.keys())
		values = [getattr(self, attr) for attr in attrs] 
		for attr, val in zip(attrs, values):
			s += (' %s: %s\n' % (attr, val))
		s = s.strip()

		return s

	def set_to_default_values(self):
		self.solver_type = L2R_L2LOSS_SVC_DUAL
		self.eps = float('inf')
		self.C = 1
		self.p = 0.1
		self.nr_thread = 1
		self.nr_weight = 0
		self.weight_label = None
		self.weight = None
		self.init_sol = None
		self.bias = -1
		self.flag_cross_validation = False
		self.flag_C_specified = False
		self.flag_solver_specified = False
		self.flag_find_C = False
		self.flag_omp = False
		self.nr_fold = 0
		self.print_func = cast(None, PRINT_STRING_FUN)

	def parse_options(self, options):
		if isinstance(options, list):
			argv = options
		elif isinstance(options, str):
			argv = options.split()
		else:
			raise TypeError("arg 1 should be a list or a str.")
		self.set_to_default_values()
		self.print_func = cast(None, PRINT_STRING_FUN)
		weight_label = []
		weight = []

		i = 0
		while i < len(argv) :
			if argv[i] == "-s":
				i = i + 1
				self.solver_type = int(argv[i])
				self.flag_solver_specified = True
			elif argv[i] == "-c":
				i = i + 1
				self.C = float(argv[i])
				self.flag_C_specified = True
			elif argv[i] == "-p":
				i = i + 1
				self.p = float(argv[i])
			elif argv[i] == "-e":
				i = i + 1
				self.eps = float(argv[i])
			elif argv[i] == "-B":
				i = i + 1
				self.bias = float(argv[i])
			elif argv[i] == "-v":
				i = i + 1
				self.flag_cross_validation = 1
				self.nr_fold = int(argv[i])
				if self.nr_fold < 2 :
					raise ValueError("n-fold cross validation: n must >= 2")
			elif argv[i] == "-n":
				i = i + 1
				self.flag_omp = True
				self.nr_thread = int(argv[i])
			elif argv[i].startswith("-w"):
				i = i + 1
				self.nr_weight += 1
				weight_label += [int(argv[i-1][2:])]
				weight += [float(argv[i])]
			elif argv[i] == "-q":
				self.print_func = PRINT_STRING_FUN(print_null)
			elif argv[i] == "-C":
				self.flag_find_C = True

			else :
				raise ValueError("Wrong options")
			i += 1

		liblinear.set_print_string_function(self.print_func)
		self.weight_label = (c_int*self.nr_weight)()
		self.weight = (c_double*self.nr_weight)()
		for i in range(self.nr_weight): 
			self.weight[i] = weight[i]
			self.weight_label[i] = weight_label[i]

		# default solver for parameter selection is L2R_L2LOSS_SVC
		if self.flag_find_C:
			if not self.flag_cross_validation:
				self.nr_fold = 5
			if not self.flag_solver_specified:
				self.solver_type = L2R_L2LOSS_SVC
				self.flag_solver_specified = True
			elif self.solver_type not in [L2R_LR, L2R_L2LOSS_SVC]:
				raise ValueError("Warm-start parameter search only available for -s 0 and -s 2")
		if self.flag_omp:
			if not self.flag_solver_specified:
				self.solver_type = L2R_L2LOSS_SVC
				self.flag_solver_specified = True
			elif self.solver_type not in [L2R_LR, L2R_L2LOSS_SVC, L2R_L2LOSS_SVR, L2R_L2LOSS_SVC_DUAL, L2R_L1LOSS_SVC_DUAL]:
				raise ValueError("Parallel LIBLINEAR is only available for -s 0, 1, 2, 3, 11 now")
	
		if self.eps == float('inf'):
			if self.solver_type in [L2R_LR, L2R_L2LOSS_SVC]:
				self.eps = 0.01
			elif self.solver_type in [L2R_L2LOSS_SVR]:
				self.eps = 0.001
			elif self.solver_type in [L2R_L2LOSS_SVC_DUAL, L2R_L1LOSS_SVC_DUAL, MCSVM_CS, L2R_LR_DUAL]:
				self.eps = 0.1
			elif self.solver_type in [L1R_L2LOSS_SVC, L1R_LR]:
				self.eps = 0.01
			elif self.solver_type in [L2R_L2LOSS_SVR_DUAL, L2R_L1LOSS_SVR_DUAL]:
				self.eps = 0.1

class model(Structure):
	_names = ["param", "nr_class", "nr_feature", "w", "label", "bias"]
	_types = [parameter, c_int, c_int, POINTER(c_double), POINTER(c_int), c_double]
	_fields_ = genFields(_names, _types)

	def __init__(self):
		self.__createfrom__ = 'python'

	def __del__(self):
		# free memory created by C to avoid memory leak
		if hasattr(self, '__createfrom__') and self.__createfrom__ == 'C':
			liblinear.free_and_destroy_model(pointer(self))

	def get_nr_feature(self):
		return liblinear.get_nr_feature(self)

	def get_nr_class(self):
		return liblinear.get_nr_class(self)

	def get_labels(self):
		nr_class = self.get_nr_class()
		labels = (c_int * nr_class)()
		liblinear.get_labels(self, labels)
		return labels[:nr_class]

	def get_decfun_coef(self, feat_idx, label_idx=0):
		return liblinear.get_decfun_coef(self, feat_idx, label_idx)

	def get_decfun_bias(self, label_idx=0):
		return liblinear.get_decfun_bias(self, label_idx)

	def get_decfun(self, label_idx=0):
		w = [liblinear.get_decfun_coef(self, feat_idx, label_idx) for feat_idx in range(1, self.nr_feature+1)]
		b = liblinear.get_decfun_bias(self, label_idx)
		return (w, b)

	def is_probability_model(self):
		return (liblinear.check_probability_model(self) == 1)

	def is_regression_model(self):
		return (liblinear.check_regression_model(self) == 1)

def toPyModel(model_ptr):
	"""
	toPyModel(model_ptr) -> model

	Convert a ctypes POINTER(model) to a Python model
	"""
	if bool(model_ptr) == False:
		raise ValueError("Null pointer")
	m = model_ptr.contents
	m.__createfrom__ = 'C'
	return m

fillprototype(liblinear.train, POINTER(model), [POINTER(problem), POINTER(parameter)])
fillprototype(liblinear.find_parameter_C, None, [POINTER(problem), POINTER(parameter), c_int, c_double, c_double, POINTER(c_double), POINTER(c_double)])
fillprototype(liblinear.cross_validation, None, [POINTER(problem), POINTER(parameter), c_int, POINTER(c_double)])

fillprototype(liblinear.predict_values, c_double, [POINTER(model), POINTER(feature_node), POINTER(c_double)])
fillprototype(liblinear.predict, c_double, [POINTER(model), POINTER(feature_node)])
fillprototype(liblinear.predict_probability, c_double, [POINTER(model), POINTER(feature_node), POINTER(c_double)])

fillprototype(liblinear.save_model, c_int, [c_char_p, POINTER(model)])
fillprototype(liblinear.load_model, POINTER(model), [c_char_p])

fillprototype(liblinear.get_nr_feature, c_int, [POINTER(model)])
fillprototype(liblinear.get_nr_class, c_int, [POINTER(model)])
fillprototype(liblinear.get_labels, None, [POINTER(model), POINTER(c_int)])
fillprototype(liblinear.get_decfun_coef, c_double, [POINTER(model), c_int, c_int])
fillprototype(liblinear.get_decfun_bias, c_double, [POINTER(model), c_int])

fillprototype(liblinear.free_model_content, None, [POINTER(model)])
fillprototype(liblinear.free_and_destroy_model, None, [POINTER(POINTER(model))])
fillprototype(liblinear.destroy_param, None, [POINTER(parameter)])
fillprototype(liblinear.check_parameter, c_char_p, [POINTER(problem), POINTER(parameter)])
fillprototype(liblinear.check_probability_model, c_int, [POINTER(model)])
fillprototype(liblinear.check_regression_model, c_int, [POINTER(model)])
fillprototype(liblinear.set_print_string_function, None, [CFUNCTYPE(None, c_char_p)])
