import json, csv, sys
import re
import os
import status
import operator
import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy import sparse
import multiprocessing as mp
import itertools
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from threading import Thread
import threading
import time
import functools
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from random import shuffle
from common_utils import gen_csv_from_tuples, read_csv_list, read_csv_list2, read_csv_dict, make_query, pickle_object, unpickle_object, send_mail

#only on linux 
from common_utils import get_ram, get_elapsed_time

import pickle
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt



class MultFSScore(object):

	def __init__(self, identifier = None, user_removal = None, value_removal = None, cores = 56, dtype = np.float32):
		self.dtype = dtype
		self.identifier = identifier
		self.cores = cores
		self.dir = identifier + "_files/"
		self.data = self.dir + "user_to_" + identifier + ".csv"
		self.matrix_file = self.dir + identifier + ".matrix.dat"
		self.backup = True
		self.user_removal = user_removal
		self.value_removal = value_removal
		self.cleanup_list = []
		self.log = ""

	def pprint(self, *objects, sep=' ', end='\n', file=sys.stdout, flush=False):
		if end != '\r':
			for i in objects:
				self.log += str(i)
			self.log += "\n"
		print(*objects, sep=sep, end=end, file=file, flush=flush)

	def separate(self, x):
		pos = len(x) - x[::-1].find('[')
		y = x[:pos - 1]
		z = x[pos: - 1]
		return y, z

	""" 
	This function generates four data structures from the dataset:
		a) user_ind: mapping user to index.
		b) value_ind: mapping value to index.
		c) dictio_of_users: mapping a user index to all the indices of the values shared by that user.
		d) dictio_of_values: mapping a value index to all the indices of the users who shared that value.
	"""
	def gen_data(self):
		tic = time.time()
		#Create the path for storing the dictionaries
		user_ind_p=self.dir + 'user_ind.pkl'
		value_ind_p = self.dir + 'value_ind.pkl'
		dictio_of_users_p = self.dir + 'dictio_of_users.pkl'
		dictio_of_values_p = self.dir + 'dictio_of_values.pkl'
		dictio_of_usage_p = self.dir + 'dictio_of_usage.pkl'

		#Adding files to list for cleanup
		self.cleanup_list.append(user_ind_p), self.cleanup_list.append(value_ind_p), self.cleanup_list.append(dictio_of_users_p), self.cleanup_list.append(dictio_of_values_p), self.cleanup_list.append(dictio_of_usage_p)

		if self.backup and os.path.exists(user_ind_p) and os.path.exists(value_ind_p) and os.path.exists(dictio_of_users_p) and os.path.exists(dictio_of_values_p) and os.path.exists(dictio_of_usage_p):
			self.pprint("Data Structures already exist, unpickling.", end='\r')
			user_ind = unpickle_object(user_ind_p)
			value_ind = unpickle_object(value_ind_p)
			dictio_of_users = unpickle_object(dictio_of_users_p)
			dictio_of_values = unpickle_object(dictio_of_values_p)
			# TODO Remove comment
			#dictio_of_usage = unpickle_object(dictio_of_usage_p)
			dictio_of_usage = None
			self.pprint("[END] Data Structures already exist, unpickling.", get_ram(), get_elapsed_time(tic))
			return user_ind, value_ind, dictio_of_users, dictio_of_values, dictio_of_usage

		lst = read_csv_list(self.data)[1:]
		
		tic = time.time()
		user_ind = {}
		value_ind = {}
		dictio_of_users = {}
		dictio_of_values = {}
		dictio_of_usage = {}
		total = len(lst)
		max_val = np.uint32(0)
		for uind, i in enumerate(lst):
			if uind % 1000 == 0:
				self.pprint("Data Structures Generation", "[%d Users Processed]" %(uind), "[%0.3f Percentage]" % ((uind / total) * 100), get_ram(), get_elapsed_time(tic), end='\r')
			uind = np.uint32(uind)
			user_ind[i[0]] = uind
			user = i[0]
			dictio_of_users[uind] = []
			dictio_of_usage[uind] = []
			for t in i[1:]:
				value, usage = self.separate(t)
				usage = np.uint32(usage)
				if value not in value_ind:
					value_ind[value] = max_val
					dictio_of_values[max_val] = []
					max_val += 1
				vind = value_ind[value]
				dictio_of_values[vind].append(uind)
				dictio_of_users[uind].append(vind)
				dictio_of_usage[uind].append(usage)
		self.pprint("[END] Data Structures Generation", "[%d Users Processed]" %(uind), "[%0.3f Percentage]" % ((uind / total) * 100), get_ram(), get_elapsed_time(tic))
		
		lst = None # Freeing space from list, no longer needed

		#self.pprint("[0/5] Storing data structures to disk", get_ram(), get_elapsed_time(tic))
		#pickle_object(user_ind, user_ind_p)
		#self.pprint("[1/5] Storing data structures to disk", get_ram(), get_elapsed_time(tic))
		#pickle_object(value_ind, value_ind_p)
		#self.pprint("[2/5] Storing data structures to disk", get_ram(), get_elapsed_time(tic))
		#pickle_object(dictio_of_users, dictio_of_users_p)
		#self.pprint("[3/5] Storing data structures to disk", get_ram(), get_elapsed_time(tic))
		#pickle_object(dictio_of_values, dictio_of_values_p)
		#self.pprint("[4/5] Storing data structures to disk", get_ram(), get_elapsed_time(tic))
		#pickle_object(dictio_of_usage, dictio_of_usage_p)
		#self.pprint("[END] [5/5] Storing data structures to disk", get_ram(), get_elapsed_time(tic))
		return user_ind, value_ind, dictio_of_users, dictio_of_values, dictio_of_usage
	""" 
	This function generates a sparse matrix whose rows represent a user, and whose columns represent the different values. The matrix is sparse and has a 1 where a user shared a value.
	"""
	def gen_sparse_matrix(self, dictio_of_users, dictio_of_usage, num_values):
		tic = time.time()
		sparse_matrix_p = self.dir + 'sparse_matrix.pkl'

		#Adding files to list for cleanup
		self.cleanup_list.append(sparse_matrix_p)

		if self.backup and os.path.exists(sparse_matrix_p):
			self.pprint("Sparse Matrix already exist, unpickling.", end='\r')
			sparse_matrix = unpickle_object(sparse_matrix_p)
			self.pprint("[END] Sparse Matrix already exist, unpickling.", get_ram(), get_elapsed_time(tic))
			return sparse_matrix

		num_users = len(dictio_of_users)
		rows = []
		cols = []
		data = []
		for ind, row in enumerate(dictio_of_users.items()):
			if ind % 1000 == 0:
				self.pprint("Sparse Matrix Generation", "[%d Users Processed]" %(ind), "[%0.3f Percentage]" % ((ind / num_users) * 100), get_ram(), get_elapsed_time(tic), end='\r')
			uind, values = row[0], row[1]
			usages = dictio_of_usage[uind]
			for value, usage in zip(values, usages):
				rows.append(uind)
				cols.append(value)
				data.append(usage)

		
		sparse_matrix = csr_matrix((data, (rows, cols)), shape=(num_users, num_values), dtype=self.dtype)
		self.pprint("[END] Sparse Matrix Generation", "[%d Users Processed]" %(ind), "[%0.3f Percentage]" % ((ind / num_users) * 100), get_ram(), get_elapsed_time(tic))
		pickle_object(sparse_matrix, sparse_matrix_p)
		return sparse_matrix



	def remove_user_value_set(self, sparse_matrix, user_ind, value_ind, dictio_of_users, dictio_of_values, user_set, value_set):
		tic = time.time()
		# We remove the data from the matrix
		users, values = sparse_matrix.shape
		rem_users = sorted(list(user_set))
		rem_values = sorted(list(value_set))
		self.pprint("[STATUS] [Original] Users: [%d] Values: [%d]" % (users, values))
		sparse_matrix = sparse_matrix[rem_users, :]
		users, values = sparse_matrix.shape
		self.pprint("[STATUS] [User Removal] Users: [%d] Values: [%d]" % (users, values), get_ram(), get_elapsed_time(tic))
		sparse_matrix = sparse_matrix[:, rem_values]
		users, values = sparse_matrix.shape
		self.pprint("[STATUS] [Value Removal] Users: [%d] Values: [%d]" % (users, values), get_ram(), get_elapsed_time(tic))


		# Users to remove
		self.pprint("Generating new user data.", get_ram(), get_elapsed_time(tic), end='\r')
		temp = {user: uind for user, uind in user_ind.items() if uind in user_set} # Remove not needed users
		#print(len(temp))
		trans_user_dict = {uind: new_uind for new_uind, (_, uind) in enumerate(temp.items())} # Translate from current index to new index
		#print(len(temp), len(trans_user_dict))
		user_ind = {user: trans_user_dict[uind] for user, uind in temp.items()} # Update user_ind with new indexes
		self.pprint("[END] Generating new user data: %d" %(len(user_ind)), get_ram(), get_elapsed_time(tic))
		self.pprint("Generating new value data.", get_ram(), get_elapsed_time(tic), end='\r')
		temp = {value: vind for value, vind in value_ind.items() if vind in value_set} # Remove not needed values
		trans_value_dict = {vind: new_vind for new_vind, (_, vind) in enumerate(temp.items())} # Translate from current index to new index
		value_ind = {value: trans_value_dict[vind] for value, vind in temp.items()} # Update the value_ind with new indexes
		self.pprint("[END] Generating new value data: %d" % (len(value_ind)), get_ram(), get_elapsed_time(tic))

		dictio_of_users = {trans_user_dict[uind]: [trans_value_dict[vind] for vind in vinds if vind in value_set] for uind, vinds in dictio_of_users.items() if uind in user_set}
		dictio_of_values = {trans_value_dict[vind]: [trans_user_dict[uind] for uind in uinds if uind in user_set] for vind, uinds in dictio_of_values.items() if vind in value_set}

		return sparse_matrix, user_ind, value_ind, dictio_of_users, dictio_of_values

	"""
	This function makes use of the sparse matrix to accelerate the removal of the rows and columns and the cleanup.
	"""
	def clean_matrix(self, sparse_matrix, user_ind, value_ind, dictio_of_users, dictio_of_values):
		tic = time.time()
		
		#inv_user_ind = {v: k for k, v in user_ind.items()}
		#inv_value_ind = {v: k for k, v in value_ind.items()}

		sparse_matrix_p = self.dir + 'clean_sparse_matrix.pkl'
		user_ind_p=self.dir + 'clean_user_ind.pkl'
		value_ind_p = self.dir + 'clean_value_ind.pkl'
		dictio_of_users_p = self.dir + 'clean_dictio_of_users.pkl'
		dictio_of_values_p = self.dir + 'clean_dictio_of_values.pkl'

		#Adding files to list for cleanup
		self.cleanup_list.append(sparse_matrix_p), self.cleanup_list.append(user_ind_p), self.cleanup_list.append(value_ind_p), self.cleanup_list.append(dictio_of_users_p), self.cleanup_list.append(dictio_of_values_p)

		if self.backup and os.path.exists(sparse_matrix_p) and os.path.exists(user_ind_p) and os.path.exists(value_ind_p) and os.path.exists(dictio_of_users_p) and os.path.exists(dictio_of_values_p):
			self.pprint("Clean data already exist, unpickling.", end='\r')
			user_ind = unpickle_object(user_ind_p)
			value_ind = unpickle_object(value_ind_p)
			dictio_of_users = unpickle_object(dictio_of_users_p)
			dictio_of_values = unpickle_object(dictio_of_values_p)
			sparse_matrix = unpickle_object(sparse_matrix_p)
			self.pprint("[END] Clean data already exist, unpickling.", get_ram(), get_elapsed_time(tic))
			return sparse_matrix, user_ind, value_ind, dictio_of_users, dictio_of_values

		user_set = set(dictio_of_users.keys())
		self.pprint("Taking values that appear once.", get_ram(), get_elapsed_time(tic), end='\r')
		value_set =  set([k for k,v in dictio_of_values.items() if len(v) > 1])
		self.pprint("[END] Taking values that appear once", "[Process Remaining: %d] [User Set: %d] [Value Set: %d]" % (len(value_set), len(user_set), len(value_set)), get_ram(), get_elapsed_time(tic))

		# We execute all user removal procedures specified
		if not self.user_removal is None:
			for ind, procedure in enumerate(self.user_removal):
				self.pprint("Executing user removal procedure [%d] " % (ind), get_ram(), get_elapsed_time(tic), end='\r')
				user_list = procedure(user_ind, value_ind, dictio_of_users, dictio_of_values)
				user_set = user_set.intersection(set(user_list))
				self.pprint("[END] Executing user removal procedure [%d]" % (ind + 1), "[Process Remaining: %d] [User Set: %d] [Value Set: %d]" % (len(user_list), len(user_set), len(value_set)), get_ram(), get_elapsed_time(tic))

		# We execute all value removal procedures specified by the user
		if not self.value_removal is None:
			for ind, procedure in enumerate(self.value_removal):
				self.pprint("Executing value removal procedure [%d]" % (ind), get_ram(), get_elapsed_time(tic), end='\r')
				value_list = procedure(user_ind, value_ind, dictio_of_users, dictio_of_values)
				value_set = value_set.intersection(set(value_list))
				self.pprint("[END] Executing value removal procedure [%d]" % (ind + 1), "[Process Remaining: %d] [User Set: %d] [Value Set: %d]" % (len(value_list), len(user_set), len(value_set)), get_ram(), get_elapsed_time(tic))

		sparse_matrix, user_ind, value_ind, dictio_of_users, dictio_of_values = self.remove_user_value_set(sparse_matrix, user_ind, value_ind, dictio_of_users, dictio_of_values, user_set, value_set)
		
		self.pprint("Obtaining empty data", end='\r')
		user_set = set(dictio_of_users.keys())
		value_set = set(dictio_of_values.keys())
		user_set_rem = set([uind for uind, vinds in dictio_of_users.items() if len(vinds) == 0])
		value_set_rem = set([vind for vind, uinds in dictio_of_values.items() if len(uinds) == 0])
		user_set = user_set.difference(user_set_rem)
		value_set = value_set.difference(value_set_rem)
		self.pprint("[END] Obtaining empty data")
		sparse_matrix, user_ind, value_ind, dictio_of_users, dictio_of_values = self.remove_user_value_set(sparse_matrix, user_ind, value_ind, dictio_of_users, dictio_of_values, user_set, value_set)


		pickle_object(sparse_matrix, sparse_matrix_p)
		pickle_object(user_ind, user_ind_p)
		pickle_object(value_ind, value_ind_p)
		pickle_object(dictio_of_users, dictio_of_users_p)
		pickle_object(dictio_of_values, dictio_of_values_p)
		return sparse_matrix, user_ind, value_ind, dictio_of_users, dictio_of_values

	def calculate_tfidf(self, sparse_matrix):
		tic = time.time()
		#sparse_matrix, user_ind, value_ind, dictio_of_users, dictio_of_values, weight_vector = multfsscore.structurize()
		self.pprint("Computing TF-IDF", get_ram(), get_elapsed_time(tic), end='\r')
		num_users, num_values = sparse_matrix.shape

		tf = sparse_matrix.multiply(1 / sparse_matrix.sum(axis=1)) # User / Total trigrams of user
		idf = np.log(num_users / (sparse_matrix != 0).sum(axis= 0)) # Num USers / Total that make use of it
		tfidf = tf.multiply(idf)
		print(tfidf.shape)
		tfidf.eliminate_zeros()
		print(tfidf.shape)
		tfidf = tfidf.tocsr()
		self.pprint("[END] Computing TF-IDF", get_ram(), get_elapsed_time(tic))
		return tfidf

	def compute_matrix_mult(self, sparse_matrix):
		tic = time.time()
		sparse_matrix_dot_p = self.dir + 'sparse_matrix_dot_a.pkl'
		print(sparse_matrix.shape)
		self.pprint("Executing coincidence computation over matrix", get_ram(), get_elapsed_time(tic), end='\r')	
		sparse_matrix_dot = sparse_matrix.dot(sparse_matrix.T)
		#sparse_matrix_dot = sparse_matrix_dot.astype(dtype='int32')
		sparse_matrix_dot = sparse.triu(sparse_matrix_dot, format='csr')
		print(sparse_matrix_dot.shape)
		sparse_matrix_dot.eliminate_zeros()
		pickle_object(sparse_matrix_dot, sparse_matrix_dot_p)
		self.pprint("[END] Executing coincidence computation over matrix", get_ram(), get_elapsed_time(tic))
		print(sparse_matrix_dot.shape)
		return sparse_matrix_dot

	def get_information_from_matrix(self, user_ind, sparse_matrix_dot):
		tic = time.time()
		lst_res = []
		inv_user_ind = {v:k for k, v in user_ind.items()}
		num_users = len(user_ind)

		#self.pprint("Transforming Matrix A", end='\r')
		#sparse_matrix_dot = sparse_matrix_dot.tocoo()
		#row, col, data = sparse_matrix_dot.row, sparse_matrix_dot.col, sparse_matrix_dot.data
		#self.pprint("[END] Transforming Matrix A")
		lst_res = []
		tx = sparse_matrix_dot.shape[0]
		print(sparse_matrix_dot.shape)
		for uind in range(tx):
			if uind % 100 == 0:
				self.pprint("Info Extraction", "[%d Users Processed]" %(uind), "[%d List Length]" %(len(lst_res)), "[%0.3f Percentage]" % ((uind / tx) * 100), get_ram(), get_elapsed_time(tic), end='\r')
			row = np.array(sparse_matrix_dot[uind].toarray())
			row = row.flatten()
			row[uind] = 0 # We do not consider the comparison with itself
			rmax = row.max()
			if rmax > 0:
				n = (row == rmax).sum()
				#max_inds = row.argsort()[-n:][::-1]
				max_uinds = np.argpartition(row, -n)[-n:] # it orders "-n" elements of the row, and then, it extracts the last n.

				for i in max_uinds:
					lst_res.append((inv_user_ind[uind], inv_user_ind[i], rmax))

		lst_res = sorted(lst_res, key=lambda x: x[2], reverse=True)
		pickle_object(lst_res, self.dir + "results.pkl")
		self.pprint("[END] Info Extraction", "[%d Users Processed]" %(uind), "[%d List Length]" %(len(lst_res)), "[%0.3f Percentage]" % ((uind / tx) * 100), get_ram(), get_elapsed_time(tic))
		
		gen_csv_from_tuples(self.dir + "results.csv", ["User1", "User2", "Relation Value"], lst_res)

		return lst_res


	def structurize(self):
		tic = time.time()
		# Reading list of users
		self.backup = False
		if self.backup and os.path.exists(self.dir + 'clean_sparse_matrix.pkl'):
			sparse_matrix, user_ind, value_ind, dictio_of_users, dictio_of_values = self.clean_matrix(*tuple([None] * 5))
			return sparse_matrix, user_ind, value_ind, dictio_of_users, dictio_of_values

		user_ind, value_ind, dictio_of_users, dictio_of_values, dictio_of_usage = self.gen_data() # No longer needed
		sparse_matrix = self.gen_sparse_matrix( dictio_of_users, dictio_of_usage, len(dictio_of_values))
		dictio_of_usage = None # No longer needed
		
		#u11, u21 = self.test("96142[4]", "100917[4]", user_ind, value_ind, dictio_of_users)
		#self.backup = False
		sparse_matrix, user_ind, value_ind, dictio_of_users, dictio_of_values = self.clean_matrix(sparse_matrix, user_ind, value_ind, dictio_of_users, dictio_of_values)
		#u12, u22 = self.test("96142[4]", "100917[4]", user_ind, value_ind, dictio_of_users)
		#print("SELF INTERSECTION", u11.intersection(u12))
		#print("SELF INTERSECTION", u21.intersection(u22))
		return sparse_matrix, user_ind, value_ind, dictio_of_users, dictio_of_values#, weight_vector

	def test(self, u1, u2, user_ind, value_ind, dictio_of_users):
		if u1 in user_ind and u2 in user_ind:
			inv_value_ind = {v:k for k, v in value_ind.items()}
			u1l = set([inv_value_ind[v] for v in dictio_of_users[user_ind[u1]]])
			u2l = set([inv_value_ind[v] for v in dictio_of_users[user_ind[u2]]])
			print("INTERSECTION", sorted(list(u1l.intersection(u2l))) )
			return u1l, u2l
		else:
			print("NO INTERSECTION")
		#print("USER1", u1l)
		#print("USER2", u2l)
	
	def get_size(self):
		user_ind, value_ind, dictio_of_users, dictio_of_values, dictio_of_usage = self.gen_data() # No longer needed
		return len(user_ind), len(value_ind)

	def get_size_clean(self):
		sparse_matrix, user_ind, value_ind, dictio_of_users, dictio_of_values = self.clean_matrix(*tuple([None] * 5))
		return sparse_matrix.shape

	def compute(self):
		tic = time.time()
		self.pprint("[>>>] Starting with %s [<<<]" % (self.identifier))
		#if self.backup and os.path.exists(self.dir + "results.pkl"):
			#return
		
		sparse_matrix, user_ind, value_ind, dictio_of_users, dictio_of_values = self.structurize()
		(value_ind, dictio_of_users, dictio_of_values) = tuple([None] * 3)
		sparse_matrix = self.calculate_tfidf(sparse_matrix)

		sparse_matrix_dot = self.compute_matrix_mult(sparse_matrix)
		#return sparse_matrix_dot
		sparse_matrix = None
		self.get_information_from_matrix(user_ind, sparse_matrix_dot)

		self.pprint("[>>>] Ended with %s [<<<]" % (self.identifier), get_ram(), get_elapsed_time(tic))
		self.pprint (self.log)
		# send_mail(self.log)

class FeatureScore:
	def __init__(self, *kwargs):
		pass
