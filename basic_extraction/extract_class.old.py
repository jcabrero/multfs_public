import json, csv
import re
import os
import status
import operator
import math
import numpy as np
from scipy.sparse import csr_matrix
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
from common_utils import gen_csv_from_tuples, read_csv_list, read_csv_list2, read_csv_dict, make_query, pickle_object, unpickle_object

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

		if self.backup and os.path.exists(user_ind_p) and os.path.exists(value_ind_p) and os.path.exists(dictio_of_users_p) and os.path.exists(dictio_of_values_p):
			print("Data Structures already exist, unpickling.", end='\r')
			user_ind = unpickle_object(user_ind_p)
			value_ind = unpickle_object(value_ind_p)
			dictio_of_users = unpickle_object(dictio_of_users_p)
			dictio_of_values = unpickle_object(dictio_of_values_p)
			print("[END] Data Structures already exist, unpickling.", get_ram(), get_elapsed_time(tic))
			return user_ind, value_ind, dictio_of_users, dictio_of_values

		lst = read_csv_list(self.data)[1:]
		
		tic = time.time()
		user_ind = {}
		value_ind = {}
		dictio_of_users = {}
		dictio_of_values = {}
		total = len(lst)
		max_val = np.uint32(0)
		for uind, i in enumerate(lst):
			if uind % 1000 == 0:
				print("Data Structures Generation", "[%d Users Processed]" %(uind), "[%0.3f Percentage]" % ((uind / total) * 100), get_ram(), get_elapsed_time(tic), end='\r')
			user_ind[i[0]] = np.uint32(uind)
			user = i[0]
			dictio_of_users[uind] = []
			for t in i[1:]:
				value, _ = self.separate(t)
				if value not in value_ind:
					value_ind[value] = max_val
					dictio_of_values[max_val] = []
					max_val += 1
				vind = value_ind[value]
				dictio_of_values[vind].append(uind)
				dictio_of_users[uind].append(vind)
		print("[END] Data Structures Generation", "[%d Users Processed]" %(uind), "[%0.3f Percentage]" % ((uind / total) * 100), get_ram(), get_elapsed_time(tic))

		pickle_object(user_ind, user_ind_p)
		pickle_object(value_ind, value_ind_p)
		pickle_object(dictio_of_users, dictio_of_users_p)
		pickle_object(dictio_of_values, dictio_of_values_p)

		return user_ind, value_ind, dictio_of_users, dictio_of_values
	""" 
	This function generates a sparse matrix whose rows represent a user, and whose columns represent the different values. The matrix is sparse and has a 1 where a user shared a value.
	"""
	def gen_sparse_matrix(self, dictio_of_users, num_values):
		tic = time.time()
		sparse_matrix_p = self.dir + 'sparse_matrix.pkl'

		if self.backup and os.path.exists(sparse_matrix_p):
			print("Sparse Matrix already exist, unpickling.", end='\r')
			sparse_matrix = unpickle_object(sparse_matrix_p)
			print("[END] Sparse Matrix already exist, unpickling.", get_ram(), get_elapsed_time(tic))
			return sparse_matrix

		num_users = len(dictio_of_users)
		rows = []
		cols = []
		data = []
		for ind, row in enumerate(dictio_of_users.items()):
			if ind % 1000 == 0:
				print("Sparse Matrix Generation", "[%d Users Processed]" %(ind), "[%0.3f Percentage]" % ((ind / num_users) * 100), get_ram(), get_elapsed_time(tic), end='\r')
			user, values = row[0], row[1]

			for value in values:
				rows.append(user)
				cols.append(value)
				data.append(1)

		
		sparse_matrix = csr_matrix((data, (rows, cols)), shape=(num_users, num_values), dtype=self.dtype)
		print("[END] Sparse Matrix Generation", "[%d Users Processed]" %(ind), "[%0.3f Percentage]" % ((ind / num_users) * 100), get_ram(), get_elapsed_time(tic))
		pickle_object(sparse_matrix, sparse_matrix_p)
		return sparse_matrix
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
		
		if self.backup and os.path.exists(sparse_matrix_p) and os.path.exists(user_ind_p) and os.path.exists(value_ind_p) and os.path.exists(dictio_of_users_p) and os.path.exists(dictio_of_values_p):
			print("Clean data already exist, unpickling.", end='\r')
			user_ind = unpickle_object(user_ind_p)
			value_ind = unpickle_object(value_ind_p)
			dictio_of_users = unpickle_object(dictio_of_users_p)
			dictio_of_values = unpickle_object(dictio_of_values_p)
			sparse_matrix = unpickle_object(sparse_matrix_p)
			print("[END] Clean data already exist, unpickling.", get_ram(), get_elapsed_time(tic))
			return sparse_matrix, user_ind, value_ind, dictio_of_users, dictio_of_values

		user_set = set(dictio_of_users.keys())
		print("Taking values that appear once.", get_ram(), get_elapsed_time(tic), end='\r')
		value_set =  set([k for k,v in dictio_of_values.items() if len(v) > 1])
		print("[END] Taking values that appear once", "[Process Remaining: %d] [User Set: %d] [Value Set: %d]" % (len(value_set), len(user_set), len(value_set)), get_ram(), get_elapsed_time(tic))

		# We execute all user removal procedures specified
		if not self.user_removal is None:
			for ind, procedure in enumerate(self.user_removal):
				print("Executing user removal procedure [%d] " % (ind), get_ram(), get_elapsed_time(tic), end='\r')
				user_list = procedure(user_ind, value_ind, dictio_of_users, dictio_of_values)
				user_set = user_set.intersection(set(user_list))
				print("[END] Executing user removal procedure [%d]" % (ind + 1), "[Process Remaining: %d] [User Set: %d] [Value Set: %d]" % (len(user_list), len(user_set), len(value_set)), get_ram(), get_elapsed_time(tic))

		# We execute all value removal procedures specified by the user
		if not self.value_removal is None:
			for procedure in self.value_removal:
				print("Executing value removal procedure [%d]" % (ind), get_ram(), get_elapsed_time(tic), end='\r')
				value_list = procedure(user_ind, value_ind, dictio_of_users, dictio_of_values)
				value_set = value_set.intersection(set(value_list))
				print("[END] Executing value removal procedure [%d]" % (ind + 1), "[Process Remaining: %d] [User Set: %d] [Value Set: %d]" % (len(value_list), len(user_set), len(value_set)), get_ram(), get_elapsed_time(tic))

		# We remove the data from the matrix
		users, values = sparse_matrix.shape
		print("[STATUS] [Original] Users: [%d] Values: [%d]" % (users, values))
		sparse_matrix = sparse_matrix[list(user_set), :]
		users, values = sparse_matrix.shape
		print("[STATUS] [User Removal] Users: [%d] Values: [%d]" % (users, values), get_ram(), get_elapsed_time(tic))
		sparse_matrix = sparse_matrix[:, list(value_set)]
		users, values = sparse_matrix.shape
		print("[STATUS] [Value Removal] Users: [%d] Values: [%d]" % (users, values), get_ram(), get_elapsed_time(tic))


		# Users to remove
		print("Generating new user data.", get_ram(), get_elapsed_time(tic), end='\r')
		temp = {user: uind for user, uind in user_ind.items() if uind in user_set} # Remove not needed users
		trans_user_dict = {uind: new_uind for new_uind, (_, uind) in enumerate(temp.items())} # Translate from current index to new index
		user_ind = {user: trans_user_dict[uind] for user, uind in temp.items()} # Update user_ind with new indexes
		print("[END] Generating new user data: %d" %(len(user_ind)), get_ram(), get_elapsed_time(tic))
		print("Generating new value data.", get_ram(), get_elapsed_time(tic), end='\r')
		temp = {value: vind for value, vind in value_ind.items() if vind in value_set} # Remove not needed values
		trans_value_dict = {vind: new_vind for new_vind, (_, vind) in enumerate(temp.items())} # Translate from current index to new index
		value_ind = {value: trans_value_dict[vind] for value, vind in temp.items()} # Update the value_ind with new indexes
		print("[END] Generating new value data: %d" % (len(value_ind)), get_ram(), get_elapsed_time(tic))

		dictio_of_users = {trans_user_dict[uind]: [trans_value_dict[vind] for vind in vinds if vind in value_set] for uind, vinds in dictio_of_users.items() if uind in user_set}
		dictio_of_values = {trans_value_dict[vind]: [trans_user_dict[uind] for uind in uinds if uind in user_set] for vind, uinds in dictio_of_values.items() if vind in value_set}

		pickle_object(sparse_matrix, sparse_matrix_p)
		pickle_object(user_ind, user_ind_p)
		pickle_object(value_ind, value_ind_p)
		pickle_object(dictio_of_users, dictio_of_users_p)
		pickle_object(dictio_of_values, dictio_of_values_p)
		return sparse_matrix, user_ind, value_ind, dictio_of_users, dictio_of_values

	"""
	This function generates a vector of the length of the values with the weigth on each entry.
	By default we make use of 255 to fit it ina numpy uint8. It could be changed by another value together with np.uint8 by np.uint32
	"""
	def gen_weight_vector(self, value_ind, dictio_of_values):
		num_elems=255
		num_values = len(value_ind)
		lst_lengths = list(set([(len(v))  for _, v in dictio_of_values.items()]))
		lst_lengths = sorted(lst_lengths, key=lambda x: x, reverse=True)
		if num_elems > len(lst_lengths):
			num_elems = len(lst_lengths)
		divisions = int(math.ceil(float(len(lst_lengths)) / float(num_elems)))
		len_rarity = {}
		weight_vector = np.zeros(num_values, dtype=self.dtype)
		for i in range(0, num_elems):
			start = i * divisions
			end = (i + 1)  * divisions
			for elem in lst_lengths[start:end]:
				len_rarity[elem] = (i + 1)
		for vind, uinds in dictio_of_values.items():
			weight_vector[vind] = len_rarity[len(uinds)]

		return weight_vector

	"""
	This function generates the set of combinations whose score is greater than 1. Thus, all "interesting combinations"
	"""
	def get_promising_combinations(self, dictio_of_users, dictio_of_values):
		tic = time.time()
		set_combinations = set()
		promising_combinations_p = self.dir + "promising_combinations.pkl"
		if self.backup and os.path.exists(promising_combinations_p):
			promising_combinations = unpickle_object(promising_combinations_p)
			return promising_combinations

		num_users = len(dictio_of_users)
		for ind, users in enumerate(dictio_of_values.items()):
			if ind % 1000 == 0:
				print("Promising Combinations Generation", "[%d Users Processed]" %(ind), "[%0.3f Percentage]" % ((ind / num_users) * 100), get_ram(), get_elapsed_time(tic), end='\r')
			users = sorted(users[1])
			combinations = list(itertools.combinations(users, 2))
			for combination in combinations:
				set_combinations.add(combination)
		print("[END] Promising Combinations Generation", "[%d Users Processed]" %(ind), "[%0.3f Percentage]" % ((ind / num_users) * 100), get_ram(), get_elapsed_time(tic), end='\r')
		promising_combinations = list(set_combinations)
		set_combinations = None
		pickle_object(promising_combinations, promising_combinations_p)
		return promising_combinations

	"""
	This function joins all the elements with the same filename. It is used to generate the unique file with the results from the partial ones.
	"""
	def join_all_results(self, origin_filename):
		tic = time.time()
		directory, filename = self.get_filename_dir(origin_filename)
		list_files = [directory + name for name in os.listdir(directory) if filename in name and not filename == name]
		result_file = origin_filename
		f1 = open(result_file, 'w+', buffering=2)
		first = True
		total = len(list_files)
		for ind, file in enumerate(list_files):
			print("Joining files", "[%d Files Processed]" %(ind), "[%0.3f Percentage]" % ((ind / total) * 100), get_ram(), get_elapsed_time(tic), end='\r')
			with open(file, 'r') as f2:
				line = f2.readline()
				if first:
					f1.write(line)
					f1.flush()
					first = False
				line = f2.readline()
				while line:
					f1.write(line)
					f1.flush()
					line = f2.readline()
			os.remove(file)
		f1.close()
		print("[END] Joining files", "[%d Files Processed]" %(ind), "[%0.3f Percentage]" % ((ind / total) * 100), get_ram(), get_elapsed_time(tic), end='\r')



	"""
	This function calculates the uniqueness and coincidence scores for each pair.
	"""
	def calculate_pair_of_users(self, pair):
		u1, u2 = pair[0], pair[1]

		sparse_matrix = self.sparse_matrix
		weight_vector = self.weight_vector
		#print(len(user_indices), bin_sparse_matrix.shape)
		v1 = np.array(sparse_matrix[u1].toarray(), dtype=self.dtype)
		v2 = np.array(sparse_matrix[u2].toarray(), dtype=self.dtype)
		pair_score1 = np.squeeze(np.dot(v1, v2.T))
		pair_score2 = np.squeeze(np.dot(v1 * v2, weight_vector))
		return (u1, u2, pair_score1, pair_score2)

	def calculate_score_for_user(self, user):
		max_users = self.max_users
		sparse_matrix = self.sparse_matrix
		weight_vector = self.weight_vector
		user2_list = [x for x in range(user + 1, max_users)]
		result_list = []
		v1 = np.array(sparse_matrix[user].toarray(), dtype=self.dtype)
		for user2 in user2_list:
			v2 = np.array(sparse_matrix[user2].toarray(), dtype=self.dtype)
			pair_score1 = np.squeeze(np.dot(v1, v2.T))
			pair_score2 = np.squeeze(np.dot(v1 * v2, weight_vector))
			if pair_score1 > 0:
				result_list.append((user, user2, pair_score1, pair_score2))
			
		return result_list
	"""
	This function computes the total of scores in a multiprocessing way
	"""
	def compute_pair_scores(self, sparse_matrix, user_ind, value_ind, promising_combinations, weight_vector):
		filename = self.dir + "results.csv"
		tic = time.time()
		self.sparse_matrix = sparse_matrix
		self.weight_vector = weight_vector

		pool = mp.Pool(self.cores)
		interval_len = 10_000
		intervals = math.ceil(len(lst_users) / interval_len)
		for i in range(intervals):
			toc = time.time()
			targets = lst_users[i * interval_len:(i + 1) * interval_len]
			print("Computation", "[%d Users Processed]" %(i * interval_len), "[%0.3f Percentage]" % ((i / intervals) * 100), get_ram(), get_elapsed_time(tic), end='\r')
			lst_res = pool.map(self.calculate_pair_of_users, targets)
			gen_csv_from_tuples(filename + "_" + str(i) , ["IdAuthor1", "IdAuthor2", "Coincidence", "Uniqueness"], lst_res1)

		pool.close()
		print("[END] Computation", "[%d Users Processed]" %(i * interval_len), "[%0.3f Percentage]" % ((i / intervals) * 100), get_ram(), get_elapsed_time(tic))
		self.join_all_results(filename)

	def compute_scores(self, sparse_matrix, user_ind, value_ind, weight_vector):
		filename = self.dir + "results.csv"
		tic = time.time()
		self.sparse_matrix = sparse_matrix
		self.weight_vector = weight_vector
		self.max_users = len(user_ind)
		inv_user_ind = {v:k for k, v in user_ind.items()}
		pool = mp.Pool(self.cores)
		interval_len = self.cores
		num_users = len(user_ind)
		for x in range(0, num_users, interval_len):
			targets = list(inv_user_ind.keys())[x:x+interval_len]
			print("Computation", "[%d Users Processed]" %(x), "[%0.3f Percentage]" % ((x / num_users) * 100), get_ram(), get_elapsed_time(tic), end='\r')
			lst_res = pool.map(self.calculate_score_for_user, targets)
			lst_res = [(inv_user_ind[x[0]], inv_user_ind[x[1]], x[2], x[3]) for x in lst_res]
			if len(lst_res) > 0:
				gen_csv_from_tuples(filename + "_" + str(uind) , ["IdAuthor1", "IdAuthor2", "Coincidence", "Uniqueness"], lst_res)
			else:
				print("[ERROR] User should have at least one relationship.")
		pool.close()
		print("[END] Computation", "[%d Users Processed]" %(i * interval_len), "[%0.3f Percentage]" % ((i / intervals) * 100), get_ram(), get_elapsed_time(tic))
		self.join_all_results(filename)

	def compute_matrix_mult(self, sparse_matrix):
		tic = time.time()
		sparse_matrix_dot_p = self.dir + 'sparse_matrix_dot_a.pkl'
		if self.backup and os.path.exists(sparse_matrix_dot_p):
			print("Compute Matrix A already exists, unpickling.", end='\r')
			sparse_matrix_dot = unpickle_object(sparse_matrix_dot_p)
			sparse_matrix_dot.eliminate_zeros()
			print("[END] Compute Matrix A already exists, unpickling.")
			return sparse_matrix_dot

		print("Executing coincidence computation over matrix", get_ram(), get_elapsed_time(tic), end='\r')	
		sparse_matrix_dot = sparse_matrix.dot(sparse_matrix.T)
		pickle_object(sparse_matrix_dot, sparse_matrix_dot_p)
		print("[END] Executing coincidence computation over matrix", get_ram(), get_elapsed_time(tic))
		return sparse_matrix_dot

	def compute_matrix_weight_mult(self, sparse_matrix, weight_vector):
		tic = time.time()
		sparse_matrix_dot_p = self.dir + 'sparse_matrix_dot_b.pkl'
		if self.backup and os.path.exists(sparse_matrix_dot_p):
			print("Compute Matrix B already exists, unpickling.", end='\r')
			sparse_matrix_dot = unpickle_object(sparse_matrix_dot_p)
			print("[END] Compute Matrix B already exists, unpickling.")
			return sparse_matrix_dot

		print("Executing uniqueness computation over matrix", get_ram(), get_elapsed_time(tic), end='\r')
		sparse_matrix_b = sparse_matrix.multiply(weight_vector).tocsr()
		sparse_matrix_dot = sparse_matrix.dot(sparse_matrix_b.T)
		sparse_matrix_dot.eliminate_zeros()
		pickle_object(sparse_matrix_dot, sparse_matrix_dot_p)
		print("[END] Executing uniqueness computation over matrix", get_ram(), get_elapsed_time(tic))
		#print(sparse_matrix.shape, sparse_matrix_dot.shape, weight_vector.shape)
		return sparse_matrix_dot

	def extract_info_pairs(self, user_ind, sparse_matrix_dot_a, sparse_matrix_dot_b):
		tic = time.time()
		lst_res = []
		inv_user_ind = {v:k for k, v in user_ind.items()}
		num_users = len(user_ind)

		print("Transforming Matrix A", end='\r')
		sparse_matrix_dot_a = sparse_matrix_dot_a.tocoo()
		print("[END] Transforming Matrix A")

		print("Transforming Matrix B", end='\r')
		sparse_matrix_dot_b = sparse_matrix_dot_b.tocoo()
		print("[END] Transforming Matrix B")

		row_a, col_a, data_a = sparse_matrix_dot_a.row, sparse_matrix_dot_a.col, sparse_matrix_dot_a.data
		data_b = sparse_matrix_dot_b.data
		return row_a, col_a, data_a, data_b
		tx = len(data_b)
		for x, (r, c, d1, d2) in enumerate(zip(row_a, col_a, data_a, data_b)):
			if x % 100_000 == 0:
				print("Info Extraction", "[%d Users Processed]" %(x), "[%d List Length]" %(len(lst_res)), "[%0.3f Percentage]" % ((x / tx) * 100), get_ram(), get_elapsed_time(tic), end='\r')
			lst_res.append((inv_user_ind[r], inv_user_ind[c], d1, d2))

		print("[END] Info Extraction", "[%d Users Processed]" %(x), "[%d List Length]" %(len(lst_res)), "[%0.3f Percentage]" % ((x / tx) * 100), get_ram(), get_elapsed_time(tic), end='\r')
		return lst_res

	def structurize(self):
		tic = time.time()
		print("[>>>] Starting with %s [<<<]" % (self.identifier))
		# Reading list of users
		user_ind, value_ind, dictio_of_users, dictio_of_values = self.gen_data()
		sparse_matrix = self.gen_sparse_matrix( dictio_of_users, len(dictio_of_values))
		sparse_matrix, user_ind, value_ind, dictio_of_users, dictio_of_values = self.clean_matrix(sparse_matrix, user_ind, value_ind, dictio_of_users, dictio_of_values)
		#promising_combinations = self.get_promising_combinations(dictio_of_users, dictio_of_values)
		weight_vector = self.gen_weight_vector(value_ind, dictio_of_values)
		print("[>>>] Ended with %s [<<<]" % (self.identifier), get_ram(), get_elapsed_time(tic))
		return sparse_matrix, user_ind, value_ind, dictio_of_users, dictio_of_values, weight_vector

	def compute(self):
		tic = time.time()
		print("[>>>] Starting with %s [<<<]" % (self.identifier))
		# Reading list of users
		user_ind, value_ind, dictio_of_users, dictio_of_values = self.gen_data()
		sparse_matrix = self.gen_sparse_matrix( dictio_of_users, len(dictio_of_values))
		sparse_matrix, user_ind, value_ind, dictio_of_users, dictio_of_values = self.clean_matrix(sparse_matrix, user_ind, value_ind, dictio_of_users, dictio_of_values)
		#promising_combinations = self.get_promising_combinations(dictio_of_users, dictio_of_values)
		weight_vector = self.gen_weight_vector(value_ind, dictio_of_values)
		print(np.count_nonzero(weight_vector))
		self.backup = False
		sparse_matrix_dot_a = self.compute_matrix_mult(sparse_matrix)
		sparse_matrix_dot_b = self.compute_matrix_weight_mult(sparse_matrix, weight_vector)
		#row_a, col_a, data_a, data_b = self.extract_info_pairs(user_ind, sparse_matrix_dot_a, sparse_matrix_dot_b)
		print("[>>>] Ended with %s [<<<]" % (self.identifier), get_ram(), get_elapsed_time(tic))
		#return row_a, col_a, data_a, data_b


def calculate_tfidf(multfsscore):
	sparse_matrix, user_ind, value_ind, dictio_of_users, dictio_of_values, weight_vector = multfsscore.structurize()

class FeatureScore(object):

	def __init__(self, identifier, dataset_filename, cdf_filename,
		bin_matrix_filename, weight_matrix_filename, 
		coincidence_score_filename, uniqueness_score_filename, 
		coincidence_csv_filename, uniqueness_csv_filename, rarity_bound, pickle_file,
		user_removal=None, value_removal=None):
		self.cores = 56

		if not identifier is None:
			self.filenames=dict()
			self.ident = identifier
			self.filenames['dataset_filename'] = dataset_filename
			self.filenames['cdf_filename'] = cdf_filename
			self.filenames['bin_matrix_filename'] = bin_matrix_filename
			self.filenames['weight_matrix_filename'] = weight_matrix_filename
			self.filenames['coincidence_score_filename'] = coincidence_score_filename
			self.filenames['uniqueness_score_filename'] = uniqueness_score_filename
			self.filenames['coincidence_csv_filename'] = coincidence_csv_filename
			self.filenames['uniqueness_csv_filename'] = uniqueness_csv_filename
			self.filenames['pickle_file'] = pickle_file
			self.filenames['promising_combinations'] = pickle_file[:-4] + "promising_combinations.pkl"
			self.filenames['bin_sparse_matrix'] = pickle_file[:-4] + "bin_sparse_matrix.pkl"
			self.filenames['weight_sparse_matrix'] = pickle_file[:-4] + "weight_sparse_matrix.pkl"
			self.rarity_bound = rarity_bound
			self.user_removal = user_removal
			self.value_removal = value_removal



	def identifier(self):
		return self.ident

	def separate(self, x):
		pos = len(x) - x[::-1].find('[')
		y = x[:pos - 1]
		z = x[pos: - 1]
		return y, z

	def store_clean_dataset(self, dictio_of_users, name = None):
		lst_users = [(key,) + tuple(values) for key, values in dictio_of_users.items()]
		if not name is None:
			gen_csv_from_tuples(name,
				["IdAuthor1", "IdAuthor2", "Score"], lst_users)
		else:
			gen_csv_from_tuples(self.filenames['dataset_filename'][:-4]+"_clean.csv",
				["IdAuthor1", "IdAuthor2", "Score"], lst_users)

	### This function generates a dictionary of the users associated to the values used by that user.
	def gen_dictio_of_users(self,lst):
		dictio = {}
		for i in lst:
			key = i[0]
			dictio[key] = []
			for j in i[1:]:
				value, usage = self.separate(j)
				dictio[key].append(value)
		return dictio

	### This function generates a dictionary of the values associated to the user.
	def gen_dictio_of_values(self,lst):
		dictio = {}
		for i in lst:
			key = i[0]
			for j in i[1:]:
				value, usage = self.separate(j)
				if value in dictio.keys():
					dictio[value].append(key)
				else:
					dictio[value] = [key]
		return dictio

	def get_upper_lower_bounds(self, dictionary, c=2.0):
		lst = [len(v) for k, v in dictionary.items()]
		v = np.array(lst)
		mean = np.mean(v)
		std = np.std(v)
		upper_bound = math.ceil(mean + (c * std))
		lower_bound = math.ceil(max((mean - (c * std)), 0))
		#upper_bound = math.ceil(mean + (c * std))
		#lower_bound = math.ceil(max((mean - (c * std)), 0))
		return upper_bound, lower_bound


	def get_mean_std(self, dictionary, c=1):
		lst = [len(v) for k, v in dictionary.items()]
		v = np.array(lst)
		mean = np.mean(v)
		std = np.std(v)
		return mean, std

	def remove_values_from_dictios(self, list_value, dictio_of_users, dictio_of_values):
		if len(list_value) == 0:
			return dictio_of_users, dictio_of_values
		elif len(list_value) == len(dictio_of_values):
			return {}, {}

		old_len_u, old_len_value = len(dictio_of_users), len(dictio_of_values)
		#print("values removed: ", len(list_value) )
		for value in list_value:
			for user in dictio_of_values[value]:
				dictio_of_users[user].remove(value)
			ret2 = dictio_of_values.pop(value, None)
			if ret2 is None:
				print("THERE IS AN ERROR: ", value)
		# Update the list of users
		users_removed = [k for k, v in dictio_of_users.items() if len(v) == 0]
		# Remove users which have no value.
		for user in users_removed:
			ret = dictio_of_users.pop(user, None)
			if ret is None:
				print("ERROR")
		row_format ="{:>15}" * 4
		print("-" * 15 * 4)
		print(row_format.format("Original values", "Removed values", "New values", "Percentage"))
		print(row_format.format("%d"% (old_len_value), "%d"%(len(list_value)), "%d"%(len(dictio_of_values)), "%f" %(len(dictio_of_values)/old_len_value)))
		print(row_format.format("Original Users", "Removed Users", "New Users", "Percentage"))
		print(row_format.format("%d"% (old_len_u), "%d"%(len(users_removed)), "%d"%(len(dictio_of_users)), "%f" %(len(dictio_of_users)/old_len_u)))
		print("-" * 15 * 4)
		return dictio_of_users, dictio_of_values
	
	### Removes users from the interchangeable dictionaries.
	def remove_users_from_dictios(self, list_users, dictio_of_users, dictio_of_values):
		if len(list_users) == 0:
			return dictio_of_users, dictio_of_values
		elif len(list_users) == len(dictio_of_values):
			return {}, {}

		old_len_u, old_len_value = len(dictio_of_users), len(dictio_of_values)
		# Remove users and values from the list 
		for user in list_users:
			for value in dictio_of_users[user]:
				dictio_of_values[value].remove(user)
			ret2 = dictio_of_users.pop(user, None)
			

		# Update the list of users
		values_removed = [k for k, v in dictio_of_values.items() if v == []]
		
		# Remove values which are no longer used by a user.
		for value in values_removed:
			ret = dictio_of_values.pop(value, None)
			if ret is None:
				print("ERROR")
		row_format ="{:>15}" * 4
		print("-" * 15 * 4)
		print(row_format.format("Original values", "Removed values", "New values", "Percentage"))
		print(row_format.format("%d"% (old_len_value), "%d"%(len(values_removed)), "%d"%(len(dictio_of_values)), "%f" %(len(dictio_of_values)/old_len_value)))
		print(row_format.format("Original Users", "Removed Users", "New Users", "Percentage"))
		print(row_format.format("%d"% (old_len_u), "%d"%(len(list_users)), "%d"%(len(dictio_of_users)), "%f" %(len(dictio_of_users)/old_len_u)))
		print("-" * 15 * 4)
		#print(len(dictio_of_users)/old_len_u, len(dictio_of_values)/old_len_value, 
			#len(dictio_of_users), len(dictio_of_values))
		return dictio_of_users, dictio_of_values

	def clean_dataset(self, dictio_of_users, dictio_of_values, user_removal=None, value_removal=None):
		# Remove values with 1 appearance.
		# print("[-] Getting user upper and lower bound...")
		# user_upper_bound, user_lower_bound = self.get_upper_lower_bounds(dictio_of_users, c=0.5)
		# user_mean, user_std = self.get_mean_std(dictio_of_users)
		# print("[-] Getting value upper and lower bound...")
		# value_upper_bound, value_lower_bound = self.get_upper_lower_bounds(dictio_of_values, c=0.5)
		# value_mean, value_std = self.get_mean_std(dictio_of_values)
		print("[-] Removing values that appear once...")
		oneapp =  [k for k,v in dictio_of_values.items() if len(v) == 1]
		dictio_of_users, dictio_of_values = self.remove_values_from_dictios(oneapp, dictio_of_users, dictio_of_values)
		
		print("User Removal:", user_removal)
		print("Value Removal: ", value_removal)
		# We execute all user removal procedures specified
		if not user_removal is None:
			for procedure in user_removal:
				user_list = procedure(dictio_of_users, dictio_of_values)
				dictio_of_users, dictio_of_values = self.remove_users_from_dictios(user_list, dictio_of_users, dictio_of_values)
		# We execute all value removal procedures specified by the user
		if not value_removal is None:
			for procedure in value_removal:
				value_list = procedure(dictio_of_users, dictio_of_values)
				dictio_of_users, dictio_of_values = self.remove_values_from_dictios(value_list, dictio_of_users, dictio_of_values)
		self.store_clean_dataset(dictio_of_users)
		# print("[-] Removing values that appear more than %d times or less than %d times" % (value_upper_bound, value_lower_bound))
		# multivalue = [k for k,v in dictio_of_values.items() if len(v) > value_upper_bound or len(v) < value_lower_bound]
		# dictio_of_users, dictio_of_values = self.remove_values_from_dictios(multivalue, dictio_of_users, dictio_of_values)
		# print("[-] Removing users that have less than %d values or more than %d values" % (user_lower_bound, user_upper_bound))
		# multiuser = [k for k,v in dictio_of_users.items() if len(v) > user_upper_bound or len(v) < user_lower_bound]
		# dictio_of_users, dictio_of_values = self.remove_users_from_dictios(multiuser, dictio_of_users, dictio_of_values)
		# oneapp = [k for k,v in dictio_of_values.items() if len(v) == 1]
		# print("[-] Removing final single values that appear once...")
		# dictio_of_users, dictio_of_values = self.remove_values_from_dictios(oneapp, dictio_of_users, dictio_of_values)
		return dictio_of_users, dictio_of_values
	
	def clean_dataset_2(self, dictio_of_users, dictio_of_values, dictio_of_weights, rarity_bound):
		print("[-] Removing values with rarity lower than %d" % (rarity_bound))
		lst_remove = [value for value, rarity in dictio_of_weights.items() if rarity < rarity_bound]
		print("DICTIO OF USERS [%d] DICTIO OF VALUES [%d]" %(len(dictio_of_users), len(dictio_of_values)),len(lst_remove))
		dictio_of_users, dictio_of_values = self.remove_values_from_dictios(lst_remove, dictio_of_users, dictio_of_values)
		#print("[-] Removing values that appear once...")
		#oneapp = [k for k,v in dictio_of_values.items() if len(v) == 1]	
		#dictio_of_users, dictio_of_values = self.remove_values_from_dictios(oneapp, dictio_of_users, dictio_of_values)
		return dictio_of_users, dictio_of_values

	def clean_dataset_optimal_red(self, dictio_of_users, dictio_of_values, dictio_of_weights, rarity_bound):
		print("[-] Removing values with rarity lower than %d" % (rarity_bound))
		lst_remove = [value for value, rarity in dictio_of_weights.items() if rarity < rarity_bound]
		dictio_of_users, dictio_of_values = self.remove_values_from_dictios(lst_remove, dictio_of_users, dictio_of_values)
		for value in lst_remove:
			dictio_of_weights.pop(value, None)
		#print("[-] Removing values that appear once...")
		#oneapp = [k for k,v in dictio_of_values.items() if len(v) == 1]	
		#dictio_of_users, dictio_of_values = self.remove_values_from_dictios(oneapp, dictio_of_users, dictio_of_values)
		return dictio_of_users, dictio_of_values, dictio_of_weights
	
	### This function generates a dictionary of the values associated to the total number of times it was used.
	def gen_special_dictio_of_usage(self,lst):
		dictio = {}
		tic = time.time()
		pool = mp.Pool(self.cores)
		dictio = {}
		from defs import whatever
		pool.map(whatever, lst)
		pool.close()
		# for ind, i in enumerate(lst):
		# 	key = i[0]
		# 	if ind % 1000 == 0:
		# 		print("[%d Users Processed]"%(ind), "[%d Values]" % (len(dictio)), "[%0.3f Percentage]" % ((ind / len(lst)) * 100), get_ram(), get_elapsed_time(tic))
		# 	for j in i[1:]:
		# 		value, usage = self.separate(j)
		# 		usage = int(usage)
		# 		if value in dictio.keys():
		# 			dictio[value] += usage
		# 		else:
		# 			dictio[value] = usage
		return dictio
	
	### This function is going to remove the most generic values.
	def special_removal(self, lst_users):
		dictio_of_special_usage = self.gen_special_dictio_of_usage(lst_users)
		

	#Generates a clean dataset from the filename.
	def generate_clean_dataset(self, filename, do_rarity=True):
		tic = time.time()
		lst_users = read_csv_list(filename)[1:]
		
		dictio_of_users = self.gen_dictio_of_users(lst_users)
		dictio_of_values = self.gen_dictio_of_values(lst_users)
		print("USERS %d, VALUES: %d" % (len(dictio_of_users), len(dictio_of_values)))
		if not "_clean" in filename:
			dictio_of_users, dictio_of_values = self.clean_dataset(dictio_of_users,dictio_of_values,
				user_removal=self.user_removal, value_removal=self.value_removal)
		

			if do_rarity:
				dictio_of_weights = self.gen_dictio_of_weigths (dictio_of_values)
				#dictio_of_users, dictio_of_values = self.clean_dataset_2(dictio_of_users, dictio_of_values, dictio_of_weights, self.rarity_bound)

		print("DICTIO OF USERS [%d] DICTIO OF VALUES [%d]" %(len(dictio_of_users), len(dictio_of_values)))
		dictio_of_weights = self.gen_dictio_of_weigths (dictio_of_values)
		#dictio_of_usage = self.gen_dictio_of_usage(dictio_of_users, dictio_of_values, lst_users)
		print(len(dictio_of_weights), len(dictio_of_values))
		print("[+] Finished Dataset Cleanup in: %f" %(time.time() - tic))
		return dictio_of_users, dictio_of_values, dictio_of_weights#, dictio_of_usage

	def gen_dictio_of_usage(self, dictio_of_users, dictio_of_values, lst):
		print("[-] Generating dictionary with usage...")
		dictio_of_usage = dict()
		for i in lst:
			user = i[0]
			if user in dictio_of_users:
				dictio_of_usage[user] = dict()
				for j in  i[1:]:
					value, usage = self.separate(j)
					if value in dictio_of_values:
						dictio_of_usage[user][value] = int(usage)
		print("[+] Finished generating dictionary with usage...")
		return dictio_of_usage


 	# Generates weights of the values depending on its rarity in the dataset
	def gen_dictio_of_weigths(self, dictio_of_values, num_elems=255):
		dictio_of_values = dict(dictio_of_values)
		lst_lengths = list(set([(len(v))  for k, v in dictio_of_values.items()]))
		lst_lengths = sorted(lst_lengths, key=lambda x: x, reverse=True)
		#print(num_elems, len(lst_lengths))
		print(len(dictio_of_values), num_elems, len(lst_lengths))
		if num_elems > len(lst_lengths):
			num_elems = len(lst_lengths)
		#print(num_elems)
		print(len(dictio_of_values), num_elems, len(lst_lengths))
		divisions = int(math.ceil(float(len(lst_lengths)) / float(num_elems)))
		len_rarity = {}
		for i in range(0, num_elems):
			start = i * divisions
			end = (i + 1)  * divisions
			for elem in lst_lengths[start:end]:
				len_rarity[elem] = (i + 1)

		#print(len_rarity)
		# We modify the dictio_of_values and return according to the rarity
		for key, values in dictio_of_values.items():
			dictio_of_values[key] = len_rarity[len(values)]

		return dictio_of_values

	def order_users(self, entry):
		if entry[0] > entry[1]:
			return entry[1], entry[0]
		else:
			return entry[0], entry[1]

	def get_promising_combinations(self, dictio_of_users, dictio_of_values):
		print("[-] Extracting promising combinations...")
		set_combinations = set()
		length = len(dictio_of_values.items())
		status.create_numbar(100, length)
		for ind, users in enumerate(dictio_of_values.items()):
			status.update_numbar(ind, length)
			users = sorted(users[1])
			combinations = list(itertools.combinations(users, 2))
			for combination in combinations:
				set_combinations.add(combination)
		status.end_numbar()
		print("[+] Extracted %d promising combinations" % (len(set_combinations)))
		return list(set_combinations)


	def gen_rarity_dist(self, dictio):
		rarity_num_dictio = {}
		for value, rarity in dictio.items():
			if rarity in rarity_num_dictio.keys():
				rarity_num_dictio[rarity] += 1
			else:
				rarity_num_dictio[rarity] = 1

		lst_dist = [(rarity,repetitions) for rarity,repetitions in rarity_num_dictio.items()]
		lst_dist = sorted(lst_dist, key=lambda x: x[0], reverse=False)
		return lst_dist

	def get_cdf(self, filename, graph_filename):
		tic = time.time()
		lst_users = read_csv_list(filename)[1:]
		dictio_of_users = self.gen_dictio_of_users(lst_users)
		dictio_of_values = self.gen_dictio_of_values(lst_users)
		dictio_of_users, dictio_of_values = self.clean_dataset(dictio_of_users,dictio_of_values,
			user_removal=self.user_removal, value_removal=self.value_removal)

	def generate_cdf(self, filename, graph_filename):
		tic = time.time()
		lst_users = read_csv_list(filename)[1:]
		
		dictio_of_users = self.gen_dictio_of_users(lst_users)
		dictio_of_values = self.gen_dictio_of_values(lst_users)
		dictio_of_users, dictio_of_values = self.clean_dataset(dictio_of_users,dictio_of_values,
			user_removal=self.user_removal, value_removal=self.value_removal)

		self.store_clean_dataset(dictio_of_users)
		
		dictio_of_weights = self.gen_dictio_of_weigths(dictio_of_values)
		#print(dictio_of_weights)
		lst_rarity = self.gen_rarity_dist(dictio_of_weights)
		#print(lst_rarity)
		#print(len(dictio_of_weights), len(dictio_of_values))

		X = np.array([float(x[0]) for x in lst_rarity])
		Y = np.array([float(x[1]) for x in lst_rarity])
		Y /= np.sum(Y)
		#print(X, Y)
		CY = np.cumsum(Y)
		percentage = 0.1
		plt.style.use('seaborn-darkgrid')
		palette = plt.get_cmap('Set1')
		for x,y,cy in zip(X,Y,CY):
			if y > percentage:
				plt.scatter(x, cy, color=palette(2), label="x(Y > %f) = %d" % (percentage, x))
		for x,y,cy in zip(X,Y,CY):
			if cy > percentage:
				plt.axvline(x=x, color=palette(2), linestyle='-.', label="x(CDF > %f) = %d" % (percentage, x))
				break
		plt.plot(X, Y, color=palette(0), label="Y")
		plt.plot(X, CY, color=palette(1), linestyle='--', label="CDF(Y)") 
			#marker='o', markersize=3, markerfacecolor=palette(0))
		plt.xlabel('Rarity')
		plt.ylabel('Percentage of this rarity (%)')

				#plt.scatter('', xy=(x, cy), xytext=(0, 0), color='red' , textcoords='offset points') 
		plt.legend()
		plt.savefig(graph_filename, format="PNG")
		plt.clf()
		print("[+] Finished Graph Generation Cleanup in: %f" %(time.time() - tic))

	def cdf(self):
		self.generate_cdf(self.filenames['dataset_filename'], self.filenames['cdf_filename'])

	def gen_binary_matrix_mem(self, dictio_of_values, dictio_of_users):
		tic = time.time()
		dictio_of_values = dict(dictio_of_values)
		num_users = len(dictio_of_users)
		num_values = len(dictio_of_values)
		# Transform dictionary to indexes
		for indk, value in enumerate(dictio_of_values.keys()):
			dictio_of_values[value] = indk
		#Transform users to matrices.
		print("[-] ESTIMATED SIZE OF BINARY MATRIX: %f GB" % (num_users * num_values * 1 / (1024 ** 3)))
		matrix_map = np.zeros(shape=(num_users, num_values), dtype=np.uint8)
		status.create_numbar(100, num_users)
		for ind, user in enumerate(dictio_of_users.keys()):
			status.update_numbar(ind, num_users)
			base = np.zeros((num_values,1), dtype=np.uint8)
			for value in dictio_of_users[user]:
				base[dictio_of_values[value]] = 1
			base = np.squeeze(base)
			matrix_map[ind] = base[:]
		status.end_numbar()
		print("[+] Finished Binary Matrix Generation in: %f" %(time.time() - tic))
		return matrix_map

	def gen_binary_matrix(self, dictio_of_values, dictio_of_users, matrix_file):
		tic = time.time()
		dictio_of_values = dict(dictio_of_values)
		num_users = len(dictio_of_users)
		num_values = len(dictio_of_values)
		# Transform dictionary to indexes
		for indk, value in enumerate(dictio_of_values.keys()):
			dictio_of_values[value] = indk
		#Transform users to matrices.
		print("[-] ESTIMATED SIZE OF BINARY MATRIX: %f GB" % (num_users * num_values * 1 / (1024 ** 3)))
		matrix_map = np.memmap(matrix_file, dtype=np.uint8, mode ='w+', shape=(num_users, num_values))
		status.create_numbar(100, num_users)
		for ind, user in enumerate(dictio_of_users.keys()):
			status.update_numbar(ind, num_users)
			base = np.zeros((num_values,1), dtype=np.uint8)
			for value in dictio_of_users[user]:
				base[dictio_of_values[value]] = 1 #* dictio_of_usage[user][value]
			base = np.squeeze(base)
			#base = base / base.sum()
			matrix_map[ind] = base[:]
		status.end_numbar()
		print("[-] Flushing binary matrix to memory")
		matrix_map.flush()
		print("[+] Finished Binary Matrix Generation in: %f" %(time.time() - tic))
	
	def gen_binary_sparse_matrix(self, dictio_of_values, dictio_of_users):
		print("[-] Generating Binary Sparse Matrix..")
		tic = time.time()
		dictio_of_values = dict(dictio_of_values)
		num_users = len(dictio_of_users)
		num_values = len(dictio_of_values)
		value_indices = {k:indk for indk, k in enumerate(dictio_of_values.keys())}
		user_indices = {k:indk for indk, k in enumerate(dictio_of_users.keys())}
		rows = []
		cols = []
		data = []

		status.create_numbar(100, num_users)
		for ind, row in enumerate(dictio_of_users.items()):
			status.update_numbar(ind, num_users)
			user, values = row[0], row[1]
			#base = np.zeros((num_values,1), dtype=np.uint8)
			#for value in dictio_of_users[user]:
				#base[value_indices[value]] = 1 #dictio_of_usage[user][value]
			#base = np.squeeze(base)
			#base = base / base.sum()
			for indj, value in enumerate(values):
				rows.append(user_indices[user])
				cols.append(value_indices[value])
				data.append(1)

		status.end_numbar()
		matrix = csr_matrix((data, (rows, cols)), shape=(num_users, num_values), dtype=np.uint8)
		print("[+] Finished Binary Sparse Matrix Generation in: %f" %(time.time() - tic))
		return matrix

	def gen_weight_matrix_mem(self, dictio_of_values, dictio_of_users,  dictio_of_weights):
		tic = time.time()
		num_users = len(dictio_of_users)
		num_values = len(dictio_of_values)
		dictio_of_values = dict(dictio_of_values)
		# Transform dictionary to indexes
		for indk, value in enumerate(dictio_of_values.keys()):
			dictio_of_values[value] = indk
		#Transform users to matrices.
		print("[-] ESTIMATED SIZE OF WEIGHTS MATRIX: %f GB" % (num_users * num_values * 1 / (1024 ** 3)))
		matrix_map = np.zeros(shape=(num_users, num_values), dtype=np.uint8)
		status.create_numbar(100, num_users)
		for ind, user in enumerate(dictio_of_users.keys()):
			status.update_numbar(ind, num_users)
			base = np.zeros((num_values,1), dtype=np.uint8)
			for value in dictio_of_users[user]:
				base[dictio_of_values[value]] = dictio_of_weights[value] 
			base = np.squeeze(base)
			matrix_map[ind] = base[:]
		status.end_numbar()
		print("[+] Finished Weight Matrix Generation in: %f" %(time.time() - tic))
		return matrix_map

	def gen_weight_matrix(self, dictio_of_values, dictio_of_users, dictio_of_weights, matrix_file):
		tic = time.time()
		num_users = len(dictio_of_users)
		num_values = len(dictio_of_values)
		dictio_of_values = dict(dictio_of_values)
		# Transform dictionary to indexes
		for indk, value in enumerate(dictio_of_values.keys()):
			dictio_of_values[value] = indk
		#Transform users to matrices.
		print("[-] ESTIMATED SIZE OF WEIGHTS MATRIX: %f GB" % (num_users * num_values * 1 / (1024 ** 3)))
		matrix_map = np.memmap(matrix_file, dtype=np.uint8, mode ='w+', shape=(num_users, num_values))
		status.create_numbar(100, num_users)
		for ind, user in enumerate(dictio_of_users.keys()):
			status.update_numbar(ind, num_users)
			base = np.zeros((num_values,1), dtype=np.uint8)
			for value in dictio_of_users[user]:
				base[dictio_of_values[value]] = dictio_of_weights[value] #dictio_of_usage[user][value]
			#base = np.squeeze(base)
			#base = base / base.sum()
			#for value in dictio_of_users[user]:
				#base[dictio_of_values[value]] *= dictio_of_weights[value] 
			base = np.squeeze(base)
			matrix_map[ind] = base[:]
		status.end_numbar()
		print("[-] Flushing weights matrix to memory")
		matrix_map.flush()
		print("[+] Finished Weight Matrix Generation in: %f" %(time.time() - tic))

	def gen_weight_sparse_matrix(self, dictio_of_values, dictio_of_users, dictio_of_weights):
		print("[-] Generating Weight Sparse Matrix..")
		tic = time.time()
		dictio_of_values = dict(dictio_of_values)
		num_users = len(dictio_of_users)
		num_values = len(dictio_of_values)
		value_indices = {k:indk for indk, k in enumerate(dictio_of_values.keys())}
		user_indices = {k:indk for indk, k in enumerate(dictio_of_users.keys())}
		rows = []
		cols = []
		data = []

		status.create_numbar(100, num_users)
		for ind, row in enumerate(dictio_of_users.items()):
			status.update_numbar(ind, num_users)
			user, values = row[0], row[1]

			#base = np.zeros((num_values,1), dtype=np.uint8)
			#for value in dictio_of_users[user]:
				#base[value_indices[value]] = dictio_of_weights[value] #* dictio_of_usage[user][value]
			#base = np.squeeze(base)
			#base = base / base.sum()
			
			for indj, value in enumerate(values):
				rows.append(user_indices[user])
				cols.append(value_indices[value])
				data.append(dictio_of_weights[value])


		status.end_numbar()
		matrix = csr_matrix((data, (rows, cols)), shape=(num_users, num_values), dtype=np.uint8)
		print("[+] Finished Weight Sparse Matrix Generation in: %f" %(time.time() - tic))
		return matrix



	def process_score(self, dictio_of_users, dictio_of_values, matrix_filename, score_filename):
		tic = time.time()
		num_users = len(dictio_of_users)
		num_values = len(dictio_of_values)

		matrix_map = np.memmap(matrix_filename, dtype=np.uint8,  shape=(num_users, num_values))
		#matrix_map = np.array(matrix_map)

		print(matrix_map.shape)
		print("[-] ESTIMATED SIZE OF SCORE MATRIX: %f GB" % (num_users * num_users * 4 / (1024 ** 3)))
		status.create_numbar(100, num_users)
		#res = np.dot(matrix_map, matrix_map.T)
		#res2 = np.memmap('value_files/value_euc_score.dat', dtype=float ,mode ='w+', shape=(num_users, num_users))
		res3 = np.memmap(score_filename, dtype=np.uint32 ,mode ='w+', shape=(num_users, num_users))
		for i1 in range(num_users):
			status.update_numbar(i1, num_users)
			v1 = np.array(matrix_map[i1], dtype=np.uint32)
			#v1p = gen_vector_for_user(list(dictio_of_users.keys())[i1], dictio_of_users, dictio_of_values)
			for i2 in range(i1 + 1, num_users):
				v2 = np.array(matrix_map[i2], dtype=np.uint32)
				#print(v1)
				#euc_score = np.linalg.norm(v1-v2)
				dis_score = np.dot(v1, v2)
				#res2[i1][i2] = euc_score
				res3[i1][i2] = dis_score
				#res2[i2][i1] = score
		status.end_numbar()
		print("[-] Flushing uniqueness score matrix to memory")
		res3.flush()
		
		print("[+] Finished score calculation in: %f SECONDS" %(time.time() - tic))

	def process_pairs_of_users(self, dictio_of_users, dictio_of_values, dictio_of_weights, lst_users, matrix, csv_filename):
		tic = time.time()
		if csv_filename is None:
			print("[WARN] No file provided, returning matrix")	
		dictio_of_users = dict(dictio_of_users)
		for indk, user in enumerate(dictio_of_users.keys()):
			dictio_of_users[user] = indk
		num_users = len(dictio_of_users)
		num_values = len(dictio_of_values)
		num_lst_users = len(lst_users)
		value_indices = {k:indk for indk, k in enumerate(dictio_of_values.keys())}
		rarity = np.zeros((num_values,1), dtype=np.uint8)
		for value in dictio_of_values.keys():
			i = value_indices[value]
			rarity[i] = dictio_of_weights[value]

		matrix_map = None


		if type(matrix) == str:
			matrix_map = np.memmap(matrix, dtype=np.uint8,  shape=(num_users, num_values))
		else:
			matrix_map = matrix
		#matrix_map = np.array(matrix_map)

		## ##print(matrix_map.shape)
		print("[-] ESTIMATED SIZE OF SCORE MATRIX: %f GB" % (num_users * num_users * 4 / (1024 ** 3)))
		status.create_numbar(100, num_lst_users)
		#res = np.dot(matrix_map, matrix_map.T)
		#res2 = np.memmap('value_files/value_euc_score.dat', dtype=float ,mode ='w+', shape=(num_users, num_users))
		#res3 = np.memmap(coincidence_score_filename, dtype=np.uint32 ,mode ='w+', shape=(num_users, num_users))
		lst_res = list()
		for ind, pair in enumerate(lst_users):
			status.update_numbar(ind, num_lst_users)
			user1, user2 = pair[0], pair[1]
			v1 = np.array(matrix_map[dictio_of_users[user1]], dtype=np.uint8)
			v2 = np.array(matrix_map[dictio_of_users[user2]], dtype=np.uint8)
			pair_score = np.dot(v1, v2)
			pair_score2 = np.squeeze(np.dot(v1 * v2, rarity))
			#pair_score3 = np.linalg.norm(v1-v2)
			#pair_score4 = np.linalg.norm(v1-v2, ord=1)		
			#lst_res.append((user1, user2, pair_score))
			if pair_score > 0:
				lst_res.append((user1, user2, pair_score, pair_score2))
			#lst_res.append((user1, user2, pair_score, pair_score2))
		status.end_numbar()
		lst_res = sorted(lst_res, key=lambda x: x[2], reverse=True)
		print("[+] Finished score calculation in: %f SECONDS" %(time.time() - tic))
		if csv_filename is None:
			return lst_res
		else:
			#lst_res = sorted(lst_res, key=lambda x: x[6], reverse=True)
			gen_csv_from_tuples(csv_filename , ["IdAuthor1", "IdAuthor2", "Score"], lst_res)
		


	def gen_vector_for_user(self, user_values, dictio_of_values, dictio_of_weights=None):
		num_values = len(dictio_of_values)
		base = np.zeros((num_values,1), dtype=np.uint32)
		if dictio_of_weights is None:
			for value in user_values:
				base[dictio_of_values[value]] = 1
		else:
			for value in user_values:
				base[dictio_of_values[value]] = dictio_of_weights[value]
		base = np.squeeze(base)
		return base


	def process_pairs_of_users_sparse_matrix(self, dictio_of_users, dictio_of_values, dictio_of_weights,
		lst_users, sparse_matrix, csv_filename):
		tic = time.time()	
		dictio_of_users = dict(dictio_of_users)
		value_indices = {k:indk for indk, k in enumerate(dictio_of_values.keys())}
		user_indices = {k:indk for indk, k in enumerate(dictio_of_users.keys())}
		num_users = len(dictio_of_users)
		num_values = len(dictio_of_values)
		num_lst_users = len(lst_users)

		rarity = np.zeros((num_values,1), dtype=np.uint8)
		for value in dictio_of_values.keys():
			i = value_indices[value]
			rarity[i] = dictio_of_weights[value]

		print("[-] ESTIMATED SIZE OF SCORE MATRIX: %f GB" % (num_users * num_users * 4 / (1024 ** 3)))
		status.create_numbar(100, num_lst_users)
		#res = np.dot(matrix_map, matrix_map.T)
		#res2 = np.memmap('value_files/value_euc_score.dat', dtype=float ,mode ='w+', shape=(num_users, num_users))
		#res3 = np.memmap(coincidence_score_filename, dtype=np.uint32 ,mode ='w+', shape=(num_users, num_users))
		lst_res = list()
		for ind, pair in enumerate(lst_users):
			status.update_numbar(ind, num_lst_users)
			user1, user2 = pair[0], pair[1]
			v1 = np.array(sparse_matrix[user_indices[user1]].toarray(), dtype=np.uint32)
			v2 = np.array(sparse_matrix[user_indices[user2]].toarray(), dtype=np.uint32)
			pair_score = np.squeeze(np.dot(v1, v2.T))
			pair_score2 = np.squeeze(np.dot(v1 * v2, rarity))
			if pair_score > 0:
				lst_res.append((user1, user2, pair_score, pair_score2)) 
		status.end_numbar()
		sortedl = sorted(lst_res, key=lambda x: str(x[2]) + x[0] + x[1], reverse=True)
		gen_csv_from_tuples(csv_filename , ["IdAuthor1", "IdAuthor2", "Score"], sortedl)
		print("[+] Finished score calculation in: %f SECONDS" %(time.time() - tic))

	def generate_processing_args(self, dictio_of_users, dictio_of_values, dictio_of_weights, 
		lst_users, bin_sparse_matrix, weight_sparse_matrix, csv_filename_bin, csv_filename_wei):
		self.processing_args = dict()
		self.processing_args['bin_sparse_matrix'] = bin_sparse_matrix
		self.processing_args['weight_sparse_matrix'] = weight_sparse_matrix

		value_indices = {k:indk for indk, k in enumerate(dictio_of_values.keys())}
		num_values = len(value_indices)
		rarity = np.zeros((num_values,1), dtype=np.uint8)
		for value in dictio_of_values.keys():
			i = value_indices[value]
			rarity[i] = dictio_of_weights[value]
		self.processing_args['rarity_vector'] = rarity
		#self.processing_args['lst_users'] = lst_users
		#self.processing_args['dictio_of_values'] = dictio_of_values
		#self.processing_args['dictio_of_users'] = dict(dictio_of_users)
		#self.processing_args['value_indices'] = {k:indk for indk, k in enumerate(dictio_of_values.keys())}
		self.processing_args['user_indices'] = {k:indk for indk, k in enumerate(dictio_of_users.keys())}
		#self.processing_args['num_users'] = len(dictio_of_users)
		#self.processing_args['num_values'] = len(dictio_of_values)
		#self.processing_args['num_lst_users'] = len(lst_users)

	def calculate_pair_of_users(self, pair):
		user1, user2 = pair[0], pair[1]

		bin_sparse_matrix = self.processing_args['bin_sparse_matrix']
		user_indices = self.processing_args['user_indices']
		rarity = self.processing_args['rarity_vector']
		#print(len(user_indices), bin_sparse_matrix.shape)
		u1 = user_indices[user1]
		u2 = user_indices[user2]
		v1 = np.array(bin_sparse_matrix[u1].toarray(), dtype=np.uint8)
		v2 = np.array(bin_sparse_matrix[u2].toarray(), dtype=np.uint8)
		pair_score1 = np.squeeze(np.dot(v1, v2.T))
		pair_score2 = np.squeeze(np.dot(v1 * v2, rarity))
		return (pair_score1, pair_score2)
		#return (0, pair_score2)
	def process_pairs_of_users_sparse_matrix_alt(self, dictio_of_users, dictio_of_values, dictio_of_weights,
		lst_users, bin_sparse_matrix, weight_sparse_matrix, csv_filename_bin, csv_filename_wei):

		tic = time.time()
		self.generate_processing_args(dictio_of_users, dictio_of_values, dictio_of_weights,
			lst_users, bin_sparse_matrix, weight_sparse_matrix, csv_filename_bin, csv_filename_wei)
		print("AHHHASASDF", self.processing_args['bin_sparse_matrix'].shape, len(self.processing_args['user_indices']))
		pool = mp.Pool(self.cores)
		print("[-] Started processing selected users")
		interval_len = 100000
		intervals = math.ceil(len(lst_users) / interval_len)
		for i in range(intervals):
			toc = time.time()
			targets = lst_users[i * interval_len:(i + 1) * interval_len]
			print("[- -]Going for interval %d of %d" % (i + 1, intervals))
			print(i, intervals, i * interval_len, (i+1) * interval_len)
			lst_res = pool.map(self.calculate_pair_of_users, targets)
			lst_res1 = [(pair[0], pair[1], elem[0], elem[1]) for pair, elem in zip(targets, lst_res)]
			gen_csv_from_tuples(csv_filename_bin + "_" + str(i) , ["IdAuthor1", "IdAuthor2", "Score"], lst_res1)
			#lst_res1 = [(pair[0], pair[1], elem[1]) for pair, elem in zip(targets, lst_res)]
			#gen_csv_from_tuples(csv_filename_wei + "_" + str(i) , ["IdAuthor1", "IdAuthor2", "Score"], lst_res1)
			#print("[+ +]Finished for interval %d of %d in %d seconds" % (i + 1, intervals, time.time() - toc))

		pool.close()
		#lst_res = [(pair[0], pair[1], elem) for pair, elem in zip(lst_users, lst_res)]
		#print("[+] Finished processing selected users")
		#sortedl = sorted(lst_res, key=lambda x: str(x[2]) + x[0] + x[1], reverse=True)
		#gen_csv_from_tuples(csv_filename , ["IdAuthor1", "IdAuthor2", "Score"], sortedl)
		print("[+] Finished score calculation in: %f SECONDS" %(time.time() - tic))


	def gen_score_csv(self, dictio_of_users, dictio_of_values, score_filename, csv_filename):
		tic = time.time()
		print("[-] Generating scores csv...")
		num_users = len(dictio_of_users)
		lst_users = list(dictio_of_users.keys())
		lst = []
		res2 = np.memmap(score_filename, dtype=np.uint32, mode ='r', shape=(num_users, num_users))
		status.create_numbar(100, num_users)
		for i in range(num_users):
			status.update_numbar(i, num_users)
			for j in range(i + 1, num_users):
				if res2[i][j] > 0:
					lst.append((lst_users[i],lst_users[j],res2[i][j]))
		status.end_numbar()
		#print("Obtained all scores: %f" %(time.time() - tic))
		sortedl = sorted(lst, key=lambda x: str(x[2]) + x[0] + x[1], reverse=True)
		gen_csv_from_tuples(csv_filename , ["IdAuthor1", "IdAuthor2", "Score"], sortedl)
		print("[+] Finished score generation in: %f SECONDS" %(time.time() - tic))

	def pickle_object(self, obj, filename):
		with open(filename, 'wb') as f:
			pickle.dump(obj, f)

	def unpickle_object(self, filename):
		with open(filename, 'rb') as f:
			obj = pickle.load(f)
			return obj

	def generate_processing_info(self, use_stored=False):
		directory = self.filenames['promising_combinations']
		pos = len(directory) - directory[::-1].find('/')
		directory = directory[:pos]
		dictio_of_users = None
		if use_stored and os.path.exists(directory + 'dictio_of_users') and os.path.exists(directory + 'dictio_of_values') and os.path.exists(directory + 'dictio_of_weights'):
			dictio_of_users = self.unpickle_object(directory + 'dictio_of_users')
			dictio_of_values = self.unpickle_object(directory + 'dictio_of_values')
			dictio_of_weights = self.unpickle_object(directory + 'dictio_of_weights')
			#dictio_of_usage = self.unpickle_object(directory + 'dictio_of_usage')
		else:
			dictio_of_users, dictio_of_values, dictio_of_weights = self.generate_clean_dataset(
				self.filenames['dataset_filename'])
			self.pickle_object(dictio_of_users, directory + 'dictio_of_users')
			self.pickle_object(dictio_of_values, directory + 'dictio_of_values')
			self.pickle_object(dictio_of_weights, directory + 'dictio_of_weights')
			#self.pickle_object(dictio_of_usage, directory + 'dictio_of_usage')

		#self.store_clean_dataset(dictio_of_users)
		promising_combinations = None
		bin_sparse_matrix = None
		weight_sparse_matrix = None

		if use_stored and os.path.exists(self.filenames['promising_combinations']):
			promising_combinations = self.unpickle_object(self.filenames['promising_combinations'])
		else:
			promising_combinations = self.get_promising_combinations(dictio_of_users, dictio_of_values)
			self.pickle_object(promising_combinations, self.filenames['promising_combinations'])

		print("Number of promissing combinations %d" % (len(promising_combinations)))
		if use_stored and os.path.exists(self.filenames['bin_sparse_matrix']):
			bin_sparse_matrix = self.unpickle_object(self.filenames['bin_sparse_matrix'])
		else: 
			bin_sparse_matrix = self.gen_binary_sparse_matrix(dictio_of_values, dictio_of_users)
			self.pickle_object(bin_sparse_matrix, self.filenames['bin_sparse_matrix'])

		if use_stored and os.path.exists(self.filenames['weight_sparse_matrix']):
			weight_sparse_matrix = self.unpickle_object(self.filenames['weight_sparse_matrix'])
		else: 
			weight_sparse_matrix = self.gen_weight_sparse_matrix(dictio_of_values, dictio_of_users, dictio_of_weights)
			self.pickle_object(weight_sparse_matrix, self.filenames['weight_sparse_matrix'])
		
		return dictio_of_users, dictio_of_values, dictio_of_weights, promising_combinations, bin_sparse_matrix, weight_sparse_matrix
	
	def get_filename_dir(self, path):
		directory = path[0:len(path)-path[::-1].find("/")]
		filename = path[len(path) - path[::-1].find("/"):]
		return directory, filename

	def join_all_results(self, origin_filename):
		directory, filename = self.get_filename_dir(origin_filename)
		print(origin_filename)
		print("[-] Joining all subfiles in same file")
		list_files = [directory + name for name in os.listdir(directory) if filename in name and not filename == name]
		print("[-] Total files for %s: %d" % (filename, len(list_files)))
		result_file = origin_filename
		f1 = open(result_file, 'w+', buffering=2)
		total_list = []
		status.create_numbar(100, len(list_files))
		first = True
		for ind, file in enumerate(list_files):
			status.update_numbar(ind, len(list_files))
			with open(file, 'r') as f2:
				line = f2.readline()
				if first:
					f1.write(line)
					f1.flush()
					first = False
				line = f2.readline()
				while line:
					f1.write(line)
					f1.flush()
					line = f2.readline()
				
			os.remove(file)
		status.end_numbar()
		f1.close()
		print("[+] Done joining all subfiles in same file")
		# print("[-] Removing files that are not needed")
		# status.create_numbar(100, len(list_files))
		# for ind, file in enumerate(list_files):
		# 	status.update_numbar(ind, len(list_files))
		# 	os.remove(file)
		# status.end_numbar()
		# print("[+] Done removing all subfiles")
		#print("[-] Sorting file before storing")
		#total_list = sorted(total_list, key=lambda x: str(x[2]) + x[0] + x[1], reverse=True)
		#print("[+] Done sorting the file, storing it to file...")
		#gen_csv_from_tuples(result_file , ["IdAuthor1", "IdAuthor2", "Score"], total_list)
	
	def get_sizes(self):
		dictio_of_users, dictio_of_values, _, _ = self.generate_clean_dataset(self.filenames['dataset_filename'])
		lu,lv = len(dictio_of_users), len(dictio_of_values)
		print("USERS: %d" %(lu))
		print("VALUES: %d" % (lv))
		return np.array([lu, lv]).reshape(1,-1)

	def do_all_sparse_matrix_link(self):
		print("[+] Generating processing information")
		dictio_of_users, dictio_of_values, dictio_of_weights, promising_combinations, bin_sparse_matrix, weight_sparse_matrix = self.generate_processing_info()

		print("Generating promising combinations v2")
		lst_users2 = [(x[0], x[1]) for x in read_csv_list("combined_results.csv")[1:]]
		self.pickle_object(lst_users2, "myname.pkl")
		promising_combinations = self.gen_intersection(promising_combinations, lst_users2)
		print("NEW PROMISING COMBINATIONS: %d" % (len(promising_combinations)))

		self.process_pairs_of_users_sparse_matrix_alt(dictio_of_users, dictio_of_values, dictio_of_weights,
		promising_combinations, bin_sparse_matrix, weight_sparse_matrix, 
		self.filenames['coincidence_csv_filename'], self.filenames['uniqueness_csv_filename'])
		# Joining all files into one
		self.join_files()

	def do_all_sparse_matrix(self):
		print("[+] Generating processing information")
		dictio_of_users, dictio_of_values, dictio_of_weights, promising_combinations, bin_sparse_matrix, weight_sparse_matrix = self.generate_processing_info()
		print("USERS %d, VALUES: %d" % (len(dictio_of_users), len(dictio_of_values)))
		# print("Generating promising combinations v2")
		# lst_users2 = [(x[0], x[1]) for x in read_csv_list("combined_results.csv")[1:]]
		# self.pickle_object(lst_users2, "myname.pkl")
		# promising_combinations = self.gen_intersection(promising_combinations, lst_users2)
		print("NEW PROMISING COMBINATIONS: %d" % (len(promising_combinations)))

		self.process_pairs_of_users_sparse_matrix_alt(dictio_of_users, dictio_of_values, dictio_of_weights,
		promising_combinations, bin_sparse_matrix, weight_sparse_matrix, 
		self.filenames['coincidence_csv_filename'], self.filenames['uniqueness_csv_filename'])
		# Joining all files into one
		self.join_files()

	def do_all(self):
		dictio_of_users, dictio_of_values, dictio_of_weights = self.generate_clean_dataset(self.filenames['dataset_filename'])
		#self.store_clean_dataset(dictio_of_users)
		promising_combinations = self.get_promising_combinations(dictio_of_users, dictio_of_values)
		self.gen_binary_matrix(dictio_of_values, dictio_of_users, self.filenames['bin_matrix_filename'])
		self.process_pairs_of_users(dictio_of_users, dictio_of_values, dictio_of_weights, promising_combinations, 
			self.filenames['bin_matrix_filename'], self.filenames['coincidence_csv_filename'])
		os.remove(self.filenames['bin_matrix_filename'])
		#self.process_score(dictio_of_users, dictio_of_values, self.filenames['bin_matrix_filename'], 
		#	self.filenames['coincidence_score_filename'])
		#self.gen_score_csv(dictio_of_users, dictio_of_values, self.filenames['coincidence_score_filename'], 
		#	self.filenames['coincidence_csv_filename'])
		#self.gen_weight_matrix(dictio_of_values, dictio_of_users, dictio_of_weights,, self.filenames['weight_matrix_filename'])
		#lst_users = read_csv_list(self.filenames['coincidence_csv_filename'])[1:]
		#self.process_pairs_of_users(dictio_of_users, dictio_of_values, lst_users, 
			#self.filenames['weight_matrix_filename'], self.filenames['uniqueness_csv_filename'])
		#os.remove(self.filenames['weight_matrix_filename'])

	def gen_distance_vector(self, entry):
		#to int
		entry = np.array([int(x) for x in entry])
		return (entry / entry.sum()).round(3)
		
	def do_all_vector_distances(self):
		raw_dataset = read_csv_list(self.filenames['dataset_filename'])[1:]
		dictio_of_users = dict()
		for entry  in raw_dataset:
			dictio_of_users[entry[0]] = self.gen_distance_vector(entry[1:])
		#self.store_clean_dataset(dictio_of_users)
		promising_combinations = self.get_promising_combinations(dictio_of_users, dictio_of_values)
		self.gen_binary_matrix(dictio_of_values, dictio_of_users, self.filenames['bin_matrix_filename'])
		self.process_pairs_of_users(dictio_of_users, dictio_of_values, promising_combinations, 
			self.filenames['bin_matrix_filename'], self.filenames['coincidence_csv_filename'])
		os.remove(self.filenames['bin_matrix_filename'])
		#self.process_score(dictio_of_users, dictio_of_values, self.filenames['bin_matrix_filename'], 
		#	self.filenames['coincidence_score_filename'])
		#self.gen_score_csv(dictio_of_users, dictio_of_values, self.filenames['coincidence_score_filename'], 
		#	self.filenames['coincidence_csv_filename'])
		self.gen_weight_matrix(dictio_of_values, dictio_of_users, dictio_of_weights, self.filenames['weight_matrix_filename'])
		lst_users = read_csv_list(self.filenames['coincidence_csv_filename'])[1:]
		self.process_pairs_of_users(dictio_of_users, dictio_of_values, lst_users, 
			self.filenames['weight_matrix_filename'], self.filenames['uniqueness_csv_filename'])
		os.remove(self.filenames['weight_matrix_filename'])
	def join_files(self):
		self.join_all_results(self.filenames['coincidence_csv_filename'])
		#self.join_all_results(self.filenames['uniqueness_csv_filename'])

##################################################################################################
## FUNCTIONS FROM THIS POINT ARE USED TO ANALYZE THE DISTRIBUTION OF DATA BUT NOT AS PROCESSING ##
##################################################################################################
	def gen_intersection(self, lst_users1, lst_users2):
		return list(set(lst_users1).intersection(set(lst_users2)))

	def evaluate_reduction(self):
		tic = time.time()
		lst_users = read_csv_list(self.filenames['dataset_filename'])[1:]

		dictio_of_users = self.gen_dictio_of_users(lst_users)
		dictio_of_values = self.gen_dictio_of_values(lst_users)
		ulen1, vlen1 = len(dictio_of_users), len(dictio_of_values)
		umean, ustd = self.get_mean_std(dictio_of_users)
		vmean, vstd = self.get_mean_std(dictio_of_values)

		dictio_of_users, dictio_of_values, _, _= self.generate_clean_dataset(self.filenames['dataset_filename']) 

		print("[+] Finished Dataset Cleanup in: %f" %(time.time() - tic))
		ulen2, vlen2 = len(dictio_of_users), len(dictio_of_values)
		return ulen1, ulen2, vlen1, vlen2, umean, ustd, vmean, vstd

	def gen_distribution_from_dictio(self, dictio):
		lst_dist = []
		v = np.array([len(v) for k, v in dictio.items()])
		maxnum = np.max(v)

		for i in range(1, maxnum):
			count = np.count_nonzero(v == i)
			if count > 0:
				lst_dist.append((i, count))
		return lst_dist

	def gen_optimal_reduction_csv(self, filename):
		tic = time.time()
		print("[-] Starting the optimal reduction calculation")
		lst_users = read_csv_list(self.filenames['dataset_filename'])[1:]
		lst_results = []
		dictio_of_users = self.gen_dictio_of_users(lst_users)
		dictio_of_values = self.gen_dictio_of_values(lst_users)

		dictio_of_users, dictio_of_values = self.clean_dataset(dictio_of_users,dictio_of_values,
			user_removal=self.user_removal, value_removal=self.value_removal)
		init_len_u, init_len_v = len(dictio_of_users), len(dictio_of_values)
		
		lst_results.append((0, init_len_v, init_len_u, 0, 0))
		dictio_of_weights = self.gen_dictio_of_weigths (dictio_of_values)
		lst_rarity = self.gen_rarity_dist(dictio_of_weights)
		for rarity, _ in lst_rarity:
			
			dictio_of_users, dictio_of_values, dictio_of_weights = self.clean_dataset_optimal_red(dictio_of_users, dictio_of_values, 
				dictio_of_weights, rarity)
			post_len_u, post_len_v = len(dictio_of_users), len(dictio_of_values)
			lst_results.append((rarity, post_len_v, post_len_u, init_len_v - post_len_v, init_len_u - post_len_u))

		gen_csv_from_tuples(filename, 
			["Rarity Removed", "Value Length", "User Length", "Values Removed", "Users Removed"], 
			lst_results)
		print("[+] Finished Dataset Cleanup in: %f" %(time.time() - tic))
		return lst_results
	def gen_reduction_graphs(self,filename, graph_filename):
		tic = time.time()

		lst = read_csv_list(filename)[1:]
		X = np.array([int(x[0]) for x in lst])
		V = np.array([float(x[1]) for x in lst])
		U = np.array([float(x[2]) for x in lst])
		RV = np.array([float(x[3]) for x in lst])
		RU = np.array([float(x[4]) for x in lst])
		maxu = U[0]
		maxv = V[0]

		V /= maxv / 100
		U /= maxu / 100 
		#print("U", U)
		#print("V", V)
		plt.style.use('seaborn-darkgrid')
		palette = plt.get_cmap('Set1')

		plt.plot(X, V, color=palette(0), label="Values (V)")
		plt.plot(X, U, color=palette(1), label="Users (U)")
		#plt.plot(X, CY, color=palette(1), linestyle='--', label="CDF(Y)") 
			#marker='o', markersize=3, markerfacecolor=palette(0))
		plt.axvline(x=self.rarity_bound, color=palette(2), linestyle='-.', 
			label="Rarity Cutoff (%d)"%(self.rarity_bound))
		plt.yticks(np.arange(30, 105, 5))#, rotation='vertical')
		plt.xlabel('Rarity', fontstyle = 'italic', fontsize=12.0)
		plt.ylabel('Removal Ratio', fontstyle = 'italic', fontsize=12.0)

				#plt.scatter('', xy=(x, cy), xytext=(0, 0), color='red' , textcoords='offset points') 
		plt.legend(loc='best')
		plt.savefig(graph_filename, format="PDF", bbox_inches='tight')
		plt.clf()
		print("[+] Finished Graph Generation Cleanup in: %f" %(time.time() - tic))

	def get_coincidences_for_users(self, lst_users, dictio = None):
		dictio_of_users, dictio_of_values, dictio_of_weights = self.generate_clean_dataset(
			self.filenames['dataset_filename'])
		dictio = dict() if dictio is None else dictio
		for tup in lst_users:
			if (tup[0] in dictio_of_users.keys()) and (tup[1] in dictio_of_users.keys()):
				if not tup in dictio.keys():
					dictio[tup] = list()
				coin = set(dictio_of_users[tup[0]]).intersection(set(dictio_of_users[tup[1]]))
				for i in coin:
					dictio[tup].append(i)

		return dictio

	def get_optimal_reduction_graph(self, filename):
		tic = time.time()

		lst = read_csv_list(filename)[1:]
		X = np.array([int(x[0]) for x in lst])
		V = np.array([float(x[1]) for x in lst])
		U = np.array([float(x[2]) for x in lst])
		RV = np.array([float(x[3]) for x in lst])
		RU = np.array([float(x[4]) for x in lst])
		maxu = U[0]
		maxv = V[0]

		V /= maxv
		U /= maxu
		#print("U", U)
		#print("V", V)

		return X, V, U, self.rarity_bound

	def gen_optimal_reduction(self, filename, graph_filename):
		self.gen_optimal_reduction_csv(filename)
		self.gen_reduction_graphs(filename, graph_filename)

	def gen_distribution_csv(self, filename):
		tic = time.time()
		lst_users = read_csv_list(self.filenames['dataset_filename'])[1:]

		dictio_of_users = self.gen_dictio_of_users(lst_users)
		dictio_of_values = self.gen_dictio_of_values(lst_users)
		#dictio_of_users, dictio_of_values = self.clean_dataset(dictio_of_users,dictio_of_values,
			#user_removal=self.user_removal, value_removal=self.value_removal)
		
		user_dist = self.gen_distribution_from_dictio(dictio_of_users)
		gen_csv_from_tuples(filename+"_u", ["n", "num_users_with_n_elements"], user_dist)
		value_dist = self.gen_distribution_from_dictio(dictio_of_values)
		gen_csv_from_tuples(filename+"_v", ["n", "num_values_with_n_users"], value_dist)
		print("[+] Finished Distribution Extraction in: %f" %(time.time() - tic))
