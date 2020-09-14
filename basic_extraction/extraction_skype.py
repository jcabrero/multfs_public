import psycopg2, json, csv
import re
import os
import status
import operator
import math
import numpy as np
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
from common_utils import gen_csv_from_tuples, read_csv_list, make_query

global_lst = []

def extract_user_to_skype_csv():
	query= """WITH "A" AS (SELECT
  		CAST("Post"."Author" AS text) || '[' || CAST("Post"."Site" AS text) || ']' as "Author", 
  		regexp_matches( "Content", 'skype\s*:\s*([a-zA-Z0-9:\.]{1,37})', 'g') AS "skype"
  		FROM "Post" WHERE "Content" ~ 'skype\s*:\s*([a-zA-Z0-9:\.]{1,37})'),
		"B" AS (SELECT "Author", lower("skype"[1]) as "skype", count(*) as "repetitions" FROM "A" GROUP BY "Author", "skype" )
		SELECT "B"."Author",
		string_agg("B"."skype", ', ') as "reps" 
		FROM "B" GROUP BY "B"."Author";"""
	rows = make_query(query)
	#rows = [row[:1] + tuple([x for x in row[1].split(", ")],) for row in rows if row[0] != -1]
	rows = [list(row[:1] + tuple([x for x in row[1].split(", ")],)) for row in rows if row[0] != -1]
	#print(rows)
	for row in range(len(rows)):
		#print(type(rows[row]), len(rows[row]))
		for col in range(1, len(rows[row])):
			#print(row, col, rows[row][col])
			if rows[row][col][-1] == '.':
				#print("Changed: %s by %s" % (rows[row][col], rows[row][col][10:]))
				rows[row][col] = rows[row][col][:-1]
				

	for row in range(len(rows)):
		rows[row] = (rows[row][0],) + tuple(set(rows[row][1:]))
	print(len(rows))
	
	print (len(rows))
	gen_csv_from_tuples("skype_files/user_to_skype.csv", ["IdAuthor", "skype"], rows)

def extract_skype_to_usage():
	query= """WITH "A" AS (SELECT
  		regexp_matches( "Content", 'skype\s*:\s*([a-zA-Z0-9:\.]{1,37})', 'g') AS "skype"
  		FROM "Post" WHERE "Content" ~ 'skype\s*:\s*([a-zA-Z0-9:\.]{1,37})'),
		"B" AS (SELECT lower("skype"[1]) as "skype", count(*) as "repetitions" FROM "A" GROUP BY "skype" )
		SELECT "skype", "repetitions" FROM "B";"""
	rows = make_query(query)
	rows = [list(row) for row in rows]
	for i in range(len(rows)):
		if rows[i][0][-1] == '.':
			rows[i][0] = rows[i][0][:-1]
	rows = [tuple(row) for row in rows]
	#print(rows)
	print(len(rows))
	#rows = [row[:1] + tuple([x for x in row[1].split(", ")],) for row in rows if row[0] != -1]
	print (len(rows))
	gen_csv_from_tuples("skype_files/skype_count.csv", ["skype", "Reps"], rows)	

def gen_vector_for_user(lst1, dictio):
	base = np.zeros((len(dictio),1), dtype=int)
	for i in range(0,len(lst1),2):
		#scorei = lst1[i]
		skype = lst1[i+1]
		base[dictio[skype]] = 1
		#base[dictio[skype]] = scorei
	return base

def get_score3(ind):
	print(ind)
	v1 = gen_vector_for_user(global_lst[ind[0]][2:], global_dictio)
	v2 = gen_vector_for_user(global_lst[ind[1]][2:], global_dictio)
	score = np.squeeze(np.dot(v1.T, v2))
	return ind + (score,)


def gen_num(string):
	string = string.split(".")
	num = 0
	for i in range(4):
		num += int(string[i]) * (1000 ** (3 - i))
	#print(num)
	return num

def modify_skype(elem):
	lst = [elem[0], elem[1]]
	for i in elem[2:]:
		j = i.split(":")
		lst.append((j[0], j[1]))
	return tuple(lst)



def gen_new_dataset():
	global global_lst
	#lst = read_csv_list("user_skypes.csv")[1:]
	lst = read_csv_list("skype_files/user_to_skype.csv")[1:]
	print("Length of the Dataset: %d" % (len(lst)))
	#pool = mp.Pool(processes=16)
	#lst = pool.map(modify_skype, lst)
	#clean_dataset(lst)
	#global_lst = sorted(lst, key=lambda x: len(x), reverse=True)
	return lst
	#gen_csv_from_tuples("please_work2.csv", ["Author", "Username", "Site", "IdPost", "skype"], lst)

def get_different_skypes(lst):
	seti = set()
	dictio = {}
	for i in lst:
		for j in i[2:]:
			seti.add(j[1])
	dictio = {i: indi for indi, i in enumerate(list(seti))}
	return dictio

def get_different_skypes2(lst, default="list"):
	seti = set()
	dictio = {}
	for i in lst:
		for j in i[2:]:
			seti.add(j[1])
	dictio = {i: 0 for indi, i in enumerate(list(seti))}
	for i in lst:
		for j in i[2:]:
			dictio[j[1]] += 1
	if default is "list":
		return [(k, v) for k,v in dictio.items()]
	else:
		return dictio

def gen_dictio_of_users(lst):
	dictio = {}
	for i in lst:
		key = i[0]
		dictio[key] = []
		for j in i[1:]:
			dictio[key].append(j)
	return dictio

def gen_dictio_of_skypes(lst):
	dictio = {}
	for i in lst:
		key = i[0]
		for j in i[1:]:
			if j in dictio.keys():
				dictio[j].append(key)
			else:
				dictio[j] = [key]
	return dictio

def clean_users_from_dictios(list_skype, dictio_of_users, dictio_of_skypes):
	old_len_u, old_len_skype = len(dictio_of_users), len(dictio_of_skypes)
	#print("skypes removed: ", len(list_skype) )
	for skype in list_skype:
		for i in dictio_of_skypes[skype]:
			dictio_of_users[i].remove(skype)
		ret2 = dictio_of_skypes.pop(skype, None)
		if ret2 is None:
			print("THERE IS AN ERROR: ", skype)
	# Update the list of users
	users_removed = [k for k, v in dictio_of_users.items() if v == []]
	#print("Users removed: ", len(users_removed))
	for user in users_removed:
		ret = dictio_of_users.pop(user, None)
		if ret is None:
			print("ERROR")
	row_format ="{:>15}" * 4
	print("-" * 15 * 4)
	print(row_format.format("Original skypes", "Removed skypes", "New skypes", "Percentage"))
	print(row_format.format("%d"% (old_len_skype), "%d"%(len(list_skype)), "%d"%(len(dictio_of_skypes)), "%f" %(len(dictio_of_skypes)/old_len_skype)))
	print(row_format.format("Original Users", "Removed Users", "New Users", "Percentage"))
	print(row_format.format("%d"% (old_len_u), "%d"%(len(users_removed)), "%d"%(len(dictio_of_users)), "%f" %(len(dictio_of_users)/old_len_u)))
	print("-" * 15 * 4)
	#print(len(dictio_of_users)/old_len_u, len(dictio_of_skypes)/old_len_skype, 
		#len(dictio_of_users), len(dictio_of_skypes))
	return dictio_of_users, dictio_of_skypes

def clean_users_from_dictios2(list_users, dictio_of_users, dictio_of_skypes):
	old_len_u, old_len_skype = len(dictio_of_users), len(dictio_of_skypes)
	#print("skypes removed: ", len(list_skype) )
	for user in list_users:
		for skype in dictio_of_users[user]:
			dictio_of_skypes[skype].remove(user)
		ret2 = dictio_of_users.pop(user, None)
		

	# Update the list of users
	skypes_removed = [k for k, v in dictio_of_skypes.items() if v == []]
	#print("Users removed: ", len(users_removed))
	for skype in skypes_removed:
		ret = dictio_of_skypes.pop(skype, None)
		if ret is None:
			print("ERROR")
	row_format ="{:>15}" * 4
	print("-" * 15 * 4)
	print(row_format.format("Original skypes", "Removed skypes", "New skypes", "Percentage"))
	print(row_format.format("%d"% (old_len_skype), "%d"%(len(skypes_removed)), "%d"%(len(dictio_of_skypes)), "%f" %(len(dictio_of_skypes)/old_len_skype)))
	print(row_format.format("Original Users", "Removed Users", "New Users", "Percentage"))
	print(row_format.format("%d"% (old_len_u), "%d"%(len(list_users)), "%d"%(len(dictio_of_users)), "%f" %(len(dictio_of_users)/old_len_u)))
	print("-" * 15 * 4)
	#print(len(dictio_of_users)/old_len_u, len(dictio_of_skypes)/old_len_skype, 
		#len(dictio_of_users), len(dictio_of_skypes))
	return dictio_of_users, dictio_of_skypes


def clean_dataset(dictio_of_users, dictio_of_skypes):
	# Remove skypes with 1 appearance.
	oneapp = [k for k,v in dictio_of_skypes.items() if len(v) == 1]
	while(len(oneapp) > 0):
		print("Removing skypes that appear once...")
		dictio_of_users, dictio_of_skypes = clean_users_from_dictios(oneapp, dictio_of_users, dictio_of_skypes)
		#print("Removing skypes that appear more than 12 times...")
		#multiskype = [k for k,v in dictio_of_skypes.items() if len(v) > 12]
		#dictio_of_users, dictio_of_skypes = clean_users_from_dictios(multiskype, dictio_of_users, dictio_of_skypes)
		#print("Removing users that have less than 5 skypes...")
		#atleast10 = [k for k,v in dictio_of_users.items() if len(v) < 5]
		#dictio_of_users, dictio_of_skypes = clean_users_from_dictios2(atleast10, dictio_of_users, dictio_of_skypes)
		oneapp = [k for k,v in dictio_of_skypes.items() if len(v) == 1]
	return dictio_of_users, dictio_of_skypes



def gen_bin_matrix_of_users(dictio_of_skypes, dictio_of_users):
	num_users = len(dictio_of_users)
	num_skypes = len(dictio_of_skypes)
	# Transform dictionary to indexes
	for indk, skype in enumerate(dictio_of_skypes.keys()):
		dictio_of_skypes[skype] = indk
	#Transform users to matrices.
	print("ESTIMATED SIZE OF MATRIX: %f GB" % (num_users * num_skypes * 1 / (1024 ** 3)))
	matrix_map = np.memmap('skype_files/skype_matrix_map.dat', dtype=np.uint8, mode ='w+', shape=(num_users, num_skypes))
	status.create_numbar(100, num_users)
	for ind, user in enumerate(dictio_of_users.keys()):
		status.update_numbar(ind, num_users)
		base = np.zeros((num_skypes,1), dtype=np.uint8)
		for skype in dictio_of_users[user]:
			base[dictio_of_skypes[skype]] = 1
		base = np.squeeze(base)
		matrix_map[ind] = base[:]
	status.end_numbar()
	print("Flushing...")
	matrix_map.flush()

def gen_new_matrix_of_users(dictio_of_skypes, dictio_of_users, dictio_of_values):
	num_users = len(dictio_of_users)
	num_skypes = len(dictio_of_skypes)
	# Transform dictionary to indexes
	for indk, skype in enumerate(dictio_of_skypes.keys()):
		dictio_of_skypes[skype] = indk
	#Transform users to matrices.
	print("ESTIMATED SIZE OF MATRIX: %f GB" % (num_users * num_skypes * 1 / (1024 ** 3)))
	matrix_map = np.memmap('skype_files/skype_matrix_map.dat', dtype=np.uint8, mode ='w+', shape=(num_users, num_skypes))
	status.create_numbar(100, num_users)
	for ind, user in enumerate(dictio_of_users.keys()):
		status.update_numbar(ind, num_users)
		base = np.zeros((num_skypes,1), dtype=np.uint8)
		for skype in dictio_of_users[user]:
			x = dictio_of_values[skype]
			base[dictio_of_skypes[skype]] = x
		base = np.squeeze(base)
		matrix_map[ind] = base[:]
	status.end_numbar()
	print("Flushing...")
	matrix_map.flush()


def generate_clean_dataset():
	tic = time.time()
	lst_users = gen_new_dataset()
	#print(lst_users[:3])
	lst_users = sorted(lst_users, key=lambda x: len(x), reverse=True)
	dictio_of_users = gen_dictio_of_users(lst_users)
	dictio_of_skypes = gen_dictio_of_skypes(lst_users)
	dictio_of_users, dictio_of_skypes = clean_dataset(dictio_of_users,dictio_of_skypes)
	print("SECONDS: %f" %(time.time() - tic))
	return dictio_of_users, dictio_of_skypes

def process_skypes_euclidean(dictio_of_users, dictio_of_skypes):
	tic = time.time()
	num_users = len(dictio_of_users)
	num_skypes = len(dictio_of_skypes)

	matrix_map = np.memmap('skype_files/skype_matrix_map.dat', dtype=np.uint8,  shape=(num_users, num_skypes))
	#matrix_map = np.array(matrix_map)

	print(matrix_map.shape)
	print("ESTIMATED SIZE OF MATRIX: %f GB" % (num_users * num_users * 4 / (1024 ** 3)))
	status.create_numbar(100, num_users)
	#res = np.dot(matrix_map, matrix_map.T)
	res2 = np.memmap('skype_files/skype_euc_score.dat', dtype=float ,mode ='w+', shape=(num_users, num_users))
	#res3 = np.memmap('skype_files/skype_dis_score.dat', dtype=np.uint32 ,mode ='w+', shape=(num_users, num_users))
	for i1 in range(num_users):
		status.update_numbar(i1, num_users)
		v1 = np.array(matrix_map[i1], dtype=np.int32)
		for i2 in range(i1 + 1, num_users):
			v2 = np.array(matrix_map[i2], dtype=np.int32)
			#print(v1)
			euc_score = np.linalg.norm(v1-v2)
			#dis_score = np.dot(v1, v2)
			res2[i1][i2] = euc_score
			#res3[i1][i2] = dis_score
			#res2[i2][i1] = score
	status.end_numbar()
	print("Flushing...")
	res2.flush()	
	print("SECONDS: %f" %(time.time() - tic))

def process_skypes_distance(dictio_of_users, dictio_of_skypes):
	tic = time.time()
	num_users = len(dictio_of_users)
	num_skypes = len(dictio_of_skypes)

	matrix_map = np.memmap('skype_files/skype_matrix_map.dat', dtype=np.uint8,  shape=(num_users, num_skypes))
	#matrix_map = np.array(matrix_map)

	print(matrix_map.shape)
	print("ESTIMATED SIZE OF MATRIX: %f GB" % (num_users * num_users * 4 / (1024 ** 3)))
	status.create_numbar(100, num_users)
	#res = np.dot(matrix_map, matrix_map.T)
	#res2 = np.memmap('skype_files/skype_euc_score.dat', dtype=float ,mode ='w+', shape=(num_users, num_users))
	res3 = np.memmap('skype_files/skype_dis_score.dat', dtype=np.uint32 ,mode ='w+', shape=(num_users, num_users))
	for i1 in range(num_users):
		status.update_numbar(i1, num_users)
		v1 = np.array(matrix_map[i1], dtype=np.uint32)
		for i2 in range(i1 + 1, num_users):
			v2 = np.array(matrix_map[i2], dtype=np.uint32)
			#print(v1)
			#euc_score = np.linalg.norm(v1-v2)
			dis_score = np.dot(v1, v2)
			#res2[i1][i2] = euc_score
			res3[i1][i2] = dis_score
			#res2[i2][i1] = score
	status.end_numbar()
	print("Flushing...")
	res3.flush()	
	print("SECONDS: %f" %(time.time() - tic))


def read_results_dis(dictio_of_users, dictio_of_skypes):
	tic = time.time()
	
	num_users = len(dictio_of_users)
	lst_users = list(dictio_of_users.keys())
	lst = []
	res2 = np.memmap('skype_files/skype_dis_score.dat', dtype=np.uint32, mode ='r', shape=(num_users, num_users))
	status.create_numbar(100, num_users)
	for i in range(num_users):
		status.update_numbar(i, num_users)
		for j in range(i + 1, num_users):
			lst.append((lst_users[i],lst_users[j],res2[i][j]))
	status.end_numbar()
	print("SECONDS: %f" %(time.time() - tic))
	sortedl = sorted(lst, key=lambda x: x[2], reverse=True)
	print(sortedl[:10])
	print("SECONDS: %f" %(time.time() - tic))
	gen_csv_from_tuples("skype_files/results_skype_dis.csv", ["IdAuthor1", "IdAuthor2", "Score"], sortedl)

def read_results_euc(dictio_of_users, dictio_of_skypes):
	tic = time.time()
	
	num_users = len(dictio_of_users)
	lst_users = list(dictio_of_users.keys())
	lst = []
	res2 = np.memmap('skype_files/skype_euc_score.dat', dtype=float, mode ='r', shape=(num_users, num_users))
	status.create_numbar(100, num_users)
	for i in range(num_users):
		status.update_numbar(i, num_users)
		for j in range(i + 1, num_users):
			lst.append((lst_users[i],lst_users[j],res2[i][j]))
	status.end_numbar()
	print("SECONDS: %f" %(time.time() - tic))
	sortedl = sorted(lst, key=lambda x: x[2], reverse=False)
	print(sortedl[:10])
	print("SECONDS: %f" %(time.time() - tic))
	gen_csv_from_tuples("skype_files/results_skype_euc.csv", ["IdAuthor1", "IdAuthor2", "Score"], sortedl)

def gen_skype_values():

	lst_skypes = read_csv_list("skype_files/skype_count.csv")[1:]
	print("Lenght Skype Count: %d" %(len(lst_skypes)))
	lst_skypes = sorted(lst_skypes, key=lambda x: x[1], reverse=True)
	print(lst_skypes[:3])
	divisions = int(math.ceil(float(len(lst_skypes)) / 254.0))
	dictio = {}
	for i in range(0, 254):
		start = i * divisions
		end = (i + 1) * divisions
		for elem in lst_skypes[start:end]:
			dictio[elem[0]] = (i + 1)
	#print(dictio)
	return dictio


def new_read_results_euc(dictio_of_users, dictio_of_skypes):
	print("New user extraction...")
	tic = time.time()
	num_users = len(dictio_of_users)
	lst_users = list(dictio_of_users.keys())
	lst = []
	res2 = np.memmap('skype_files/skype_dis_score.dat', dtype=np.uint32, mode ='r', shape=(num_users, num_users))
	status.create_numbar(100, num_users)
	for i in range(num_users):
		status.update_numbar(i, num_users)
		for j in range(i + 1, num_users):
			if res2[i][j] > 0:
				lst.append((lst_users[i],lst_users[j],res2[i][j]))
	status.end_numbar()
	print("SECONDS: %f" %(time.time() - tic))
	print("Old length: %d" % (len(lst)))
	lst = [x for x in lst if x[2] > 0]
	print("New length: %d" % (len(lst)))
	sortedl = sorted(lst, key=lambda x: x[2], reverse=True)
	print("SECONDS: %f" %(time.time() - tic))
	gen_csv_from_tuples("skype_files/new_results_skype_dis.csv", ["IdAuthor1", "IdAuthor2", "Score"], sortedl)

	lst = []
	res3 = np.memmap('skype_files/skype_euc_score.dat', dtype=float, mode ='r', shape=(num_users, num_users))
	dic_inv = {user: indi for indi, user in enumerate(lst_users)}
	status.create_numbar(100, len(sortedl))

	for indi, row in enumerate(sortedl):
		status.update_numbar(indi, len(sortedl))
		u1, u2 = row[0], row[1]
		ind1, ind2 = dic_inv[u1], dic_inv[u2]
		lst.append((u1, u2, res3[ind1][ind2]))
	status.end_numbar()
	print("SECONDS: %f" %(time.time() - tic))
	sortedl = sorted(lst, key=lambda x: x[2], reverse=False)
	print(sortedl[:10])
	print("SECONDS: %f" %(time.time() - tic))
	gen_csv_from_tuples("skype_files/new_results_skype_euc.csv", ["IdAuthor1", "IdAuthor2", "Score"], sortedl)

	

def combin(n,r):
	return int(np.math.factorial(n) / (np.math.factorial(r) * np.math.factorial(n - r)))

def main():
	# extract_skype_to_usage()
	#extract_user_to_skype_csv()
	dictio_of_users, dictio_of_skypes = generate_clean_dataset()
	dictio_of_values = gen_skype_values()
	#gen_bin_matrix_of_users(dictio_of_skypes, dictio_of_users)
	#process_skypes_euclidean(dictio_of_users, dictio_of_skypes)
	gen_new_matrix_of_users(dictio_of_skypes, dictio_of_users, dictio_of_values)
	process_skypes_distance(dictio_of_users, dictio_of_skypes)
	new_read_results_euc(dictio_of_users, dictio_of_skypes)
	# read_results_euc(dictio_of_users, dictio_of_skypes)
	# read_results_dis(dictio_of_users, dictio_of_skypes)
	# read_results(dictio_of_users, dictio_of_skypes)
	

if __name__ == "__main__":
	main()
