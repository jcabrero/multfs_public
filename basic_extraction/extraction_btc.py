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
import random
import string
from common_utils import gen_csv_from_tuples, read_csv_list, make_query

global_lst = []

def extract_user_to_btc_csv():
	query= """WITH "A" AS (SELECT
  		CAST("Post"."Author" AS text) || '[' || CAST("Post"."Site" AS text) || ']' as "Author",
  		regexp_matches( "Content", '([13][a-km-zA-HJ-NP-Z1-9]{25,34})', 'g') AS "btc"
  		FROM "Post" WHERE "Content" ~ '([13][a-km-zA-HJ-NP-Z1-9]{25,34})'),
		"B" AS (SELECT "Author", lower("btc"[1]) as "btc", count(*) as "repetitions" FROM "A" GROUP BY "Author", "btc" )
		SELECT "B"."Author",
		string_agg("B"."btc", ', ') as "reps" 
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
	gen_csv_from_tuples("btc_files/user_to_btc.csv", ["IdAuthor", "btc"], rows)

def extract_btc_to_usage():
	query= """WITH "A" AS (SELECT
		regexp_matches( "Content", '([13][a-km-zA-HJ-NP-Z1-9]{25,34})', 'g') AS "btc"
  		FROM "Post" WHERE "Content" ~ '([13][a-km-zA-HJ-NP-Z1-9]{25,34})'),
		"B" AS (SELECT lower("btc"[1]) as "btc", count(*) as "repetitions" FROM "A" GROUP BY "btc" )
		SELECT "btc", "repetitions" FROM "B";"""
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
	gen_csv_from_tuples("btc_files/btc_count.csv", ["btc", "Reps"], rows)	

def gen_vector_for_user(lst1, dictio):
	base = np.zeros((len(dictio),1), dtype=int)
	for i in range(0,len(lst1),2):
		#scorei = lst1[i]
		btc = lst1[i+1]
		base[dictio[btc]] = 1
		#base[dictio[btc]] = scorei
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

def modify_btc(elem):
	lst = [elem[0], elem[1]]
	for i in elem[2:]:
		j = i.split(":")
		lst.append((j[0], j[1]))
	return tuple(lst)



def gen_new_dataset():
	global global_lst
	#lst = read_csv_list("user_btcs.csv")[1:]
	lst = read_csv_list("btc_files/user_to_btc.csv")[1:]
	print("Length of the Dataset: %d" % (len(lst)))
	#pool = mp.Pool(processes=16)
	#lst = pool.map(modify_btc, lst)
	#clean_dataset(lst)
	#global_lst = sorted(lst, key=lambda x: len(x), reverse=True)
	return lst
	#gen_csv_from_tuples("please_work2.csv", ["Author", "Username", "Site", "IdPost", "btc"], lst)

def get_different_btcs(lst):
	seti = set()
	dictio = {}
	for i in lst:
		for j in i[2:]:
			seti.add(j[1])
	dictio = {i: indi for indi, i in enumerate(list(seti))}
	return dictio

def get_different_btcs2(lst, default="list"):
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

def gen_dictio_of_btcs(lst):
	dictio = {}
	for i in lst:
		key = i[0]
		for j in i[1:]:
			if j in dictio.keys():
				dictio[j].append(key)
			else:
				dictio[j] = [key]
	return dictio

def clean_users_from_dictios(list_btc, dictio_of_users, dictio_of_btcs):
	old_len_u, old_len_btc = len(dictio_of_users), len(dictio_of_btcs)
	#print("btcs removed: ", len(list_btc) )
	for btc in list_btc:
		for i in dictio_of_btcs[btc]:
			dictio_of_users[i].remove(btc)
		ret2 = dictio_of_btcs.pop(btc, None)
		if ret2 is None:
			print("THERE IS AN ERROR: ", btc)
	# Update the list of users
	users_removed = [k for k, v in dictio_of_users.items() if v == []]
	#print("Users removed: ", len(users_removed))
	for user in users_removed:
		ret = dictio_of_users.pop(user, None)
		if ret is None:
			print("ERROR")
	row_format ="{:>15}" * 4
	print("-" * 15 * 4)
	print(row_format.format("Original btcs", "Removed btcs", "New btcs", "Percentage"))
	print(row_format.format("%d"% (old_len_btc), "%d"%(len(list_btc)), "%d"%(len(dictio_of_btcs)), "%f" %(len(dictio_of_btcs)/old_len_btc)))
	print(row_format.format("Original Users", "Removed Users", "New Users", "Percentage"))
	print(row_format.format("%d"% (old_len_u), "%d"%(len(users_removed)), "%d"%(len(dictio_of_users)), "%f" %(len(dictio_of_users)/old_len_u)))
	print("-" * 15 * 4)
	#print(len(dictio_of_users)/old_len_u, len(dictio_of_btcs)/old_len_btc, 
		#len(dictio_of_users), len(dictio_of_btcs))
	return dictio_of_users, dictio_of_btcs

def clean_users_from_dictios2(list_users, dictio_of_users, dictio_of_btcs):
	old_len_u, old_len_btc = len(dictio_of_users), len(dictio_of_btcs)
	#print("btcs removed: ", len(list_btc) )
	for user in list_users:
		for btc in dictio_of_users[user]:
			dictio_of_btcs[btc].remove(user)
		ret2 = dictio_of_users.pop(user, None)
		

	# Update the list of users
	btcs_removed = [k for k, v in dictio_of_btcs.items() if v == []]
	#print("Users removed: ", len(users_removed))
	for btc in btcs_removed:
		ret = dictio_of_btcs.pop(btc, None)
		if ret is None:
			print("ERROR")
	row_format ="{:>15}" * 4
	print("-" * 15 * 4)
	print(row_format.format("Original btcs", "Removed btcs", "New btcs", "Percentage"))
	print(row_format.format("%d"% (old_len_btc), "%d"%(len(btcs_removed)), "%d"%(len(dictio_of_btcs)), "%f" %(len(dictio_of_btcs)/old_len_btc)))
	print(row_format.format("Original Users", "Removed Users", "New Users", "Percentage"))
	print(row_format.format("%d"% (old_len_u), "%d"%(len(list_users)), "%d"%(len(dictio_of_users)), "%f" %(len(dictio_of_users)/old_len_u)))
	print("-" * 15 * 4)
	#print(len(dictio_of_users)/old_len_u, len(dictio_of_btcs)/old_len_btc, 
		#len(dictio_of_users), len(dictio_of_btcs))
	return dictio_of_users, dictio_of_btcs


def clean_dataset(dictio_of_users, dictio_of_btcs):
	# Remove btcs with 1 appearance.
	oneapp = [k for k,v in dictio_of_btcs.items() if len(v) == 1]
	while(len(oneapp) > 0):
		print("Removing btcs that appear once...")
		dictio_of_users, dictio_of_btcs = clean_users_from_dictios(oneapp, dictio_of_users, dictio_of_btcs)
		#print("Removing btcs that appear more than 12 times...")
		#multibtc = [k for k,v in dictio_of_btcs.items() if len(v) > 12]
		#dictio_of_users, dictio_of_btcs = clean_users_from_dictios(multibtc, dictio_of_users, dictio_of_btcs)
		#print("Removing users that have less than 5 btcs...")
		#atleast10 = [k for k,v in dictio_of_users.items() if len(v) < 5]
		#dictio_of_users, dictio_of_btcs = clean_users_from_dictios2(atleast10, dictio_of_users, dictio_of_btcs)
		oneapp = [k for k,v in dictio_of_btcs.items() if len(v) == 1]
	return dictio_of_users, dictio_of_btcs



def gen_bin_matrix_of_users(dictio_of_btcs, dictio_of_users):
	num_users = len(dictio_of_users)
	num_btcs = len(dictio_of_btcs)
	# Transform dictionary to indexes
	for indk, btc in enumerate(dictio_of_btcs.keys()):
		dictio_of_btcs[btc] = indk
	#Transform users to matrices.
	print("ESTIMATED SIZE OF MATRIX: %f GB" % (num_users * num_btcs * 1 / (1024 ** 3)))
	matrix_map = np.memmap('btc_files/btc_matrix_map.dat', dtype=np.uint8, mode ='w+', shape=(num_users, num_btcs))
	status.create_numbar(100, num_users)
	for ind, user in enumerate(dictio_of_users.keys()):
		status.update_numbar(ind, num_users)
		base = np.zeros((num_btcs,1), dtype=np.uint8)
		for btc in dictio_of_users[user]:
			base[dictio_of_btcs[btc]] = 1
		base = np.squeeze(base)
		matrix_map[ind] = base[:]
	status.end_numbar()
	print("Flushing...")
	matrix_map.flush()
	

def gen_new_matrix_of_users(dictio_of_btcs, dictio_of_users, dictio_of_values):
	num_users = len(dictio_of_users)
	num_btcs = len(dictio_of_btcs)
	# Transform dictionary to indexes
	for indk, btc in enumerate(dictio_of_btcs.keys()):
		dictio_of_btcs[btc] = indk
	#Transform users to matrices.
	print("ESTIMATED SIZE OF MATRIX: %f GB" % (num_users * num_btcs * 1 / (1024 ** 3)))
	matrix_map = np.memmap('btc_files/btc_matrix_map.dat', dtype=np.uint8, mode ='w+', shape=(num_users, num_btcs))
	status.create_numbar(100, num_users)
	for ind, user in enumerate(dictio_of_users.keys()):
		status.update_numbar(ind, num_users)
		base = np.zeros((num_btcs,1), dtype=np.uint8)
		for btc in dictio_of_users[user]:
			x = dictio_of_values[btc]
			base[dictio_of_btcs[btc]] = x
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
	dictio_of_btcs = gen_dictio_of_btcs(lst_users)
	dictio_of_users, dictio_of_btcs = clean_dataset(dictio_of_users,dictio_of_btcs)
	print("SECONDS: %f" %(time.time() - tic))
	return dictio_of_users, dictio_of_btcs

def process_btcs_euclidean(dictio_of_users, dictio_of_btcs):
	tic = time.time()
	num_users = len(dictio_of_users)
	num_btcs = len(dictio_of_btcs)

	matrix_map = np.memmap('btc_files/btc_matrix_map.dat', dtype=np.uint8,  shape=(num_users, num_btcs))
	#matrix_map = np.array(matrix_map)

	print(matrix_map.shape)
	print("ESTIMATED SIZE OF MATRIX: %f GB" % (num_users * num_users * 4 / (1024 ** 3)))
	status.create_numbar(100, num_users)
	#res = np.dot(matrix_map, matrix_map.T)
	res2 = np.memmap('btc_files/btc_euc_score.dat', dtype=float ,mode ='w+', shape=(num_users, num_users))
	#res3 = np.memmap('btc_files/btc_dis_score.dat', dtype=np.uint32 ,mode ='w+', shape=(num_users, num_users))
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

def process_btcs_distance(dictio_of_users, dictio_of_btcs):
	tic = time.time()
	num_users = len(dictio_of_users)
	num_btcs = len(dictio_of_btcs)

	matrix_map = np.memmap('btc_files/btc_matrix_map.dat', dtype=np.uint8,  shape=(num_users, num_btcs))
	#matrix_map = np.array(matrix_map)

	print(matrix_map.shape)
	print("ESTIMATED SIZE OF MATRIX: %f GB" % (num_users * num_users * 4 / (1024 ** 3)))
	status.create_numbar(100, num_users)
	#res = np.dot(matrix_map, matrix_map.T)
	#res2 = np.memmap('btc_files/btc_euc_score.dat', dtype=float ,mode ='w+', shape=(num_users, num_users))
	res3 = np.memmap('btc_files/btc_dis_score.dat', dtype=np.uint32 ,mode ='w+', shape=(num_users, num_users))
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


def read_results_dis(dictio_of_users, dictio_of_btcs):
	tic = time.time()
	
	num_users = len(dictio_of_users)
	lst_users = list(dictio_of_users.keys())
	lst = []
	res2 = np.memmap('btc_files/btc_dis_score.dat', dtype=np.uint32, mode ='r', shape=(num_users, num_users))
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
	gen_csv_from_tuples("btc_files/results_btc_dis.csv", ["IdAuthor1", "IdAuthor2", "Score"], sortedl)

def read_results_euc(dictio_of_users, dictio_of_btcs):
	tic = time.time()
	
	num_users = len(dictio_of_users)
	lst_users = list(dictio_of_users.keys())
	lst = []
	res2 = np.memmap('btc_files/btc_euc_score.dat', dtype=float, mode ='r', shape=(num_users, num_users))
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
	gen_csv_from_tuples("btc_files/results_btc_euc.csv", ["IdAuthor1", "IdAuthor2", "Score"], sortedl)

def new_read_results_euc(dictio_of_users, dictio_of_btcs):
	print("New user extraction...")
	tic = time.time()
	num_users = len(dictio_of_users)
	lst_users = list(dictio_of_users.keys())
	lst = []
	res2 = np.memmap('btc_files/btc_dis_score.dat', dtype=np.uint32, mode ='r', shape=(num_users, num_users))
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
	gen_csv_from_tuples("btc_files/new_results_btc_dis.csv", ["IdAuthor1", "IdAuthor2", "Score"], sortedl)

	lst = []
	res3 = np.memmap('btc_files/btc_euc_score.dat', dtype=float, mode ='r', shape=(num_users, num_users))
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
	gen_csv_from_tuples("btc_files/new_results_btc_euc.csv", ["IdAuthor1", "IdAuthor2", "Score"], sortedl)


def gen_btc_values():

	lst_btcs = read_csv_list("btc_files/btc_count.csv")[1:]
	print("Lenght btc Count: %d" %(len(lst_btcs)))
	lst_btcs = sorted(lst_btcs, key=lambda x: x[1], reverse=True)
	print(lst_btcs[:3])
	divisions = int(math.ceil(float(len(lst_btcs)) / 254.0))
	dictio = {}
	for i in range(0, 254):
		start = i * divisions
		end = (i + 1) * divisions
		for elem in lst_btcs[start:end]:
			dictio[elem[0]] = (i + 1)
	#print(dictio)
	return dictio



def combin(n,r):
	return int(np.math.factorial(n) / (np.math.factorial(r) * np.math.factorial(n - r)))


import requests
import datetime

def get_first_seen(btc_address):
	url='https://blockchain.info/q/addressfirstseen/'+str(btc_address)
	response=requests.get(url)
	if response.status_code==200:
		try:
			timestamp=int(reponse.content)
			return datetime.fromtimestamp(timestamp)
		except:
			return None
	return None

def get_valid_btc_addresses(dictio_of_btcs):
	tic = time.time()
	num_requests = 0
	results = []
	status.create_numbar(100, len(dictio_of_btcs))
	for indi, i in enumerate(dictio_of_btcs.keys()):
		status.update_numbar(indi, len(dictio_of_btcs))
		appears = 1 if (not (get_first_seen(i) is None)) else 0 
		results.append((i, appears))
		time.sleep(1)
		
	status.end_numbar()
	gen_csv_from_tuples("btc_files/valid_btcs.csv", ["BTC", "Valid"], results)


def combin(n,r):
	return int(np.math.factorial(n) / (np.math.factorial(r) * np.math.factorial(n - r)))

def basic_implementation(dictio_of_users, dictio_of_values, n):
	scores = {}
	dictio_of_users = { k:dictio_of_users[k] for k in list(dictio_of_users.keys())[:n]}
	lst_users = list(dictio_of_users.keys())
	tic = time.time()
	for ind1, user1 in enumerate(lst_users):
		for ind2, user2 in enumerate(lst_users[ind1:]):
			score = 0
			for btc1 in dictio_of_users[user1]:
				for btc2 in dictio_of_users[user2]:
					if btc1 == btc2:
						score += dictio_of_values[btc1] ** 2
			scores[user1 + user2] = score
	toc = time.time()
	dif = toc - tic
	print("TIME BASIC: ", dif)
	return dif


def vector_implementation(dictio_of_users,  dictio_of_values, n):
	scores = {}
	dictio_of_users = { k:dictio_of_users[k] for k in list(dictio_of_users.keys())[:n]}
	lst_users = list(dictio_of_users.keys())

	dictio_of_btcs = {}
	for k, v in dictio_of_users.items():
		for i in v:
			if not i in dictio_of_btcs.keys():
				dictio_of_btcs[i] = []
			dictio_of_btcs[i].append(k)
	dictio_of_btcs = {k:i for i, k in enumerate(dictio_of_btcs.keys())}
	num_users = len(lst_users)
	num_btcs = len(dictio_of_btcs)
	matrix = np.zeros((num_users, num_btcs))

	tic = time.time()
	for ind, user in enumerate(dictio_of_users.keys()):
		base = np.zeros((num_btcs, 1))
		for btc in dictio_of_users[user]:
			base[dictio_of_btcs[btc]] = dictio_of_values[btc]
		base = np.squeeze(base)
		matrix[ind] = base[:]

	toc = time.time()
	for i1 in range(num_users):
		v1 = matrix[i1]
		for i2 in range(i1 + 1, num_users):
			v2 = matrix[i2]
			dis_score = np.dot(v1, v2)
			#res2[i1][i2] = euc_score
			scores[i1, i2] = dis_score

	tuc = time.time()
	print("TIME ADVANCED: ")
	print("Vector Generation\t", toc - tic)
	print("Calculations\t\t", tuc - toc)
	print("Total\t\t\t", tuc - tic)
	return toc - tic, tuc - toc, tuc - tic 


def random_string(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def gen_random_dictio_of_users(size_users, size_ips):
	dictio_of_users = {}
	dictio_of_ips = {}
	for useri in range(size_users):
		key = random_string(10)
		dictio_of_users[key] = []
	for ipj in range(size_ips):
		key = random_string(10)
		dictio_of_ips[key] = []

	lst_ips = list(dictio_of_ips.keys())
	#print(len(lst_ips), size_ips)
	for i in dictio_of_users.keys():
		#print(i, i in dictio_of_users.keys())
		num = random.randint(1, size_ips)
		for j in range(num):
			order = random.randint(0, size_ips - 1)
			dictio_of_users[i].append(lst_ips[order])
			dictio_of_ips[lst_ips[order]].append(i)
	return dictio_of_users, dictio_of_ips

def gen_dictio_of_values(dictio_of_ips):
	lst_btcs = [(k, len(v)) for k, v in dictio_of_ips.items()]
	lst_btcs = sorted(lst_btcs, key= lambda x: x[1], reverse=True )
	divisions = int(math.ceil(float(len(lst_btcs)) / 254.0))
	dictio = {}
	for i in range(0, 254):
		start = i * divisions
		end = (i + 1) * divisions
		for elem in lst_btcs[start:end]:
			dictio[elem[0]] = (i + 1)
	return dictio
def do_test(size_users, size_ips, filename):

	print("[-] Generating dictionaries...")
	dictio_of_users, dictio_of_btcs = gen_random_dictio_of_users(size_users, size_ips)
	dictio_of_values = gen_dictio_of_values(dictio_of_btcs)
	print("[OK] Generating dictionaries...")
	lst_results = []
	for i in range(50,1001, 50):
		a = basic_implementation(dictio_of_users, dictio_of_values, i)
		b, c, d = vector_implementation(dictio_of_users, dictio_of_values, i)
		tup = (i, combin(i, 2) ,a, b, c, d)
		print(tup)
		lst_results.append(tup)
	gen_csv_from_tuples("test_result/" + filename, ["Elements", "Combinations", "Basic", "Vector Generation", "Vector Calculation", "Vector Total"], lst_results)



def main():
	# size_users = 1000
	# for size_ips in [10, 100, 250, 500, 1000]:
	# 	print("size_ips", size_ips)
	# 	do_test(size_users, size_ips, "results_"+str(size_users)+"_"+str(size_ips)+ ".csv")
	#extract_btc_to_usage()
	#extract_user_to_btc_csv()
	dictio_of_users, dictio_of_btcs = generate_clean_dataset()
	#dictio_of_values = gen_btc_values()
	


	#get_valid_btc_addresses(dictio_of_btcs)
	#gen_bin_matrix_of_users(dictio_of_btcs, dictio_of_users)
	#process_btcs_euclidean(dictio_of_users, dictio_of_btcs)
	#gen_new_matrix_of_users(dictio_of_btcs, dictio_of_users, dictio_of_values)
	#process_btcs_distance(dictio_of_users, dictio_of_btcs)
	new_read_results_euc(dictio_of_users, dictio_of_btcs)
	#read_results_euc(dictio_of_users, dictio_of_btcs)
	#read_results_dis(dictio_of_users, dictio_of_btcs)
	
	

if __name__ == "__main__":
	main()
