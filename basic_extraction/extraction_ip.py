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


def extract_user_to_ip_csv():
	query= """WITH "A" AS (SELECT
		CAST("Post"."Author" AS text) || '[' || CAST("Post"."Site" AS text) || ']' as "Author", 
		regexp_matches( "Content", '(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)', 'g') AS "ip"
		FROM "Post"	WHERE "Content" ~ '(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'),
		"B" AS (SELECT "Author", "ip", count(*) as "repetitions" FROM "A" GROUP BY "Author", "ip" )
		SELECT "B"."Author",
		string_agg(CAST("B"."repetitions" AS text) || ':' || "B"."ip"[1] || '.' ||"B"."ip"[2] || '.' ||"B"."ip"[3]|| '.' ||"B"."ip"[4], ', ') as "reps" 
		FROM "B" GROUP BY "B"."Author";"""
	query= """WITH "A" AS (SELECT
		CAST("Post"."Author" AS text) || '[' || CAST("Post"."Site" AS text) || ']' as "Author",
		regexp_matches( "Content", '(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)', 'g') AS "ip"
		FROM "Post"	WHERE "Content" ~ '(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'),
		"B" AS (SELECT "Author", "ip", count(*) as "repetitions" FROM "A" GROUP BY "Author", "ip" )
		SELECT "B"."Author",
		string_agg("B"."ip"[1] || '.' ||"B"."ip"[2] || '.' ||"B"."ip"[3]|| '.' ||"B"."ip"[4], ', ') as "reps" 
		FROM "B" GROUP BY "B"."Author";"""
	rows = make_query(query)
	#print(rows)
	print(len(rows))
	rows = [row[:1] + tuple([x for x in row[1].split(", ")],) for row in rows if row[0] != -1]
	print (len(rows))
	gen_csv_from_tuples("ip_files/user_to_ip.csv", ["IdAuthor", "IP"], rows)

def extract_ip_to_usage():
	query= """WITH "A" AS (SELECT
		regexp_matches( "Content", '(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)', 'g') AS "ip"
		FROM "Post"	WHERE "Content" ~ '(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'),
		"B" AS (SELECT "ip", count(*) as "repetitions" FROM "A" GROUP BY "ip" )
		SELECT "B"."ip"[1] || '.' ||"B"."ip"[2] || '.' ||"B"."ip"[3]|| '.' ||"B"."ip"[4] as "ip", "B"."repetitions" FROM "B";"""
	rows = make_query(query)
	#print(rows)
	print(len(rows))
	#rows = [row[:1] + tuple([x for x in row[1].split(", ")],) for row in rows if row[0] != -1]
	print (len(rows))
	gen_csv_from_tuples("ip_files/ip_count.csv", ["IP", "Reps"], rows)	

def gen_vector_for_user(lst1, dictio):
	base = np.zeros((len(dictio),1), dtype=int)
	for i in range(0,len(lst1),2):
		#scorei = lst1[i]
		ip = lst1[i+1]
		base[dictio[ip]] = 1
		#base[dictio[ip]] = scorei
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

def modify_ip(elem):
	lst = [elem[0], elem[1]]
	for i in elem[2:]:
		j = i.split(":")
		lst.append((j[0], j[1]))
	return tuple(lst)



def gen_new_dataset():
	global global_lst
	#lst = read_csv_list("user_ips.csv")[1:]
	lst = read_csv_list("ip_files/user_to_ip.csv")[1:]
	print("Length of the Dataset: %d" % (len(lst)))
	#pool = mp.Pool(processes=16)
	#lst = pool.map(modify_ip, lst)
	#clean_dataset(lst)
	#global_lst = sorted(lst, key=lambda x: len(x), reverse=True)
	return lst
	#gen_csv_from_tuples("please_work2.csv", ["Author", "Username", "Site", "IdPost", "IP"], lst)

def get_different_ips(lst):
	seti = set()
	dictio = {}
	for i in lst:
		for j in i[2:]:
			seti.add(j[1])
	dictio = {i: indi for indi, i in enumerate(list(seti))}
	return dictio

def get_different_ips2(lst, default="list"):
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

def gen_dictio_of_ips(lst):
	dictio = {}
	for i in lst:
		key = i[0]
		for j in i[1:]:
			if j in dictio.keys():
				dictio[j].append(key)
			else:
				dictio[j] = [key]
	return dictio

def clean_users_from_dictios(list_ip, dictio_of_users, dictio_of_ips):
	old_len_u, old_len_ip = len(dictio_of_users), len(dictio_of_ips)
	#print("IPs removed: ", len(list_ip) )
	for ip in list_ip:
		for i in dictio_of_ips[ip]:
			dictio_of_users[i].remove(ip)
		ret2 = dictio_of_ips.pop(ip, None)
		if ret2 is None:
			print("THERE IS AN ERROR: ", ip)
	# Update the list of users
	users_removed = [k for k, v in dictio_of_users.items() if v == []]
	#print("Users removed: ", len(users_removed))
	for user in users_removed:
		ret = dictio_of_users.pop(user, None)
		if ret is None:
			print("ERROR")
	row_format ="{:>15}" * 4
	print("-" * 15 * 4)
	print(row_format.format("Original IPs", "Removed IPs", "New IPs", "Percentage"))
	print(row_format.format("%d"% (old_len_ip), "%d"%(len(list_ip)), "%d"%(len(dictio_of_ips)), "%f" %(len(dictio_of_ips)/old_len_ip)))
	print(row_format.format("Original Users", "Removed Users", "New Users", "Percentage"))
	print(row_format.format("%d"% (old_len_u), "%d"%(len(users_removed)), "%d"%(len(dictio_of_users)), "%f" %(len(dictio_of_users)/old_len_u)))
	print("-" * 15 * 4)
	#print(len(dictio_of_users)/old_len_u, len(dictio_of_ips)/old_len_ip, 
		#len(dictio_of_users), len(dictio_of_ips))
	return dictio_of_users, dictio_of_ips

def clean_users_from_dictios2(list_users, dictio_of_users, dictio_of_ips):
	old_len_u, old_len_ip = len(dictio_of_users), len(dictio_of_ips)
	#print("IPs removed: ", len(list_ip) )
	for user in list_users:
		for ip in dictio_of_users[user]:
			dictio_of_ips[ip].remove(user)
		ret2 = dictio_of_users.pop(user, None)
		

	# Update the list of users
	ips_removed = [k for k, v in dictio_of_ips.items() if v == []]
	#print("Users removed: ", len(users_removed))
	for ip in ips_removed:
		ret = dictio_of_ips.pop(ip, None)
		if ret is None:
			print("ERROR")
	row_format ="{:>15}" * 4
	print("-" * 15 * 4)
	print(row_format.format("Original IPs", "Removed IPs", "New IPs", "Percentage"))
	print(row_format.format("%d"% (old_len_ip), "%d"%(len(ips_removed)), "%d"%(len(dictio_of_ips)), "%f" %(len(dictio_of_ips)/old_len_ip)))
	print(row_format.format("Original Users", "Removed Users", "New Users", "Percentage"))
	print(row_format.format("%d"% (old_len_u), "%d"%(len(list_users)), "%d"%(len(dictio_of_users)), "%f" %(len(dictio_of_users)/old_len_u)))
	print("-" * 15 * 4)
	#print(len(dictio_of_users)/old_len_u, len(dictio_of_ips)/old_len_ip, 
		#len(dictio_of_users), len(dictio_of_ips))
	return dictio_of_users, dictio_of_ips


def clean_dataset(dictio_of_users, dictio_of_ips):
	localips = []
	for ip in dictio_of_ips.keys():
		if re.search(r'192\.168\.\d{1,3}\.\d{1,3}', 
			ip,  re.IGNORECASE | re.DOTALL | re.VERBOSE | re.MULTILINE):
			localips.append(ip)
		elif re.search(r'172\.16\.\d{1,3}\.\d{1,3}', 
			ip,  re.IGNORECASE | re.DOTALL | re.VERBOSE | re.MULTILINE):
			localips.append(ip)
		elif re.search(r'10\.\d{1,3}\.\d{1,3}\.\d{1,3}', 
			ip,  re.IGNORECASE | re.DOTALL | re.VERBOSE | re.MULTILINE):
			localips.append(ip)
		elif ip == '127.0.0.1':
			localips.append(ip)
	print("Removing non-public ips...")
	dictio_of_users, dictio_of_ips = clean_users_from_dictios(localips, dictio_of_users, dictio_of_ips)
	# Remove IPs with 1 appearance.
	oneapp = [k for k,v in dictio_of_ips.items() if len(v) == 1]
	while(len(oneapp) > 0):
		print("Removing IPs that appear once...")
		dictio_of_users, dictio_of_ips = clean_users_from_dictios(oneapp, dictio_of_users, dictio_of_ips)
		print("Removing IPs that appear more than 12 times...")
		multiip = [k for k,v in dictio_of_ips.items() if len(v) > 12]
		dictio_of_users, dictio_of_ips = clean_users_from_dictios(multiip, dictio_of_users, dictio_of_ips)
		print("Removing users that have less than 5 IPs...")
		atleast10 = [k for k,v in dictio_of_users.items() if len(v) < 5]
		#print(len(atleast10))
		dictio_of_users, dictio_of_ips = clean_users_from_dictios2(atleast10, dictio_of_users, dictio_of_ips)
		oneapp = [k for k,v in dictio_of_ips.items() if len(v) == 1]
	return dictio_of_users, dictio_of_ips



def gen_bin_matrix_of_users(dictio_of_ips, dictio_of_users):
	num_users = len(dictio_of_users)
	num_ips = len(dictio_of_ips)
	# Transform dictionary to indexes
	for indk, ip in enumerate(dictio_of_ips.keys()):
		dictio_of_ips[ip] = indk
	#Transform users to matrices.
	print("ESTIMATED SIZE OF MATRIX: %f GB" % (num_users * num_ips * 1 / (1024 ** 3)))
	matrix_map = np.memmap('ip_files/ip_matrix_map.dat', dtype=np.uint8, mode ='w+', shape=(num_users, num_ips))
	status.create_numbar(100, num_users)
	for ind, user in enumerate(dictio_of_users.keys()):
		status.update_numbar(ind, num_users)
		base = np.zeros((num_ips,1), dtype=np.uint8)
		for ip in dictio_of_users[user]:
			base[dictio_of_ips[ip]] = 1
		base = np.squeeze(base)
		matrix_map[ind] = base[:]
	status.end_numbar()
	print("Flushing...")
	matrix_map.flush()

def gen_new_matrix_of_users(dictio_of_ips, dictio_of_users, dictio_of_values):
	num_users = len(dictio_of_users)
	num_ips = len(dictio_of_ips)
	# Transform dictionary to indexes
	for indk, ip in enumerate(dictio_of_ips.keys()):
		dictio_of_ips[ip] = indk
	#Transform users to matrices.
	print("ESTIMATED SIZE OF MATRIX: %f GB" % (num_users * num_ips * 1 / (1024 ** 3)))
	matrix_map = np.memmap('ip_files/ip_matrix_map.dat', dtype=np.uint8, mode ='w+', shape=(num_users, num_ips))
	status.create_numbar(100, num_users)
	for ind, user in enumerate(dictio_of_users.keys()):
		status.update_numbar(ind, num_users)
		base = np.zeros((num_ips,1), dtype=np.uint8)
		for ip in dictio_of_users[user]:
			base[dictio_of_ips[ip]] = dictio_of_values[ip]
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
	dictio_of_ips = gen_dictio_of_ips(lst_users)
	dictio_of_users, dictio_of_ips = clean_dataset(dictio_of_users,dictio_of_ips)
	print("SECONDS: %f" %(time.time() - tic))
	return dictio_of_users, dictio_of_ips

def process_ips_euclidean(dictio_of_users, dictio_of_ips):
	tic = time.time()
	num_users = len(dictio_of_users)
	num_ips = len(dictio_of_ips)

	matrix_map = np.memmap('ip_files/ip_matrix_map.dat', dtype=np.uint8,  shape=(num_users, num_ips))
	#matrix_map = np.array(matrix_map)

	print(matrix_map.shape)
	print("ESTIMATED SIZE OF MATRIX: %f GB" % (num_users * num_users * 4 / (1024 ** 3)))
	status.create_numbar(100, num_users)
	#res = np.dot(matrix_map, matrix_map.T)
	res2 = np.memmap('ip_files/ip_euc_score.dat', dtype=float ,mode ='w+', shape=(num_users, num_users))
	#res3 = np.memmap('ip_files/ip_dis_score.dat', dtype=np.uint32 ,mode ='w+', shape=(num_users, num_users))
	for i1 in range(num_users):
		status.update_numbar(i1, num_users)
		v1 = np.array(matrix_map[i1], dtype=np.uint32)
		for i2 in range(i1 + 1, num_users):
			v2 = np.array(matrix_map[i2], dtype=np.uint32)
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

def process_ips_distance(dictio_of_users, dictio_of_ips):
	tic = time.time()
	num_users = len(dictio_of_users)
	num_ips = len(dictio_of_ips)

	matrix_map = np.memmap('ip_files/ip_matrix_map.dat', dtype=np.uint8,  shape=(num_users, num_ips))
	#matrix_map = np.array(matrix_map)

	print(matrix_map.shape)
	print("ESTIMATED SIZE OF MATRIX: %f GB" % (num_users * num_users * 4 / (1024 ** 3)))
	status.create_numbar(100, num_users)
	#res = np.dot(matrix_map, matrix_map.T)
	#res2 = np.memmap('ip_files/ip_euc_score.dat', dtype=float ,mode ='w+', shape=(num_users, num_users))
	res3 = np.memmap('ip_files/ip_dis_score.dat', dtype=np.uint32 ,mode ='w+', shape=(num_users, num_users))
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


def read_results_dis(dictio_of_users, dictio_of_ips):
	tic = time.time()
	
	num_users = len(dictio_of_users)
	lst_users = list(dictio_of_users.keys())
	lst = []
	res2 = np.memmap('ip_files/ip_dis_score.dat', dtype=np.uint32, mode ='r', shape=(num_users, num_users))
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
	gen_csv_from_tuples("ip_files/results_ip_dis.csv", ["IdAuthor1", "IdAuthor2", "Score"], sortedl)

def read_results_euc(dictio_of_users, dictio_of_ips):
	tic = time.time()
	
	num_users = len(dictio_of_users)
	lst_users = list(dictio_of_users.keys())
	lst = []
	res2 = np.memmap('ip_files/ip_euc_score.dat', dtype=float, mode ='r', shape=(num_users, num_users))
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
	gen_csv_from_tuples("ip_files/results_ip_euc.csv", ["IdAuthor1", "IdAuthor2", "Score"], sortedl)

def gen_ip_values():

	lst_ips = read_csv_list("ip_files/ip_count.csv")[1:]
	lst_ips = sorted(lst_ips, key=lambda x: x[1], reverse=True)
	print(lst_ips[:3])
	divisions = int(math.ceil(float(len(lst_ips)) / 254.0))
	dictio = {}
	for i in range(0, 254):
		start = i * divisions
		end = (i + 1) * divisions
		for elem in lst_ips[start:end]:
			dictio[elem[0]] = (i + 1)
	print(dictio)
	return dictio

def new_read_results_euc(dictio_of_users, dictio_of_ips):
	print("New user extraction...")
	tic = time.time()
	num_users = len(dictio_of_users)
	lst_users = list(dictio_of_users.keys())
	lst = []
	res2 = np.memmap('ip_files/ip_dis_score.dat', dtype=np.uint32, mode ='r', shape=(num_users, num_users))
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
	gen_csv_from_tuples("ip_files/new_results_ip_dis.csv", ["IdAuthor1", "IdAuthor2", "Score"], sortedl)

	lst = []
	res3 = np.memmap('ip_files/ip_euc_score.dat', dtype=float, mode ='r', shape=(num_users, num_users))
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
	gen_csv_from_tuples("ip_files/new_results_ip_euc.csv", ["IdAuthor1", "IdAuthor2", "Score"], sortedl)

def combin(n,r):
	return int(np.math.factorial(n) / (np.math.factorial(r) * np.math.factorial(n - r)))

def main():
	# extract_ip_to_usage()
	# extract_user_to_ip_csv()
	dictio_of_users, dictio_of_ips = generate_clean_dataset()
	# dictio_of_values = gen_ip_values()
	#gen_bin_matrix_of_users(dictio_of_ips, dictio_of_users)
	#process_ips_euclidean(dictio_of_users, dictio_of_ips)
	# gen_new_matrix_of_users(dictio_of_ips, dictio_of_users, dictio_of_values)
	# process_ips_distance(dictio_of_users, dictio_of_ips)
	new_read_results_euc(dictio_of_users, dictio_of_ips)
	# read_results_euc(dictio_of_users, dictio_of_ips)
	# read_results_dis(dictio_of_users, dictio_of_ips)
	# #read_results(dictio_of_users, dictio_of_ips)
	

if __name__ == "__main__":
	main()
