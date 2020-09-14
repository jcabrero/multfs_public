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


dictio_of_sites = {
	"raidforums.com": 12,
	"antichat.ru": 10,
	"hackforums.net": 0,
	"blackhatworld.com": 8,
	"mpgh.net": 4,
	"garage4hackers.com": 7,
	"greysec.net": 6,
	"stresserforums.net": 5,
	"kernelmode.info": 1,
	"safeskyhacks.com": 13,
	"offensivecommunity.net": 3
}

inverse_dictio_of_sites = {v:k for k,v in dictio_of_sites.items()}
def extract_user_to_link_csv():
	query= """WITH "A" AS (SELECT
  		CAST("Post"."Author" AS text) || '[' || CAST("Post"."Site" AS text) || ']' as "Author",
  		regexp_matches( "Content", '(http[s]?://(?:[a-zA-Z]|[0-9]|[$-\)+-Z^-_@.&+]|[!\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)', 'g') AS "link"
  		FROM "Post" WHERE "Content" ~ '(http[s]?://(?:[a-zA-Z]|[0-9]|[$-\)+-Z^-_@.&+]|[!\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)'),
		"B" AS (SELECT "Author", lower("link"[1]) as "link", count(*) as "repetitions" FROM "A" GROUP BY "Author", "link" )
		SELECT "B"."Author",
		string_agg("B"."link", ', ') as "reps" 
		FROM "B" GROUP BY "B"."Author";"""	
	rows = make_query(query)
	#rows = [row[:1] + tuple([x for x in row[1].split(", ")],) for row in rows if row[0] != -1]
	rows = [row[:1] + tuple([x for x in row[1].split(", ")],) for row in rows if row[0] != -1]
	# #print(rows)
	# for row in range(len(rows)):
	# 	#print(type(rows[row]), len(rows[row]))
	# 	for col in range(1, len(rows[row])):
	# 		#print(row, col, rows[row][col])
	# 		if rows[row][col][-1] == '.':
	# 			#print("Changed: %s by %s" % (rows[row][col], rows[row][col][10:]))
	# 			print("CHANGED")
	# 			rows[row][col] = rows[row][col][:-1]
				

	# for row in range(len(rows)):
	# 	rows[row] = (rows[row][0],) + tuple(set(rows[row][1:]))
	print (len(rows))
	gen_csv_from_tuples("link_files/user_to_link3.csv", ["IdAuthor", "link"], rows)

def extract_link_to_usage():
	query= """WITH "A" AS (SELECT
  		regexp_matches( "Content", '(http[s]?://(?:[a-zA-Z]|[0-9]|[$-\)+-Z^-_@.&+]|[!\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)', 'g') AS "link"
  		FROM "Post" WHERE "Content" ~ '(http[s]?://(?:[a-zA-Z]|[0-9]|[$-\)+-Z^-_@.&+]|[!\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)'),
		"B" AS (SELECT lower("link"[1]) as "link", count(*) as "repetitions" FROM "A" GROUP BY "link" )
		SELECT "link", "repetitions" FROM "B";"""
	rows = make_query(query)

	print (len(rows))
	gen_csv_from_tuples("link_files/link_count.csv", ["link", "Reps"], rows)	

def gen_vector_for_user(lst1, dictio):
	base = np.zeros((len(dictio),1), dtype=int)
	for i in range(0,len(lst1),2):
		#scorei = lst1[i]
		link = lst1[i+1]
		base[dictio[link]] = 1
		#base[dictio[link]] = scorei
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

def modify_link(elem):
	lst = [elem[0], elem[1]]
	for i in elem[2:]:
		j = i.split(":")
		lst.append((j[0], j[1]))
	return tuple(lst)



def gen_new_dataset():
	global global_lst
	#lst = read_csv_list("user_links.csv")[1:]
	#lst = read_csv_list("link_files/user_to_link.csv")[1:]
	lst = read_csv_list("link2files/user_to_link2.csv")[1:]
	print("Length of the Dataset: %d" % (len(lst)))
	#pool = mp.Pool(processes=16)
	#lst = pool.map(modify_link, lst)
	#clean_dataset(lst)
	#global_lst = sorted(lst, key=lambda x: len(x), reverse=True)
	return lst
	#gen_csv_from_tuples("please_work2.csv", ["Author", "Username", "Site", "IdPost", "link"], lst)

def get_different_links(lst):
	seti = set()
	dictio = {}
	for i in lst:
		for j in i[2:]:
			seti.add(j[1])
	dictio = {i: indi for indi, i in enumerate(list(seti))}
	return dictio

def get_different_links2(lst, default="list"):
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

def gen_dictio_of_links(lst):
	dictio = {}
	for i in lst:
		key = i[0]
		for j in i[1:]:
			if j in dictio.keys():
				dictio[j].append(key)
			else:
				dictio[j] = [key]
	return dictio

def clean_users_from_dictios(list_link, dictio_of_users, dictio_of_links):
	old_len_u, old_len_link = len(dictio_of_users), len(dictio_of_links)
	#print("links removed: ", len(list_link) )
	for link in list_link:
		for i in dictio_of_links[link]:
			dictio_of_users[i].remove(link)
		ret2 = dictio_of_links.pop(link, None)
		if ret2 is None:
			print("THERE IS AN ERROR: ", link)
	# Update the list of users
	users_removed = [k for k, v in dictio_of_users.items() if v == []]
	#print("Users removed: ", len(users_removed))
	for user in users_removed:
		ret = dictio_of_users.pop(user, None)
		if ret is None:
			print("ERROR")
	row_format ="{:>15}" * 4
	print("-" * 15 * 4)
	print(row_format.format("Original links", "Removed links", "New links", "Percentage"))
	print(row_format.format("%d"% (old_len_link), "%d"%(len(list_link)), "%d"%(len(dictio_of_links)), "%f" %(len(dictio_of_links)/old_len_link)))
	print(row_format.format("Original Users", "Removed Users", "New Users", "Percentage"))
	print(row_format.format("%d"% (old_len_u), "%d"%(len(users_removed)), "%d"%(len(dictio_of_users)), "%f" %(len(dictio_of_users)/old_len_u)))
	print("-" * 15 * 4)
	#print(len(dictio_of_users)/old_len_u, len(dictio_of_links)/old_len_link, 
		#len(dictio_of_users), len(dictio_of_links))
	return dictio_of_users, dictio_of_links

def clean_users_from_dictios2(list_users, dictio_of_users, dictio_of_links):
	old_len_u, old_len_link = len(dictio_of_users), len(dictio_of_links)
	#print("links removed: ", len(list_link) )
	for user in list_users:
		for link in dictio_of_users[user]:
			dictio_of_links[link].remove(user)
		ret2 = dictio_of_users.pop(user, None)
		

	# Update the list of users
	links_removed = [k for k, v in dictio_of_links.items() if v == []]
	#print("Users removed: ", len(users_removed))
	for link in links_removed:
		ret = dictio_of_links.pop(link, None)
		if ret is None:
			print("ERROR")
	row_format ="{:>15}" * 4
	print("-" * 15 * 4)
	print(row_format.format("Original links", "Removed links", "New links", "Percentage"))
	print(row_format.format("%d"% (old_len_link), "%d"%(len(links_removed)), "%d"%(len(dictio_of_links)), "%f" %(len(dictio_of_links)/old_len_link)))
	print(row_format.format("Original Users", "Removed Users", "New Users", "Percentage"))
	print(row_format.format("%d"% (old_len_u), "%d"%(len(list_users)), "%d"%(len(dictio_of_users)), "%f" %(len(dictio_of_users)/old_len_u)))
	print("-" * 15 * 4)
	#print(len(dictio_of_users)/old_len_u, len(dictio_of_links)/old_len_link, 
		#len(dictio_of_users), len(dictio_of_links))
	return dictio_of_users, dictio_of_links


def clean_own_links(dictio_of_users, dictio_of_links):
	old_len_u, old_len_link = len(dictio_of_users), len(dictio_of_links)
	lst_links = list(dictio_of_sites.keys())
	#inverse_dictio_of_sites
	for user in list_users:
		site = int(user[user.index('[') + 1:-1])
		for link in dictio_of_users[user]:
			dictio_of_links[link].remove(user)
		ret2 = dictio_of_users.pop(user, None)
		

	# Update the list of users
	links_removed = [k for k, v in dictio_of_links.items() if v == []]
	#print("Users removed: ", len(users_removed))
	for link in links_removed:
		ret = dictio_of_links.pop(link, None)
		if ret is None:
			print("ERROR")
	row_format ="{:>15}" * 4
	print("-" * 15 * 4)
	print(row_format.format("Original links", "Removed links", "New links", "Percentage"))
	print(row_format.format("%d"% (old_len_link), "%d"%(len(links_removed)), "%d"%(len(dictio_of_links)), "%f" %(len(dictio_of_links)/old_len_link)))
	print(row_format.format("Original Users", "Removed Users", "New Users", "Percentage"))
	print(row_format.format("%d"% (old_len_u), "%d"%(len(list_users)), "%d"%(len(dictio_of_users)), "%f" %(len(dictio_of_users)/old_len_u)))
	print("-" * 15 * 4)
	#print(len(dictio_of_users)/old_len_u, len(dictio_of_links)/old_len_link, 
		#len(dictio_of_users), len(dictio_of_links))
	return dictio_of_users, dictio_of_links

def clean_dataset(dictio_of_users, dictio_of_links):
	# Remove links with 1 appearance.
	oneapp = [k for k,v in dictio_of_links.items() if len(v) == 1]
	while(len(oneapp) > 0):
		print("Removing links that appear once...")
		dictio_of_users, dictio_of_links = clean_users_from_dictios(oneapp, dictio_of_users, dictio_of_links)
		local_links = []
		print("Removing links from sites...")
		for link in dictio_of_links.keys():
			is_link_internal = False
			for site in dictio_of_sites.keys():
				if site in link:
					is_link_internal = True
					break
			if is_link_internal == True:
				local_links.append(link)
		dictio_of_users, dictio_of_links = clean_users_from_dictios(local_links, dictio_of_users, dictio_of_links)
		print("Removing links that appear more than 7 times...")
		multilink = [k for k,v in dictio_of_links.items() if len(v) > 7]
		dictio_of_users, dictio_of_links = clean_users_from_dictios(multilink, dictio_of_users, dictio_of_links)
		print("Removing users that have less than 12 links...")
		atleast10 = [k for k,v in dictio_of_users.items() if len(v) < 12]
		dictio_of_users, dictio_of_links = clean_users_from_dictios2(atleast10, dictio_of_users, dictio_of_links)
		oneapp = [k for k,v in dictio_of_links.items() if len(v) == 1]
	return dictio_of_users, dictio_of_links



def gen_bin_matrix_of_users(dictio_of_links, dictio_of_users):
	num_users = len(dictio_of_users)
	num_links = len(dictio_of_links)
	# Transform dictionary to indexes
	for indk, link in enumerate(dictio_of_links.keys()):
		dictio_of_links[link] = indk
	#Transform users to matrices.
	print("ESTIMATED SIZE OF MATRIX: %f GB" % (num_users * num_links * 1 / (1024 ** 3)))
	matrix_map = np.memmap('link_files/link_matrix_map.dat', dtype=np.uint8, mode ='w+', shape=(num_users, num_links))
	status.create_numbar(100, num_users)
	for ind, user in enumerate(dictio_of_users.keys()):
		status.update_numbar(ind, num_users)
		base = np.zeros((num_links,1), dtype=np.uint8)
		for link in dictio_of_users[user]:
			base[dictio_of_links[link]] = 1
		base = np.squeeze(base)
		matrix_map[ind] = base[:]
	status.end_numbar()
	print("Flushing...")
	matrix_map.flush()
	
def gen_new_matrix_of_users(dictio_of_links, dictio_of_users, dictio_of_values):
	num_users = len(dictio_of_users)
	num_links = len(dictio_of_links)
	# Transform dictionary to indexes
	for indk, link in enumerate(dictio_of_links.keys()):
		dictio_of_links[link] = indk
	#Transform users to matrices.
	print("ESTIMATED SIZE OF MATRIX: %f GB" % (num_users * num_links * 1 / (1024 ** 3)))
	matrix_map = np.memmap('link_files/link_matrix_map.dat', dtype=np.uint8, mode ='w+', shape=(num_users, num_links))
	status.create_numbar(100, num_users)
	for ind, user in enumerate(dictio_of_users.keys()):
		status.update_numbar(ind, num_users)
		base = np.zeros((num_links,1), dtype=np.uint8)
		for link in dictio_of_users[user]:
			x = dictio_of_values[link]
			base[dictio_of_links[link]] = x
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
	dictio_of_links = gen_dictio_of_links(lst_users)

	lst_links = [(k, len(v)) for k, v in dictio_of_links.items()]
	dictio_of_values = gen_link_values(lst_links)

	dictio_of_users, dictio_of_links = clean_dataset(dictio_of_users,dictio_of_links)
	print("SECONDS: %f" %(time.time() - tic))
	return dictio_of_users, dictio_of_links, dictio_of_values

def process_links_euclidean(dictio_of_users, dictio_of_links):
	tic = time.time()
	num_users = len(dictio_of_users)
	num_links = len(dictio_of_links)

	matrix_map = np.memmap('link_files/link_matrix_map.dat', dtype=np.uint8,  shape=(num_users, num_links))
	#matrix_map = np.array(matrix_map)

	print(matrix_map.shape)
	print("ESTIMATED SIZE OF MATRIX: %f GB" % (num_users * num_users * 4 / (1024 ** 3)))
	status.create_numbar(100, num_users)
	#res = np.dot(matrix_map, matrix_map.T)
	res2 = np.memmap('link_files/link_euc_score.dat', dtype=float ,mode ='w+', shape=(num_users, num_users))
	#res3 = np.memmap('link_files/link_dis_score.dat', dtype=np.uint32 ,mode ='w+', shape=(num_users, num_users))
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

def process_links_distance(dictio_of_users, dictio_of_links):
	tic = time.time()
	num_users = len(dictio_of_users)
	num_links = len(dictio_of_links)

	matrix_map = np.memmap('link_files/link_matrix_map.dat', dtype=np.uint8,  shape=(num_users, num_links))
	#matrix_map = np.array(matrix_map)

	print(matrix_map.shape)
	print("ESTIMATED SIZE OF MATRIX: %f GB" % (num_users * num_users * 4 / (1024 ** 3)))
	status.create_numbar(100, num_users)
	#res = np.dot(matrix_map, matrix_map.T)
	#res2 = np.memmap('link_files/link_euc_score.dat', dtype=float ,mode ='w+', shape=(num_users, num_users))
	res3 = np.memmap('link_files/link_dis_score.dat', dtype=np.uint32 ,mode ='w+', shape=(num_users, num_users))
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


def read_results_dis(dictio_of_users, dictio_of_links):
	tic = time.time()
	
	num_users = len(dictio_of_users)
	lst_users = list(dictio_of_users.keys())
	lst = []
	res2 = np.memmap('link_files/link_dis_score.dat', dtype=np.uint32, mode ='r', shape=(num_users, num_users))
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
	gen_csv_from_tuples("link_files/results_link_dis.csv", ["IdAuthor1", "IdAuthor2", "Score"], sortedl)

def read_results_euc(dictio_of_users, dictio_of_links):
	tic = time.time()
	
	num_users = len(dictio_of_users)
	lst_users = list(dictio_of_users.keys())
	lst = []
	res2 = np.memmap('link_files/link_euc_score.dat', dtype=float, mode ='r', shape=(num_users, num_users))
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
	gen_csv_from_tuples("link_files/results_link_euc.csv", ["IdAuthor1", "IdAuthor2", "Score"], sortedl)

def gen_link_values(lst_links):

	#lst_links = read_csv_list("link_files/link_count.csv")[1:]
	print("Lenght link Count: %d" %(len(lst_links)))
	lst_links = sorted(lst_links, key=lambda x: x[1], reverse=True)
	print(lst_links[:3])
	divisions = int(math.ceil(float(len(lst_links)) / 254.0))
	dictio = {}
	for i in range(0, 254):
		start = i * divisions
		end = (i + 1) * divisions
		for elem in lst_links[start:end]:
			dictio[elem[0]] = (i + 1)
	#print(dictio)
	return dictio

def new_read_results_euc(dictio_of_users, dictio_of_links):
	print("New user extraction...")
	tic = time.time()
	num_users = len(dictio_of_users)
	lst_users = list(dictio_of_users.keys())
	lst = []
	res2 = np.memmap('link_files/link_dis_score.dat', dtype=np.uint32, mode ='r', shape=(num_users, num_users))
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
	gen_csv_from_tuples("link_files/new_results_link_dis.csv", ["IdAuthor1", "IdAuthor2", "Score"], sortedl)

	lst = []
	res3 = np.memmap('link_files/link_euc_score.dat', dtype=float, mode ='r', shape=(num_users, num_users))
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
	gen_csv_from_tuples("link_files/new_results_link_euc.csv", ["IdAuthor1", "IdAuthor2", "Score"], sortedl)


def check_links(dictio_of_links, dictio_of_values):
	totsum = 0
	for i in dictio_of_links.keys():
		if not i in dictio_of_values.keys():
			totsum += 1
			print("ERROR: %s" %(i))
	if totsum == 0:
		print("VERIFIED DATASET")

def combin(n,r):
	return int(np.math.factorial(n) / (np.math.factorial(r) * np.math.factorial(n - r)))

def store_dictio_of_users(dictio_of_users, filename2 = "link_files/clean_user_to_link.csv"):
	lst2 = []
	for user in dictio_of_users.keys():
		lst2.append((user,) + tuple(dictio_of_users[user]))
	gen_csv_from_tuples(filename2, ["IdAuthor", "link"], lst2)

def main():
	#extract_link_to_usage()
	#extract_user_to_link_csv()
	dictio_of_users, dictio_of_links, dictio_of_values = generate_clean_dataset()
	# dictio_of_values = gen_link_values()
	# store_dictio_of_users(dictio_of_users, filename2 = "link_files/clean_user_to_link2.csv")
	# check_links(dictio_of_links, dictio_of_values)
	#gen_bin_matrix_of_users(dictio_of_links, dictio_of_users)
	#process_links_euclidean(dictio_of_users, dictio_of_links)
	# gen_new_matrix_of_users(dictio_of_links, dictio_of_users, dictio_of_values)
	# process_links_distance(dictio_of_users, dictio_of_links)
	new_read_results_euc(dictio_of_users, dictio_of_links)
	#read_results_euc(dictio_of_users, dictio_of_links)
	# read_results_dis(dictio_of_users, dictio_of_links)

	

if __name__ == "__main__":
	main()
