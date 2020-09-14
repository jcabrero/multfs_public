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

def extract_user_to_email_csv():
	query= """WITH "A" AS (SELECT
  		CAST("Post"."Author" AS text) || '[' || CAST("Post"."Site" AS text) || ']' as "Author",
  		regexp_matches( "Content", '(?:(?![*]))([A-Za-z0-9\._%-\)\+]+@[A-Za-z0-9\.-]+[.][A-Za-z]+)', 'g') AS "email"
  		FROM "Post" WHERE "Content" ~ '(?:(?![*]))([A-Za-z0-9\._%-\)\+]+@[A-Za-z0-9\.-]+[.][A-Za-z]+)'),
		"B" AS (SELECT "Author", lower("email"[1]) as "email", count(*) as "repetitions" FROM "A" GROUP BY "Author", "email" )
		SELECT "B"."Author",
		string_agg("B"."email", ', ') as "reps" 
		FROM "B" GROUP BY "B"."Author";"""
	rows = make_query(query)
	rows = [list(row[:1] + tuple([x for x in row[1].split(", ")],)) for row in rows if row[0] != -1]
	#print(rows)
	for row in range(len(rows)):
		#print(type(rows[row]), len(rows[row]))
		for col in range(1, len(rows[row])):
			#print(row, col, rows[row][col])
			if len(rows[row][col]) > len("***LINK***") and rows[row][col][:len("***LINK***")] == "***LINK***":
				#print("Changed: %s by %s" % (rows[row][col], rows[row][col][10:]))
				rows[row][col] = rows[row][col][len("***LINK***"):]
				

	for row in range(len(rows)):
		rows[row] = (rows[row][0],) + tuple(set(rows[row][1:]))
	#print(rows) 

	#print(rows)
	print(len(rows))
	
	print (len(rows))
	gen_csv_from_tuples("email_files/user_to_email.csv", ["IdAuthor", "email"], rows)

def extract_email_to_usage():
	query= """WITH "A" AS (SELECT
  		regexp_matches( "Content", '(?:(?![*]))([A-Za-z0-9\._%-\)\+]+@[A-Za-z0-9\.-]+[.][A-Za-z]+)', 'g') AS "email"
  		FROM "Post" WHERE "Content" ~ '(?:(?![*]))([A-Za-z0-9\._%-\)\+]+@[A-Za-z0-9\.-]+[.][A-Za-z]+)'),
		"B" AS (SELECT lower("email"[1]) as "email", count(*) as "repetitions" FROM "A" GROUP BY "email" )
		SELECT "email", "repetitions" FROM "B";"""
	rows = make_query(query)
	rows = [list(row) for row in rows]
	for i in range(len(rows)):
		if len(rows[i][0]) > len("***LINK***") and rows[i][0][:len("***LINK***")] == "***LINK***":
			#print("Changed: %s by %s" % (rows[row][col], rows[row][col][10:]))
			rows[i][0] = rows[i][0][len("***LINK***"):]
	rows = [tuple(row) for row in rows]
	#print(rows)
	print(len(rows))
	#rows = [row[:1] + tuple([x for x in row[1].split(", ")],) for row in rows if row[0] != -1]
	print (len(rows))
	gen_csv_from_tuples("email_files/email_count.csv", ["email", "Reps"], rows)	

def gen_vector_for_user(lst1, dictio):
	base = np.zeros((len(dictio),1), dtype=int)
	for i in range(0,len(lst1),2):
		#scorei = lst1[i]
		email = lst1[i+1]
		base[dictio[email]] = 1
		#base[dictio[email]] = scorei
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

def modify_email(elem):
	lst = [elem[0], elem[1]]
	for i in elem[2:]:
		j = i.split(":")
		lst.append((j[0], j[1]))
	return tuple(lst)



def gen_new_dataset():
	global global_lst
	#lst = read_csv_list("user_emails.csv")[1:]
	lst = read_csv_list("email_files/user_to_email.csv")[1:]
	print("Length of the Dataset: %d" % (len(lst)))
	#pool = mp.Pool(processes=16)
	#lst = pool.map(modify_email, lst)
	#clean_dataset(lst)
	#global_lst = sorted(lst, key=lambda x: len(x), reverse=True)
	return lst
	#gen_csv_from_tuples("please_work2.csv", ["Author", "Username", "Site", "IdPost", "email"], lst)

def get_different_emails(lst):
	seti = set()
	dictio = {}
	for i in lst:
		for j in i[2:]:
			seti.add(j[1])
	dictio = {i: indi for indi, i in enumerate(list(seti))}
	return dictio

def get_different_emails2(lst, default="list"):
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

def gen_dictio_of_emails(lst):
	dictio = {}
	for i in lst:
		key = i[0]
		for j in i[1:]:
			if j in dictio.keys():
				dictio[j].append(key)
			else:
				dictio[j] = [key]
	return dictio

def clean_users_from_dictios(list_email, dictio_of_users, dictio_of_emails):
	old_len_u, old_len_email = len(dictio_of_users), len(dictio_of_emails)
	#print("emails removed: ", len(list_email) )
	for email in list_email:
		for i in dictio_of_emails[email]:
			dictio_of_users[i].remove(email)
		ret2 = dictio_of_emails.pop(email, None)
		if ret2 is None:
			print("THERE IS AN ERROR: ", email)
	# Update the list of users
	users_removed = [k for k, v in dictio_of_users.items() if v == []]
	#print("Users removed: ", len(users_removed))
	for user in users_removed:
		ret = dictio_of_users.pop(user, None)
		if ret is None:
			print("ERROR")
	row_format ="{:>15}" * 4
	print("-" * 15 * 4)
	print(row_format.format("Original emails", "Removed emails", "New emails", "Percentage"))
	print(row_format.format("%d"% (old_len_email), "%d"%(len(list_email)), "%d"%(len(dictio_of_emails)), "%f" %(len(dictio_of_emails)/old_len_email)))
	print(row_format.format("Original Users", "Removed Users", "New Users", "Percentage"))
	print(row_format.format("%d"% (old_len_u), "%d"%(len(users_removed)), "%d"%(len(dictio_of_users)), "%f" %(len(dictio_of_users)/old_len_u)))
	print("-" * 15 * 4)
	#print(len(dictio_of_users)/old_len_u, len(dictio_of_emails)/old_len_email, 
		#len(dictio_of_users), len(dictio_of_emails))
	return dictio_of_users, dictio_of_emails

def clean_users_from_dictios2(list_users, dictio_of_users, dictio_of_emails):
	old_len_u, old_len_email = len(dictio_of_users), len(dictio_of_emails)
	#print("emails removed: ", len(list_email) )
	for user in list_users:
		for email in dictio_of_users[user]:
			dictio_of_emails[email].remove(user)
		ret2 = dictio_of_users.pop(user, None)
		

	# Update the list of users
	emails_removed = [k for k, v in dictio_of_emails.items() if v == []]
	#print("Users removed: ", len(users_removed))
	for email in emails_removed:
		ret = dictio_of_emails.pop(email, None)
		if ret is None:
			print("ERROR")
	row_format ="{:>15}" * 4
	print("-" * 15 * 4)
	print(row_format.format("Original emails", "Removed emails", "New emails", "Percentage"))
	print(row_format.format("%d"% (old_len_email), "%d"%(len(emails_removed)), "%d"%(len(dictio_of_emails)), "%f" %(len(dictio_of_emails)/old_len_email)))
	print(row_format.format("Original Users", "Removed Users", "New Users", "Percentage"))
	print(row_format.format("%d"% (old_len_u), "%d"%(len(list_users)), "%d"%(len(dictio_of_users)), "%f" %(len(dictio_of_users)/old_len_u)))
	print("-" * 15 * 4)
	#print(len(dictio_of_users)/old_len_u, len(dictio_of_emails)/old_len_email, 
		#len(dictio_of_users), len(dictio_of_emails))
	return dictio_of_users, dictio_of_emails


def clean_dataset(dictio_of_users, dictio_of_emails):
	# Remove emails with 1 appearance.
	oneapp = [k for k,v in dictio_of_emails.items() if len(v) == 1]
	while(len(oneapp) > 0):
		print("Removing emails that appear once...")
		dictio_of_users, dictio_of_emails = clean_users_from_dictios(oneapp, dictio_of_users, dictio_of_emails)
		print("Removing emails that appear more than 12 times...")
		multiemail = [k for k,v in dictio_of_emails.items() if len(v) > 12]
		dictio_of_users, dictio_of_emails = clean_users_from_dictios(multiemail, dictio_of_users, dictio_of_emails)
		print("Removing users that have less than 5 emails...")
		atleast10 = [k for k,v in dictio_of_users.items() if len(v) < 5]
		dictio_of_users, dictio_of_emails = clean_users_from_dictios2(atleast10, dictio_of_users, dictio_of_emails)
		oneapp = [k for k,v in dictio_of_emails.items() if len(v) == 1]
	return dictio_of_users, dictio_of_emails



def gen_bin_matrix_of_users(dictio_of_emails, dictio_of_users):
	num_users = len(dictio_of_users)
	num_emails = len(dictio_of_emails)
	# Transform dictionary to indexes
	for indk, email in enumerate(dictio_of_emails.keys()):
		dictio_of_emails[email] = indk
	#Transform users to matrices.
	print("ESTIMATED SIZE OF MATRIX: %f GB" % (num_users * num_emails * 1 / (1024 ** 3)))
	matrix_map = np.memmap('email_files/email_matrix_map.dat', dtype=np.uint8, mode ='w+', shape=(num_users, num_emails))
	status.create_numbar(100, num_users)
	for ind, user in enumerate(dictio_of_users.keys()):
		status.update_numbar(ind, num_users)
		base = np.zeros((num_emails,1), dtype=np.uint8)
		for email in dictio_of_users[user]:
			base[dictio_of_emails[email]] = 1
		base = np.squeeze(base)
		matrix_map[ind] = base[:]
	status.end_numbar()
	print("Flushing...")
	matrix_map.flush()
	

def gen_new_matrix_of_users(dictio_of_emails, dictio_of_users, dictio_of_values):
	num_users = len(dictio_of_users)
	num_emails = len(dictio_of_emails)
	# Transform dictionary to indexes
	for indk, email in enumerate(dictio_of_emails.keys()):
		dictio_of_emails[email] = indk
	#Transform users to matrices.
	print("ESTIMATED SIZE OF MATRIX: %f GB" % (num_users * num_emails * 1 / (1024 ** 3)))
	matrix_map = np.memmap('email_files/email_matrix_map.dat', dtype=np.uint8, mode ='w+', shape=(num_users, num_emails))
	status.create_numbar(100, num_users)
	for ind, user in enumerate(dictio_of_users.keys()):
		status.update_numbar(ind, num_users)
		base = np.zeros((num_emails,1), dtype=np.uint8)
		for email in dictio_of_users[user]:
			x = dictio_of_values[email]
			base[dictio_of_emails[email]] = x
		base = np.squeeze(base)
		matrix_map[ind] = base[:]
	status.end_numbar()
	print("Flushing...")
	matrix_map.flush()


def gen_vector_for_user(user, dictio_of_users, dictio_of_emails, dictio_of_values = None, dtype=np.uint8):
	for indk, email in enumerate(dictio_of_emails.keys()):
		dictio_of_emails[email] = indk

	base = np.zeros((len(dictio_of_emails), 1), dtype=dtype)
	#print(type(dictio_of_users))
	for email in dictio_of_users[user]:
		base[dictio_of_emails[email]] = 1 if dictio_of_values is None else dictio_of_values[email]
	base = np.squeeze(base)
	return base

def gen_link_values(lst_links):

	#lst_links = read_csv_list("link_files/link_count.csv")[1:]
	print("Lenght link Count: %d" %(len(lst_links)))
	lst_links = sorted(lst_links, key=lambda x: x[1], reverse=True)
	print(lst_links[:3])
	divisions = int(math.ceil(float(len(lst_links)) / 255.0))
	dictio = {}
	for i in range(0, 254):
		start = i * divisions
		end = (i + 1) * divisions
		for elem in lst_links[start:end]:
			dictio[elem[0]] = (i + 1)
	#print(dictio)
	return dictio


def generate_clean_dataset():
	tic = time.time()
	lst_users = gen_new_dataset()
	#print(lst_users[:3])
	#lst_users = sorted(lst_users, key=lambda x: len(x), reverse=True)
	dictio_of_users = gen_dictio_of_users(lst_users)
	dictio_of_emails = gen_dictio_of_emails(lst_users)

	lst_emails = [(k, len(v)) for k, v in dictio_of_emails.items()]
	dictio_of_values = gen_link_values(lst_emails)

	dictio_of_users, dictio_of_emails = clean_dataset(dictio_of_users,dictio_of_emails)
	print("SECONDS: %f" %(time.time() - tic))
	return dictio_of_users, dictio_of_emails, dictio_of_values

def process_emails_euclidean(dictio_of_users, dictio_of_emails):
	tic = time.time()
	num_users = len(dictio_of_users)
	num_emails = len(dictio_of_emails)
	matrix_map = np.memmap('email_files/email_matrix_map.dat', dtype=np.uint8,  shape=(num_users, num_emails))
	lst_users = list(dictio_of_users.keys())
	#matrix_map = np.array(matrix_map)
	lst_res = []
	print(matrix_map.shape)
	print("ESTIMATED SIZE OF MATRIX: %f GB" % (num_users * num_users * 4 / (1024 ** 3)))
	status.create_numbar(100, num_users)
	#res = np.dot(matrix_map, matrix_map.T)
	res2 = np.memmap('email_files/email_euc_score.dat', dtype=float ,mode ='w+', shape=(num_users, num_users))
	#res3 = np.memmap('email_files/email_dis_score.dat', dtype=np.uint32 ,mode ='w+', shape=(num_users, num_users))
	for i1 in range(num_users):
		status.update_numbar(i1, num_users)
		v1 = np.array(matrix_map[i1], dtype=np.int32)
		#v1p = gen_vector_for_user(list(dictio_of_users.keys())[i1], dictio_of_users, dictio_of_emails)

		#print(np.array_equal(v1, v1p))
		for i2 in range(i1 + 1, num_users):
			v2 = np.array(matrix_map[i2], dtype=np.int32)
			#print(v1.shape, v2.shape)
			#print(v1)
			euc_score = np.linalg.norm(v1-v2)
			#dis_score = np.dot(v1, v2)
			lst_res.append((lst_users[i1], lst_users[i2], euc_score))
			res2[i1][i2] = euc_score
			#res3[i1][i2] = dis_score
			#res2[i2][i1] = score
	lst_res = sorted(lst_res, key=lambda x: x[2], reverse=False)
	gen_csv_from_tuples("email_files/results_email_euc_alt.csv", ["IdAuthor1", "IdAuthor2", "Score"], lst_res)
	status.end_numbar()
	print("Flushing...")
	res2.flush()
	
	print("SECONDS: %f" %(time.time() - tic))

def process_emails_distance(dictio_of_users, dictio_of_emails):
	tic = time.time()
	num_users = len(dictio_of_users)
	num_emails = len(dictio_of_emails)

	matrix_map = np.memmap('email_files/email_matrix_map.dat', dtype=np.uint8,  shape=(num_users, num_emails))
	#matrix_map = np.array(matrix_map)

	print(matrix_map.shape)
	print("ESTIMATED SIZE OF MATRIX: %f GB" % (num_users * num_users * 4 / (1024 ** 3)))
	status.create_numbar(100, num_users)
	#res = np.dot(matrix_map, matrix_map.T)
	#res2 = np.memmap('email_files/email_euc_score.dat', dtype=float ,mode ='w+', shape=(num_users, num_users))
	res3 = np.memmap('email_files/email_dis_score.dat', dtype=np.uint32 ,mode ='w+', shape=(num_users, num_users))
	for i1 in range(num_users):
		status.update_numbar(i1, num_users)
		v1 = np.array(matrix_map[i1], dtype=np.uint32)
		#v1p = gen_vector_for_user(list(dictio_of_users.keys())[i1], dictio_of_users, dictio_of_emails)
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


def read_results_dis(dictio_of_users, dictio_of_emails):
	tic = time.time()
	
	num_users = len(dictio_of_users)
	lst_users = list(dictio_of_users.keys())
	lst = []
	res2 = np.memmap('email_files/email_dis_score.dat', dtype=np.uint32, mode ='r', shape=(num_users, num_users))
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
	gen_csv_from_tuples("email_files/results_email_dis.csv", ["IdAuthor1", "IdAuthor2", "Score"], sortedl)

def read_results_euc(dictio_of_users, dictio_of_emails):
	tic = time.time()
	
	num_users = len(dictio_of_users)
	lst_users = list(dictio_of_users.keys())
	lst = []
	res2 = np.memmap('email_files/email_euc_score.dat', dtype=float, mode ='r', shape=(num_users, num_users))
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
	gen_csv_from_tuples("email_files/results_email_euc.csv", ["IdAuthor1", "IdAuthor2", "Score"], sortedl)

def gen_email_values():

	lst_emails = read_csv_list("email_files/email_count.csv")[1:]
	print("Lenght email Count: %d" %(len(lst_emails)))
	lst_emails = sorted(lst_emails, key=lambda x: x[1], reverse=True)
	print(lst_emails[:3])
	divisions = int(math.ceil(float(len(lst_emails)) / 254.0))
	dictio = {}
	for i in range(0, 254):
		start = i * divisions
		end = (i + 1) * divisions
		for elem in lst_emails[start:end]:
			dictio[elem[0]] = (i + 1)
	#print(dictio)
	return dictio

def new_read_results_euc(dictio_of_users, dictio_of_emails):
	print("New user extraction...")
	tic = time.time()
	num_users = len(dictio_of_users)
	lst_users = list(dictio_of_users.keys())
	lst = []
	res2 = np.memmap('email_files/email_dis_score.dat', dtype=np.uint32, mode ='r', shape=(num_users, num_users))
	status.create_numbar(100, num_users)
	for i in range(num_users):
		status.update_numbar(i, num_users)
		for j in range(i + 1, num_users):
			if res2[i][j] > 0:
				lst.append((lst_users[i],lst_users[j],res2[i][j]))
	status.end_numbar()
	print("SECONDS: %f" %(time.time() - tic))
	print("Old length: %d" % (len(lst)))
	#lst = [x for x in lst if x[2] > 0]
	print("New length: %d" % (len(lst)))
	sortedl = sorted(lst, key=lambda x: x[2], reverse=True)
	print("SECONDS: %f" %(time.time() - tic))
	gen_csv_from_tuples("email_files/new_results_email_dis.csv", ["IdAuthor1", "IdAuthor2", "Score"], sortedl)

	lst = []
	res3 = np.memmap('email_files/email_euc_score.dat', dtype=float, mode ='r', shape=(num_users, num_users))
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
	gen_csv_from_tuples("email_files/new_results_email_euc.csv", ["IdAuthor1", "IdAuthor2", "Score"], sortedl)


def combin(n,r):
	return int(np.math.factorial(n) / (np.math.factorial(r) * np.math.factorial(n - r)))

def print_non_zero(v):
	for indi, i in enumerate(v):
		if i != 0:
			print(indi, i)
def calc_scores_for_users(u1, u2, dictio_of_users, dictio_of_emails, dictio_of_values):
	v1e = gen_vector_for_user(u1, dictio_of_users, dictio_of_emails, dtype=np.int32)
	v2e = gen_vector_for_user(u2, dictio_of_users, dictio_of_emails, dtype=np.int32)
	v1d = gen_vector_for_user(u1, dictio_of_users, dictio_of_emails, dictio_of_values=dictio_of_values)
	v2d = gen_vector_for_user(u2, dictio_of_users, dictio_of_emails, dictio_of_values=dictio_of_values)
	print("v1e")
	print_non_zero(v1e)
	print("v2e")
	print_non_zero(v2e)
	print("v1d")
	print_non_zero(v1d)
	print("v2d")
	print_non_zero(v2d)
	print("dot: ", np.dot(v1d, v2d), "euc:", np.linalg.norm(v1e-v2e))
def main():
	#extract_email_to_usage()
	#extract_user_to_email_csv()
	# a = "1389627[0]"
	# b = "923181[4]"
	dictio_of_users, dictio_of_emails, dictio_of_values = generate_clean_dataset()
	# lst_users = list(dictio_of_users.keys())
	# print(type(lst_users[0]))
	
	#print(a in lst_users, b in lst_users)	
	#calc_scores_for_users(a, b, dictio_of_users, dictio_of_emails, dictio_of_values)
	#dictio_of_values = gen_email_values()
	#gen_bin_matrix_of_users(dictio_of_emails, dictio_of_users)
	#process_emails_euclidean(dictio_of_users, dictio_of_emails)
	#gen_new_matrix_of_users(dictio_of_emails, dictio_of_users, dictio_of_values)
	#process_emails_distance(dictio_of_users, dictio_of_emails)
	new_read_results_euc(dictio_of_users, dictio_of_emails)
	#read_results_euc(dictio_of_users, dictio_of_emails)
	#read_results_dis(dictio_of_users, dictio_of_emails)
	

if __name__ == "__main__":
	main()
