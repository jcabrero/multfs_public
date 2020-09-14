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
from common_utils import gen_csv_from_tuples, read_csv_list, make_query, file_len
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#Global variables
set_of_words = set()
DEBUG = True

#Functions
def preprocess_content(string):
	stop_words = set(stopwords.words('english')) 
	ps = PorterStemmer()
	#table = str.maketrans('', '', string.punctuation)
	#print (string)
	# 1 - Lower case
	#string = string.lower()
	# 2 - Remove special sentences
	string = re.sub(r'\*{3}IMG\*{3}.*?\*{3}IMG\*{3}', ' imgimgimg ', string)
	string = re.sub(r'\*{3}CITING\*{3}.*?\*{3}CITING\*{3}', ' citecitecite ', string)
	string = re.sub(r'\*{3}IFRAME\*{3}.*?\*{3}IFRAME\*{3}', ' frameframeframe ', string)
	string = re.sub(r'\*{3}ILINK\*{3}.*?\*{3}ILINK\*{3}', ' linklinklink ', string)
	string = re.sub(r'\d+', ' numnumnum ', string)
	string = re.sub(r'\$', ' dollardollardollar ', string)
	string = re.sub(r'\€', ' euroeuroeuro ', string)
	#string = " ".join([ps.stem(w) for w in word_tokenize(string.lower()) if not w in stop_words])
	string = " ".join([w for w in word_tokenize(string.lower()) if not w in stop_words and w.isalpha()])
	#print (string)
	return string


def tf_idf(corpus):
	global set_of_words
	num_words = 25
	if DEBUG:
		print("[-] Calculating TFIDF")
	ret = []
	corpus = " ".join([preprocess_content(string) for string in corpus])
	#corpus = [preprocess_content(string) for string in corpus]
	#print(corpus)
	tfidf = TfidfVectorizer()
	response = tfidf.fit_transform([corpus])
	feature_names = tfidf.get_feature_names()
	for col in response.nonzero()[1]:
		ret +=  [(feature_names[col], response[0, col])]
		#print (feature_names[col], ' - ', response[0, col])
	#Sorting words
	ret = sorted(ret, key=lambda x: x[-1], reverse=True)
	for i in ret[:num_words]:
		set_of_words.add(i[0])
	#Tuple generation
	ret = [i for j in ret[:num_words] for i in j]

	return ret


def get_user_and_site(userid):
	user = userid[0:userid.find("[")]
	site = userid[userid.find("[") + 1: userid.find("]")]
	return int(user), int(site)


def get_post_content(user, site):
	if DEBUG:
		print("[-] Getting post content %d, %d" % (user, site))
	query = """SELECT "Post"."Content" from "Post" 
					WHERE "Post"."Author" = %d AND "Post"."Site" = %d;""" % (user, site)
	rows = make_query(query)
	rows = [tup[0] for tup in rows]
	return rows


def get_num_threads(user, site):
	if DEBUG:
		print("[-] Getting num_threads")
	#Posts created by user
	query1 = """SELECT count("IdThread") FROM "Thread" WHERE "Thread"."Author" = %d AND "Thread"."Site" = %d;""" % (user, site)
	#Posts he participated in
	query2 = """WITH A AS 
	(SELECT "Post"."Thread" FROM "Post" 
	WHERE "Post"."Author" = %d AND "Post"."Site" = %d 
	GROUP BY "Post"."Thread") 
	SELECT count(A."Thread") FROM A;""" % (user, site)
	rows1 = make_query(query1)
	rows2 = make_query(query2)
	#print(rows1, rows2)
	return rows1[0][0], rows2[0][0]

def extract_tfidf(userid):
	pass


def get_metrics(post_content, num_threads, num_participations):
	if DEBUG:
		print("[-] Getting metrics")
	num_posts = len(post_content)
	post_to_own_thread = num_posts / num_threads if num_threads != 0 else 0
	post_to_gen_thread = num_posts/num_participations if num_participations != 0 else 0
	own_thread_to_gen_thread = num_threads/num_participations if num_participations != 0 else 0
	return [post_to_own_thread, post_to_gen_thread, own_thread_to_gen_thread]

def process_user(userid):
	user, site = get_user_and_site(userid)

	if DEBUG:
		print("[-] User: %d Site: %d" % (user, site))
	post_content = get_post_content(user, site)
	res_tfidf = tf_idf(post_content)
	num_threads, num_participations = get_num_threads(user, site)
	res_metrics = get_metrics(post_content, num_threads, num_participations)
	#print(res_metrics + res_tfidf)
	print(userid)
	return [userid] + res_metrics + res_tfidf



def process_all_users(lst):
	print("[-] Processing users")
	ind_res = []
	pool = mp.Pool(24)
	ind_res = pool.map(process_user, lst)
	# status.create_numbar(100, len(lst))
	# index = 0
	# for indi,user in enumerate(lst):
	# 	status.update_numbar(indi, len(lst))
	# 	result = process_user(user)
	# 	ind_res.append(ind_res)
	# 	if (indi % 1000) == 0:

	# 		ind_res = []
	# 		index += 1
	print("[-] Creating csv ...")
	gen_csv_from_tuples("ind_users.csv", ["User", "Post to Own Thread", 
						"Post to Gen Thread", "Own Thread to Gen Thread", 
						"TFIDF"], ind_res)

	status.end_numbar()


def string_similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def get_username(user, site):
	query = """SELECT "Member"."Username" FROM "Member" WHERE "Member"."IdMember" = %d AND "Member"."Site" = %d;""" %(user, site)
	rows = make_query(query)
	return rows[0][0]


def process_pair(pair):
	userid1, userid2 = pair[0], pair[1]
	user1, site1 = get_user_and_site(userid1)
	user2, site2 = get_user_and_site(userid2)
	if DEBUG:
		print("[-] User 1: %d Site 1: %d User 2: %d Site 2: %d" % (user1, site1, user2, site2))
	username1, username2 = get_username(user1, site1), get_username(user2, site2)
	sim = string_similar(username1, username2)
	#print("[-] User 1: %d Site 1: %d User 2: %d Site 2: %d" % (user1, site1, user2, site2))
	print(sim, username1, username2)
	return [userid1, userid2] + [sim, username1, username2]




def process_pairs(lst):
	lst = [x for x in lst if float(x[-1]) < 1.0 and x[0][0] != '-' and x[1][0] != '-']
	print("[-] Processing pairs")
	ind_res = []
	pool = mp.Pool(16)
	ind_res = pool.map(process_pair, lst)
	a = np.array([x[2] for x in ind_res])
	b = [x for x in ind_res if x[2] > 0.75]
	print("B", b)
	print(np.mean(a), a.shape)
	# print(ind_res)
	# status.create_numbar(100, len(lst))
	# index = 0
	# for indi,user in enumerate(lst):
	# 	status.update_numbar(indi, len(lst))
	# 	result = process_pair(user)
	# 	ind_res.append(result)
	print("[-] Creating csv ...")
	gen_csv_from_tuples("gen_users.csv", ["User", "Post to Own Thread", 
						"Post to Gen Thread", "Own Thread to Gen Thread", 
						"TFIDF"], ind_res)


def extract_all_users():

	print("[-] Extracting users")
	lst = read_csv_list("weighted_average.csv")[1:]
	set_user = set()
	for i in lst:
		set_user.add(i[0])
		set_user.add(i[1])
	return list(set_user), [x[0:3] for x  in lst]


def extract_features():
	print("[-] Preprocessing users...")
	lst_users, lst_pairs = extract_all_users()
	process_pairs(lst_pairs)
	#process_all_users(lst_users)


def word_from_features(ind_features):
	set_words = set()
	for user in ind_features:
		words = [x for x in user[4::2]]
		for word in words:
			set_words.add(word)
	
	set_words = list(set_words)
	dictio = {}
	for indi, i in enumerate(set_words):
		dictio[i] = indi
	return dictio


def store_dictio(dictio):
	lst = [(k, v) for k,v in dictio.items()]
	gen_csv_from_tuples("word_index.csv",[""], lst)

def get_dictio_from_file():
	lst = read_csv_list("word_index.csv")[1:]
	return {x[0]:int(x[1]) for x in lst}


def generate_word_dictionary():
	print("[-] Generating dictionary")
	lst = read_csv_list("ind_users.csv")[1:]
	dictio = word_from_features(lst)
	store_dictio(dictio)


def get_ind_features():
	lst = read_csv_list("ind_users.csv")[1:]
	return {x[0]:x[1:] for x in lst}

def get_gen_fetures():
	lst = read_csv_list("gen_users.csv")[1:]
	return {x[0]+"-"+x[1]:x[2:] for x in lst}


def gen_vector_for_pairs(word_dictio, ind_features, weights):
	
	words_per_user = len(word_dictio)
	num_words = words_per_user * 2 + 1
	num_pairs = len(weights)
	print("ESTIMATED SIZE OF MATRIX: %f GB" % (num_words * num_pairs * 4 / (1024 ** 3)))
	matrix_map = np.memmap('input_keras.dat', dtype=np.single, mode ='w+', shape=(num_pairs, num_words))
	status.create_numbar(100, num_pairs)
	for i in range(0, num_pairs):

		status.update_numbar(i, num_pairs)
		
		v = np.zeros((num_words))
		u1, u2 = weights[i][0], weights[i][1]
		for j in range(len(ind_features[u1][4::2])):
			a = ind_features[u1][2 * j + 3]
			b = ind_features[u1][2 * j + 4]
			word_index = int(word_dictio[a])
			#print(a, b, word_index)
			v[word_index] = b
		for j in range(len(ind_features[u1][4::2])):
			a = ind_features[u1][2 * j + 3]
			b = ind_features[u1][2 * j + 4]
			word_index = int(word_dictio[a])
			v[words_per_user + word_index] = b
		v[num_words - 1] = weights[i][2]
		print(v.shape, matrix_map.shape, matrix_map[i].shape)
		matrix_map[i] = v[:]
	status.end_numbar()


def main():
	extract_features()
	return
	ind_users , weights = extract_all_users()
	process_all_users(ind_users)
	word_dictio = get_dictio_from_file()
	ind_dictio = get_ind_features()
	
	num = 10000
	#weights = [(x[0], x[1], 1) for x in weights[:num]] + [(x[0], x[1], 0) for x in weights[-num:]]
	weights = [x for x in weights[:num]] + [x for x in weights[-num:]]  
	gen_vector_for_pairs(word_dictio, ind_dictio, weights)
	pass


if __name__ == "__main__":
	main()