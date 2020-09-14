import psycopg2, json, csv
import re
import os
import status
import operator
import math
import numpy as np
import multiprocessing
import itertools
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from threading import Thread
import threading
import time
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from random import shuffle
from common_utils import gen_csv_from_tuples, read_csv_list
from db_password import  *



##############################################################################################
# FUNCTION: string_similar
# DESCRIPTION:  Returns a scoring of the similarity of two strings
##############################################################################################
def string_similar(a, b):
	return SequenceMatcher(None, a, b).ratio()

##############################################################################################
# FUNCTION: preprocess_content
# DESCRIPTION:  Performs preprocessing of a string according to rules.
##############################################################################################
def preprocess_content(string):
	stop_words = set(stopwords.words('english')) 
	ps = PorterStemmer()
	# Remove special sentences
	string = re.sub(r'\*{3}IMG\*{3}.*?\*{3}IMG\*{3}', ' XIMAGEV ', string)
	string = re.sub(r'\*{3}CITING\*{3}.*?\*{3}CITING\*{3}', ' XCITEV ', string)
	string = re.sub(r'\*{3}IFRAME\*{3}.*?\*{3}IFRAME\*{3}', ' XFRAMEV ', string)
	string = re.sub(r'\*{3}ILINK\*{3}.*?\*{3}ILINK\*{3}', ' XLINKV ', string)
	string = re.sub(r'\d+', ' xnumberv ', string)
	string = re.sub(r'\$', ' xdollarv ', string)
	string = re.sub(r'\€', ' xeurov ', string)
	#Remove stop words and tokenize string
	#string = " ".join([ps.stem(w) for w in word_tokenize(string.lower()) if not w in stop_words])
	string = " ".join([w for w in word_tokenize(string.lower()) if not w in stop_words and w.isalpha()])
	return string

##############################################################################################
# FUNCTION: tf_idf
# DESCRIPTION:  Returns the tfidf with scoring and string.
##############################################################################################
def tf_idf(corpus):
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
	return ret

##############################################################################################
# FUNCTION: clean_usernames
# DESCRIPTION:  Creates a dictionary with each user and the forums where it has been seen,
#				then it extracts the posts and performs tfidf for each user in the list.
##############################################################################################
def clean_usernames():
	lst = read_csv_list("similar_usernames_full.csv")[1:]
	lst2 = read_csv_list("all_posts_all_users.csv")
	lst = [x for x in lst if x[0] != '']
	lst2 = [x for x in lst2 if x[0] != '']
	total = 0
	count = 0
	dictio = {}
	results = []
	threshold = 20
	for i in lst2:
		#print (i[0:2], i[2])
		dictio[i[0]] = {}
	for i in lst2:
		#print (i[0:2], i[2])
		dictio[i[0]][i[1]] = int(i[2])	
	
	not_both = [x for x in lst if x[0] not in dictio.keys()]
	print(len(dictio), len(list(set([x[0] for x in lst]))), len(not_both), len(not_both)+ len(dictio))

	for i in lst:
		boolean = True
		for j in i[1:]:
			if i[0] in dictio.keys() and dictio[i[0]][j] < threshold:
				boolean = False
		if boolean:
			#print (i)
			results += [i]
		else:
			del dictio[i[0]]
	for i in results:
		if (len (i) > 3):
			count += 1
		if (len (i) > 4):
			total += 1

	print ("At least two: %d" % (len(dictio)), "At least three: %d" % (count) , "At least four: %d" % (total))
	conn = psycopg2.connect(database="crimebb", user=db_username, password=db_password,  host="127.0.0.1", port="5432")
	print("Database Connected....")
	rows_processed = []
	
	status.create_numbar(100, len(dictio))

	for indi, i in enumerate(dictio.keys()):
		#print (multiprocessing.current_process(), "%0.2f %%" % ( indi * 100 / len(lst)))
		status.update_numbar(indi, len(dictio))
		for key in dictio[i].keys():
			cur = conn.cursor()
			cur.execute("""SELECT "Post"."Content"
			from "Post" JOIN "Member" ON "Post"."Author" = "Member"."IdMember"
			WHERE ("Member"."Username" = %s) AND "Member"."Site" = %s;""", (i, int(key)))
			rows = [row[0] for row in cur.fetchall()]
			#print (rows[0])
			tfidf = tf_idf(rows)
			tfidf = sorted(tfidf, key=lambda x: x[-1], reverse=True)
			#print(i, key, tfidf[:3])
			tfidf = [i for j in tfidf[:50] for i in j]
			dictio[i][key] = tuple(tfidf)
			#print (i[0], j, count)
		if indi == 100:
			k = list(dictio.keys())
			rows_processed = [(user, forum) + dictio[user][forum] for user in k[:100] for forum in dictio[user].keys()]
			gen_csv_from_tuples("tfidf_prov.csv", [""], rows_processed)

	status.end_numbar()
	rows_processed = [(user, forum) + dictio[user][forum] for user in dictio.keys() for forum in dictio[user].keys()]
	gen_csv_from_tuples("tfidf.csv", [""], rows_processed)
	conn.close()

def main():
	clean_usernames()
	
if __name__ == "__main__":
	main()