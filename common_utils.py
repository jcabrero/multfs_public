#import psycopg2
import json, csv
import re
import os
#import status
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
# from db_credentials import  *
# from mail_credentials import  *
import requests
import resource
import multiprocessing as mp
import pickle
import smtplib
import getpass


def create_dir(filename):
	try:
	    # Create target Directory
	    os.mkdir(filename)
	    #print("Directory " , filename ,  " Created ") 
	except FileExistsError:
	    print("Directory " , filename ,  " already exists")
	    
def get_ram():
	kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
	mb, kb = kb / 1024, kb % 1024
	gb, mb = mb / 1024, mb % 1024
	return "[%0.4f GB]" % (gb)

def get_elapsed_time(tic):
	elapsed = time.time() - tic
	mins, secs = int(elapsed / 60), elapsed % 60
	hours, mins = int(mins / 60), mins % 60
	return "[%d HOURS, %02d MINS, %02d SECS]" % (hours, mins, secs)

def pickle_object(obj, filename):
	pickle.dump(obj, open(filename, 'wb'), protocol=4)

def unpickle_object(filename):
	with open(filename, 'rb') as f:
		obj = pickle.load(f)
		return obj
##############################################################################################
# FUNCTION: gen_csv_from_tuples
# DESCRIPTION:  Generates a csv with all the links and the user who posted it.
# OUTPUT_FORMAT: (index, "AuthorId", "AuthorName", "Link")
##############################################################################################
def gen_csv_from_tuples(name, titles, rows):
	#file = open('id_user_url.csv', mode='w+')
	file = open(name, mode='w+')
	writer = csv.writer(file, delimiter=',', quotechar="'", quoting=csv.QUOTE_MINIMAL)
	writer.writerow(titles)
	for row in rows:
		writer.writerow(row)
##############################################################################################
# FUNCTION: read_csv_as_list
# DESCRIPTION:  Generates a list of tuples of the CSV
# OUTPUT_FORMAT: (index, "AuthorId", "AuthorName", "Link")
##############################################################################################
def read_csv_list(name):
	with open(name) as f:
		data=[tuple(line) for line in csv.reader(f, delimiter=',', quotechar="'", quoting=csv.QUOTE_MINIMAL)]
		return data
		# lst = []
		# status.create_numbar(100, 10000000)
		# for line in csv.reader(f, delimiter=',', quotechar="'", quoting=csv.QUOTE_MINIMAL):
		# 	status.update_numbar(len(lst), 10000000)
		# 	lst.append(tuple(line))
		# status.end_numbar()
		# return lst

def read_csv_list2(name):
	pool = mp.Pool(16)
	with open(name) as file:
		from defs import f
		data= pool.map(f,csv.reader(file, delimiter=',', quotechar="'", quoting=csv.QUOTE_MINIMAL))
		pool.close()
		return data

def read_csv_list2(name):
	pool = mp.Pool(56)
	with open(name) as file:
		from defs import f
		data= pool.map(f,file)
		pool.close()
		return data

def read_csv_dict(name):
	dictio = {}
	with open(name) as f:
		for line in csv.reader(f, delimiter=',', quotechar="'", quoting=csv.QUOTE_MINIMAL):
			line = tuple(line)
			dictio[line[0]] = line[1:]
	return dictio
##############################################################################################
# FUNCTION: make_query
# DESCRIPTION:  Makes a query to the database
# OUTPUT_FORMAT: (index, "AuthorId", "AuthorName", "Link")
##############################################################################################
# conn = None
def make_query(query,conn=None):
	closeConn=False
	if conn is None:
		closeConn=True
		conn = psycopg2.connect(user=getpass.getuser(), database='crimebb')
	#print("[DB] Extracting data")

	cur = conn.cursor()
	cur.execute(query)
	rows = cur.fetchall()
	if closeConn:
		conn.close()
	return rows
def make_query2(query):
	global conn
	if conn is None:
		conn = psycopg2.connect(user=getpass.getuser(), database='crimebb')
		# conn = psycopg2.connect(database="crimebb", user=db_username, password=db_password,  host="127.0.0.1", port="5432")
	#print("[DB] Extracting data")

	cur = conn.cursor()
	cur.execute(query)
	rows = cur.fetchall()
	# conn.close()
	return rows

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1



def get_filename_dir(path):
	directory = path[0:len(path)-path[::-1].find("/")]
	filename = path[len(path) - path[::-1].find("/"):]
	return directory, filename

"""
This function joins all the elements with the same filename. It is used to generate the unique file with the results from the partial ones.
"""
def join_all_results(origin_filename):
	tic = time.time()
	directory, filename = get_filename_dir(origin_filename)
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



def send_mail(message):
	# creates SMTP session 
	smtpserver = smtplib.SMTP('smtp.gmail.com', 587) 

	smtpserver.ehlo()

	# start TLS for security 
	smtpserver.starttls() 

	smtpserver.ehlo()
	  
	# Authentication 
	smtpserver.login(mail_username, mail_password) 
	  
	# message to be sent 
	#message = "Message_you_need_to_send"
	  
	# sending the mail 
	smtpserver.sendmail(mail_username, mail_dest, message) 
	  
	# terminating the session 
	smtpserver.quit() 
