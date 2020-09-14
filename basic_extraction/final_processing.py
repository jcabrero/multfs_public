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
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dictio_of_results = {}
lst_res = ['skype','email','ip','btc', 'link'] 
lst_res2 = ['skype','email','ip', 'link', 'btc']
dictio_len = {'skype': 162513406, 'email': 1296855, 'link': 111983095, 'ip': 6663075, 'btc': 244591903}
#lst_res = ['email'] 
def check_file_attr(filename):
	exists = os.path.isfile(filename)
	if exists:
		statinfo = os.stat(filename)
		print("%s [%0.2f GB] %s" % ("[OK]", statinfo.st_size / 1024 ** 3, filename))
	else:
		print("%s %s" % ("[ERROR]", filename))
	return exists

def get_file_lengths():
	dictio = {}
	for res in lst_res:
		for x in ["_euc", "_dis"]:
			filename = "new_"+ res + "_files/results_" + res + x
			filelen = file_len(filename)
			dictio[res+x] = filelen
	return dictio


def join_euc_results_csv():
	global dictio_of_results
	limit = 1000000 # 1 million
	#dictio_of_results = {}
	for res in lst_res:
		tic = time.time()
		filename = res + "_files/new_results_" + res + "_euc.csv"
		# Check if file exists
		if not check_file_attr(filename):
			continue
		print("[-] Getting file length...")
		#filelen = file_len(filename)
		filelen = dictio_len[res]
		indi = 0
		status.create_numbar(100, filelen)
		with open(filename) as f:
			for line in csv.reader(f, delimiter=',', quotechar="'", quoting=csv.QUOTE_MINIMAL):
				indi += 1
				status.update_numbar(indi, filelen)
				if indi == 1:
					#The first line of the csv are titles
					continue
				x = tuple(line)
				i = (x[0], x[1], float(x[2]))

				#if (i[2] > limit):
					#If we surpass the limit we stop searching since it is ordered
					#break
				
				if not i[0] in dictio_of_results.keys():
				 	dictio_of_results[i[0]] = {}

				if not i[1] in dictio_of_results[i[0]].keys():
					dictio_of_results[i[0]][i[1]] = {}

				dictio_of_results[i[0]][i[1]][res+"_euc"] = i[2]

			status.end_numbar()
			print("[-] SECONDS: %f" %(time.time() - tic))
			print("[-] Length of dictionary: %d" % (len(dictio_of_results)))


def join_dis_results_csv():
	global dictio_of_results
	for res in lst_res:
		tic = time.time()
		filename = res + "_files/new_results_" + res + "_dis.csv"
		# Check if file exists
		if not check_file_attr(filename):
			continue
		print("[-] Getting file length...")
		#filelen = file_len(filename)
		filelen = dictio_len[res]
		indi = 0
		status.create_numbar(100, filelen)
		with open(filename) as f:
			for line in csv.reader(f, delimiter=',', quotechar="'", quoting=csv.QUOTE_MINIMAL):
				indi += 1
				status.update_numbar(indi, filelen)
				if indi == 1:
					#The first line of the csv are titles
					continue
				x = tuple(line)
				i = (x[0], x[1], float(x[2]))

				# if (i[2] == 0):
				# 	#If we surpass the limit we stop searching since it is ordered
				# 	break

				if not i[0] in dictio_of_results.keys():
					dictio_of_results[i[0]] = {}

				if not i[1] in dictio_of_results[i[0]].keys():
					dictio_of_results[i[0]][i[1]] = {}

				dictio_of_results[i[0]][i[1]][res+"_dis"] = i[2]

			status.end_numbar()
			print("[-] SECONDS: %f" %(time.time() - tic))
			print("[-] Length of dictionary: %d" % (len(dictio_of_results)))

def get_max():
	maximum = 0
	k1s, k2s, k3s = None, None, None
	status.create_numbar(100, len(dictio_of_results))
	for indi, k1 in enumerate(list(dictio_of_results.keys())):
		status.update_numbar(indi, len(dictio_of_results))
		for k2 in dictio_of_results[k1].keys():
			for k3 in dictio_of_results[k1][k2].keys():
				if dictio_of_results[k1][k2][k3] > maximum:
					maximum = dictio_of_results[k1][k2][k3]
					k1s, k2s, k3s = k1, k2, k3
	status.end_numbar()
	return maximum, k1s, k2s, k3s


def normalize_data(headers, lst):
	# We fill with 0 the n elements of a tuple
	max_items = [0 for i in range(len(lst[0][2:]))]
	num_metrics = len(max_items)
	num_users = len(lst)
	array = np.memmap('matrix.dat', dtype=float, mode ='w+', shape=(num_users, num_metrics))

	for indi, i in enumerate(lst):
		i = [i[0], i[1]] + [float(x) for x in i[2:]]

		for indj, j in enumerate(i[2:]):
			if max_items[indj] < j:
				max_items[indj] = j
	print(max_items)
	lst2 = []
	for indi, i in enumerate(lst):
		#print(i)
		i = [float(x) for x in i[2:]]
		for indj, j in enumerate(i):
			if j == -1:
				i[indj] = 1
			else:	
				i[indj] /= max_items[indj]
				if indj % 2 == 1:
					i[indj] = 1 - i[indj]
		temp = np.array(i, dtype=float)
		array[indi] = temp[:]
		#lst2.append(i)
	#array1 = np.array(lst2, dtype=float)
	print(array[:3])
	
	#array = np.copy(array1)

def weighted_average(headers, lst):
	max_items = [0 for i in range(len(lst[0][2:]))]
	num_metrics = len(max_items)
	num_users = len(lst)
	array = np.memmap('matrix.dat', dtype=float, mode ='r', shape=(num_users, num_metrics))
	arraymod = np.array(array,dtype=float)
	mean_arr = np.mean(arraymod, axis = 1)
	#std_arr = np.std(arraymod, axis = 1)
	zeros = (num_metrics - np.count_nonzero(arraymod == 0, axis = 1)) / float(num_metrics)
	quarter1 = (num_metrics - np.count_nonzero(arraymod <= 0.25, axis = 1)) / float(num_metrics)
	quarter2 = (num_metrics - np.count_nonzero(arraymod <= 0.5, axis = 1)) / float(num_metrics)
	quarter3 = (num_metrics - np.count_nonzero(arraymod <= 0.75, axis = 1)) / float(num_metrics)
	quarter4 = (num_metrics - np.count_nonzero(arraymod < 1.0, axis = 1)) / float(num_metrics)
	ones = np.count_nonzero(arraymod == 1, axis = 1) / float(num_metrics)

	print(arraymod.shape, mean_arr.shape, zeros.shape, quarter4.shape)
	#file_lengths = get_file_lengths()
	#for indi, i in enumerate(headers):
	#arraymod[indi] = arraymod[indi] / file
	results = []
	#status.create_numbar(100, num_users)
	for indi, i in enumerate(lst):
		#status.update_numbar(indi, num_users)

		metric = mean_arr[indi] * zeros[indi] * quarter1[indi] * quarter2[indi] * quarter3[indi] * quarter4[indi] * ones[indi]
		#print(mean_arr[indi], ((num_metrics - zeros[indi]) / float(num_metrics)), (ones[indi] / float(num_metrics)))
		a = (i[0], i[1], metric, mean_arr[indi], np.rint((1 - zeros[indi]) * 10), 
			np.rint((1 - quarter1[indi]) * 10), np.rint((1 - quarter2[indi]) * 10), 
			np.rint((1 - quarter3[indi]) * 10), np.rint((1 - quarter4[indi]) * 10), 
			np.rint(ones[indi] * 10))
		results.append(a)
	results = sorted(results, key=lambda x: x[2], reverse=False)
	#status.end_numbar()
	gen_csv_from_tuples("weighted_average.csv", ['user_a', 'user_b', 'metric', 'mean','zeros', '<0.25', '<0.5', '<0.75', '<1.0', '=1.0'], results)


def generate_graph():
	print("[-] Extracting data")
	lst = read_csv_list("weighted_average.csv")[1:]
	print("[-] Generating list")
	#from_nodes = [x[0] for x in lst]
	#to_nodes = [x[1] for x in lst]
	#weight = [x[2] for x in lst]
	elist = [(x[0], x[1], x[2]) for x in lst if float(x[2]) < 1.0]
	print("[-] Generating graph")
	G = nx.Graph()
	G.add_weighted_edges_from(elist)
	print("[-] Pickling")
	#nx.write_gpickle(G, "graph.pkl")
	nx.write_gexf(G, "graph.gexf")
	return G

def process_pickle():
	G = generate_graph()
	#graphs = list(nx.connected_component_subgraphs(G))
	# print("[-] Removing -1's")
	# for n1 in list(G):
	# 	if ('-1' in n1):
	# 		print("[X] Removed node -1")
	# 		G.remove_node(n1)
	# toRemove=[]
	# THRESHOLD=0.2
	# print("[-] Removing THRESHOLD")
	# for (n1, n2, w) in G.edges.data('weight'):
	# 	if float(w)>THRESHOLD:
	# 		print("[X] Removed node THRESHOLD")
	# 		toRemove.append((n1,n2))

	# for n1,n2 in toRemove:
	# 	G.remove_edge(n1,n2)
	
	print("[-] Computing graphs")
	lst_results = []
	connected_components = list(nx.connected_component_subgraphs(G))
	dic_conn = {}
	for i, graph in enumerate(connected_components):
		lst_temp = [i, graph.number_of_nodes()]
		num_nodes = graph.number_of_nodes()
		if not num_nodes in dic_conn:
			dic_conn[num_nodes] = 0
		dic_conn[num_nodes] += 1
		print(">>>%d<<<"%(i))
		print("\tNumber of nodes:%d" %(graph.number_of_nodes()))
		#print(nx.info(graph))
		for j in graph.nodes():
			lst_temp.append(j)
		lst_results.append(lst_temp)
	print(dic_conn)
	for k, v in dic_conn.items():
		print("There are %d groups with %d components" % (v,k))

	for k, v in dic_conn.items():
		print("\t%d\t%d" % (v,k)) 
	gen_csv_from_tuples("xd.csv", [], lst_results)

	#dirname='/Users/sergio/Documents/Teaching/TFM_JoseCabrero/subgraphsREMOVED_w0.2/'	# 


def process_data(lst):
	max_items = [0 for i in range(len(lst[0][2:]))]
	num_metrics = len(max_items)
	num_users = len(lst)
	array = np.memmap('matrix.dat', dtype=float, mode ='r', shape=(num_users, num_metrics))
	
	print("[-] Generating dataframe...")
	print(array[:3])
	a = [x[0] for x in lst]
	b = [x[1] for x in lst]
	df = pd.DataFrame({ 'from': a, 'to': b})

	# Build your graph
	#G=nx.from_pandas_dataframe(df, 'from', 'to')
	print("[-] Generating edgelist...")
	G = nx.from_pandas_edgelist(df, 'from', 'to')
	plt.figure(figsize=(50,50))
	node_color = [100 * G.degree(node) for node in G]
	node_size =  [1000 * G.degree(node) for node in G]
	#pos = nx.spring_layout(G, k=0.04)
	graph = nx.draw_spring(G, k=0.09, with_labels=True, node_size=node_size, 
		node_color=node_color, node_shape="o", 
		alpha=0.5, linewidths=4, font_size=25, 
		font_color="black", font_weight="bold", 
		width=2, edge_color="grey")
	
	#plt.savefig("Graph_spring.png", format="PNG")
	# graph = nx.draw_spectral(G, with_labels=True, node_size=node_size, 
	# 	node_color=node_color, node_shape="o", 
	# 	alpha=0.5, linewidths=4, font_size=25, 
	# 	font_color="black", font_weight="bold", 
	# 	width=2, edge_color="grey")
	# plt.savefig("Graph_spectral.png", format="PNG")
	# #graph = nx.draw_planar(G, with_labels=True, node_size=node_size, 
	# 	#node_color=node_color, node_shape="o", 
	# 	#alpha=0.5, linewidths=4, font_size=25, 
	# 	#font_color="black", font_weight="bold", 
	# 	#width=2, edge_color="grey")
	# #plt.savefig("Graph_planar.png", format="PNG")
	# graph = nx.draw_shell(G, with_labels=True, node_size=node_size, 
	# 	node_color=node_color, node_shape="o", 
	# 	alpha=0.5, linewidths=4, font_size=25, 
	# 	font_color="black", font_weight="bold", 
	# 	width=2, edge_color="grey")
	# plt.savefig("Graph_shell.png", format="PNG")
def find_user_list(user1, user2, lst):
	u1, u2 = None, None
	for i in lst:
		if u1 is None or u2 is None:
			if i[0] == user1:
				u1 = i
			elif i[0] == user2:
				u2 = i
		else:
			u1 = set(list(u1[1:]))
			u2 = set(list(u2[1:]))
			#print(u1, u2)
			return u1, u2
	return None, None

def get_most_important():
	lst = read_csv_list("weighted_average.csv")[1:]
	dictio_lst = {}
	status.create_numbar(100, len(lst_res2))
	for indi, i in enumerate(lst_res2):
		status.update_numbar(indi, len(lst_res2))
		dictio_lst[i] = read_csv_list(i+"_files/user_to_"+i+".csv")[1:]
	status.end_numbar()
	final_lst = []
	num = 100
	status.create_numbar(100, num)
	for indi, i in enumerate(lst[:num]):
		status.update_numbar(indi, num)
		userilist = [i[0],i[1],i[2]]
		for key in lst_res2:
			u1, u2 = find_user_list(i[0], i[1], dictio_lst[key])
			if u1 is None or u2 is None:
				continue
			userilist+= list(u1.intersection(u2))
		final_lst.append(tuple(userilist))
	status.end_numbar()
	gen_csv_from_tuples("croos_val.csv", [ 'user_a', 'user_b', 'metric', 'similar_vals'], final_lst)


			


def store_dictio_to_file():
	global dictio_of_results
	lst = []
	headers = ['user_a', 'user_b'] + [x + y for x in lst_res for y in ["_euc", "_dis"]]
	for indi, k1 in enumerate(list(dictio_of_results.keys())):
		status.update_numbar(indi, len(dictio_of_results))
		for k2 in dictio_of_results[k1].keys():
			temp_lst = [k1, k2]
			for k3 in lst_res:
				for k4 in ["_euc", "_dis"]:
					if (k3 + k4) in dictio_of_results[k1][k2].keys():
						temp_lst.append(dictio_of_results[k1][k2][k3+k4])
					else:
						temp_lst.append(-1)
			lst.append(tuple(temp_lst))

	gen_csv_from_tuples("combination.csv", headers, lst)


def gen_new_dataset():
	lst = read_csv_list("combination.csv")
	return lst

def gen_dictio_from_csv():
	global dictio_of_results
	lst = read_csv_list("combination.csv")
	headers, lst = lst[0][2:], lst[1:]
	status.create_numbar(100, len(lst))
	for indi, i in enumerate(lst):
		status.update_numbar(indi, len(lst))
		if not i[0] in dictio_of_results.keys():
			dictio_of_results[i[0]] = {}
		if not i[1] in dictio_of_results[i[0]].keys():
			dictio_of_results[i[0]][i[1]] = {}

		for indj, j in enumerate(i[2:]):
			dictio_of_results[i[0]][i[1]][headers[indj]] = j
	status.end_numbar()
	return lst

def generate_weights_dictio():
	join_dis_results_csv()
	join_euc_results_csv()
	store_dictio_to_file()

def gen_dictionary_from_file():
	#lst = gen_dictio_from_csv()
	lst = gen_new_dataset()
	headers, lst = lst[0][2:], lst[1:]
	return headers, lst

def gen_and_normalize_matrix(headers, lst):
	normalize_data(headers, lst)
	weighted_average(headers, lst)
def main():
	process_pickle()
	return
	#generate_weights_dictio()
	headers, lst = gen_dictionary_from_file()
	gen_and_normalize_matrix(headers, lst)
	get_most_important()
	generate_graph()

if __name__ == "__main__":
	main()
