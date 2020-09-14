import re
from common_utils import gen_csv_from_tuples, read_csv_list, pickle_object, unpickle_object
import numpy as np
import math
import status
import time
import os
import unicodedata as ud

def user_removal_based_on_participation():
	
	keep_users_p = 'num_files/keep_users.pkl'
	if os.path.exists(keep_users_p):
		#print("\tUser participation extraction exists", end='\r')
		keep_users = unpickle_object(keep_users_p)
		#print("\t[END] User participation extraction finished [%d Users to keep]" % (len(keep_users)))
		return keep_users

	lst = read_csv_list('num_files/user_to_num.csv')[1:]
	#print("\t[-] User participation detection.", end='\r')
	users = [i[0] for i in lst]
	# Number of posts per user
	x = np.array([int(x[1]) for x in lst])
	# Characters per user post
	y = [np.array([int(y) for y in x[2:]]) for x in lst]

	# Average characters per post of a user
	z = np.array([i.mean() for i in y if len(i) > 0])
	keep_users = set()
	limi = np.quantile(x, .50)
	limk = np.quantile(z, .50)
	for user, i, k in zip(users, x, z):
		if i > limi or k > limk:
			keep_users.add(user)

	pickle_object(keep_users, keep_users_p)
	#print('[END] Extracted all the user participations [%d]' % (len(keep_users)))
	return keep_users
	#xg = [np.quantile(x, i) for i in np.arange(0,1.01, 0.01)]
	#zg = [np.quantile(z, i) for i in np.arange(0,1.01, 0.01)]

keep_users = user_removal_based_on_participation()

def trigram_value_removal(user_ind, value_ind, dictio_of_users, dictio_of_values):
	##print("\t[-] Removing less used trigrams", end='\r')
	usage = [(k, len(v)) for k, v in dictio_of_values.items() if len(v) > 10]
	usage = sorted(usage, key=lambda x: x[1], reverse=True)
	##print("\t[+] Ended finding less used trigrams")
	return [x[0] for x in usage[10_000:]]

def trigram_non_european_chars_removal(user_ind, value_ind, dictio_of_users, dictio_of_values):
	def is_latin(trigram):
		return all(['LATIN' in ud.name(uchr) for uchr in trigram if uchr.isalpha()])

	def is_ascii(trigram):
		return all(ord(c) < 128 for c in trigram)

	to_keep = [vind for value, vind in value_ind.items() if is_ascii(value)]
	return to_keep
		
def general_user_removal(user_ind, value_ind, dictio_of_users, dictio_of_values):
	#print("\t[-] General user removal by post size")
	lst_keep = []
	for user, uind in user_ind.items():
		if user in keep_users:
			lst_keep.append(uind)
	return lst_keep

def ip_value_removal(user_ind, value_ind, dictio_of_users, dictio_of_values):
	#print("\t[+] Removing reserved IP addresses...")
	keep_list = []
	for ip, index in value_ind.items():
		if re.search(r'192\.168\.\d{1,3}\.\d{1,3}', 
			ip,  re.IGNORECASE | re.DOTALL | re.VERBOSE | re.MULTILINE):
			continue
		elif re.search(r'172\.\d{1,3}\.\d{1,3}\.\d{1,3}', 
			ip,  re.IGNORECASE | re.DOTALL | re.VERBOSE | re.MULTILINE):
			continue
		elif re.search(r'10\.\d{1,3}\.\d{1,3}\.\d{1,3}', 
			ip,  re.IGNORECASE | re.DOTALL | re.VERBOSE | re.MULTILINE):
			continue
		elif ip == '127.0.0.1':
			continue
		keep_list.append(index)
	return keep_list

def ip_user_removal(user_ind, value_ind, dictio_of_users, dictio_of_values):
	#print("\t[-] Removing IP addresses appearing more than 30 times")
	return [key for key, values in dictio_of_users.items() if len(values) < 30]

def skype_user_removal(user_ind, value_ind, dictio_of_users, dictio_of_values):
	#print("\t[-] Removing Skype appearing more than 5 times")
	return [key for key, values in dictio_of_users.items() if len(values) < 5]

def email_user_removal(user_ind, value_ind, dictio_of_users, dictio_of_values):
	#print("\t[-] Removing emails appearing more than 5 times")
	return [key for key, values in dictio_of_users.items() if len(values) < 5]

def link_user_removal(user_ind, value_ind, dictio_of_users, dictio_of_values):
	keep_user_list = []
	dictio_of_sites = {
		"hackforums.net": 0,
		"mpgh.net": 4,
		"raidforums.com": 12,
		"antichat.ru": 10,
		"blackhatworld.com": 8,
		"garage4hackers.com": 7,
		"greysec.net": 6,
		"stresserforums.net": 5,
		"kernelmode.info": 1,
		"safeskyhacks.com": 13,
		"offensivecommunity.net": 3
	}
	list_sitenums = [v for _, v in dictio_of_sites.items()]
	for user, uind in user_ind.items():
		user_site = int(user[len(user) - user[::-1].find('['):-1])
		if user_site in list_sitenums:
			keep_user_list.append(uind)
	return keep_user_list
def link_value_removal_2(user_ind, value_ind, dictio_of_users, dictio_of_values):
	#print("\t[+] Highlighting links to other forums...")
	dictio_of_sites = {
		'hackforums.net':0,
		'www.kernelmode.info':1,
		'thehub7xbw4dc5r2.onion':2,
		'offensivecommunity.net':3,
		'www.mpgh.net':4,
		'greysec.net':6,
		'garage4hackers.com':7,
		'blackhatworld.com':8,
		'forum.antichat.ru':10,
		'www.raidforums.com':12,
		'www.safeskyhacks.com':13,
		'stresserforums.me':18,
		'dreadditevelidot.onion':19,
		'torum6uvof666pzw.onion':20,
		'envoys5appps3bin.onion':21,
		'germanyruvvy2tcw.onion':23,
		'lwplxqzvmgu43uff.onion':24,
		'www.unknowncheats.me':26,
	}
	
	def get_num_site(link, dictio_of_sites):
		for link_site, num in dictio_of_sites.items():
			if link_site in link:
				return num
		return None
	def is_external_ref(num_site, uinds, inv_user_ind):
		for uind in uinds:
			user = inv_user_ind[uind]
			user_site = int(user[len(user) - user[::-1].find('['):-1])
			if user_site != num_site:
				return True
		return False
	inv_user_ind = {v:k for k,v in user_ind.items()}
	keep_link_list = []
	count_external_refs = 0
	for ind, (link, vind) in enumerate(value_ind.items()):

		uinds = dictio_of_values[vind]
		num_site = get_num_site(link, dictio_of_sites)

		if num_site == None:
			keep_link_list.append(vind)
		elif is_external_ref(num_site, uinds, inv_user_ind):
			keep_link_list.append(vind)

	return keep_link_list

def link_value_removal_keep_params(user_ind, value_ind, dictio_of_users, dictio_of_values):
	#print("\tRemoving short links...")
	keep_link_list = []
	for link, index in value_ind.items():
		if not (link[:-1].count("\t/") > 3) and not (".onion" in link):
			keep_link_list.append(index)
	return keep_link_list
