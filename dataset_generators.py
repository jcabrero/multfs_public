from common_utils import gen_csv_from_tuples, read_csv_list, make_query, join_all_results
#import psycopg2
import os, time, string, math
import itertools as it
import status, sys
import getpass
import multiprocessing as mp
# from extract_class import FeatureScore
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from common_utils import create_dir
from functools import partial
import ipaddress
import socket,getpass
#only on linux 
from common_utils import get_ram, get_elapsed_time
#import psycopg2
# List of ids of non-english forums to exclude for trigram analysis
nonEnglishForums=[9,10,15,17,23,24,25]
database_connections={}

def extract_user_to_link_csv(filename):
	tic = time.time()
	print("[+] Generating Link dataset")
	query= """WITH "A" AS (SELECT
  		CAST("Post"."Author" AS text) || '[' || CAST("Post"."Site" AS text) || ']' as "Author",
  		regexp_matches( "Content", '(http[s]?://(?:[a-zA-Z]|[0-9]|[$-\)+-Z^-_@.&+]|[!\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)', 'g') AS "link"
  		FROM "Post" WHERE "Content" ~ '(http[s]?://(?:[a-zA-Z]|[0-9]|[$-\)+-Z^-_@.&+]|[!\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)'),
		"B" AS (SELECT "Author", lower("link"[1]) as "link", count(*) as "repetitions" FROM "A" GROUP BY "Author", "link" )
		SELECT "B"."Author",
		string_agg(LEFT(CAST("B"."link" AS text), 500) || '[' || CAST("B"."repetitions" AS text) || ']', ', ') as "reps" 
		FROM "B" GROUP BY "B"."Author";"""	
	rows = make_query(query)
	rows = [row[:1] + tuple([x for x in row[1].split(", ")],) for row in rows if row[0] != str(-1)]
	gen_csv_from_tuples(filename, ["IdAuthor", "link"], rows)
	print("[+] Finished generating Link dataset in %d seconds" % (time.time() - tic))

def check_ip (ip):
	if ip.count('.')!=3:
		return False
	try:
		ipaddress.ip_address(ip)
		return True
	except:
		return False

def extract_user_to_ip_csv(filename):
	tic = time.time()
	print("[+] Generating IP Address dataset")
	query= """WITH "A" AS (SELECT
		CAST("Post"."Author" AS text) || '[' || CAST("Post"."Site" AS text) || ']' as "Author",
		-- regexp_matches( "Content", '(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)', 'g') AS "ip"
		regexp_matches( "Content", '([0-9]+25[0-5]|[0-9]+2[0-4][0-9]|[0-9]+[01]?[0-9][0-9]?)\.([0-9]+25[0-5]|[0-9]+2[0-4][0-9]|[0-9]+[01]?[0-9][0-9]?)\.([0-9]+25[0-5]|[0-9]+2[0-4][0-9]|[0-9]+[01]?[0-9][0-9]?)\.(25[0-5]|[0-9]+2[0-4][0-9]|[0-9]+[01]?[0-9][0-9]?)', 'g') AS "ip"
		FROM "Post"	WHERE "Content" ~ '(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'),
		"B" AS (SELECT "Author", "ip", count(*) as "repetitions" FROM "A" GROUP BY "Author", "ip" )
		SELECT "B"."Author",
		string_agg("B"."ip"[1] || '.' ||"B"."ip"[2] || '.' ||"B"."ip"[3]|| '.' ||"B"."ip"[4] || '[' || CAST("B"."repetitions" AS text) || ']', ', ') as "reps" 
		FROM "B" GROUP BY "B"."Author";"""
	rows = make_query(query)
	rows = [row[:1] + tuple([x for x in row[1].split(", ")],) for row in rows if row[0] != str(-1)]
	cleaned_rows=[]
	for r in rows:
		cleaned=[r[0]]
		for ip in r[1:]:
			if check_ip(ip.split('[')[0]):
				cleaned.append(ip)
		cleaned_rows.append(tuple(cleaned))

	gen_csv_from_tuples(filename, ["IdAuthor", "IP"], cleaned_rows)
	print("[+] Finished generating IP Address dataset in %d seconds" % (time.time() - tic))

def extract_user_to_email_csv(filename):
	tic = time.time()
	print("[+] Generating Email dataset")
	query= """WITH "A" AS (SELECT
  		CAST("Post"."Author" AS text) || '[' || CAST("Post"."Site" AS text) || ']' as "Author",
  		regexp_matches( "Content", '(?:(?![*]))([A-Za-z0-9\._%-\)\+]+@[A-Za-z0-9\.-]+[.][A-Za-z]+)', 'g') AS "email"
  		FROM "Post" WHERE "Content" ~ '(?:(?![*]))([A-Za-z0-9\._%-\)\+]+@[A-Za-z0-9\.-]+[.][A-Za-z]+)'),
		"B" AS (SELECT "Author", lower("email"[1]) as "email", count(*) as "repetitions" FROM "A" GROUP BY "Author", "email" )
		SELECT "B"."Author",
		string_agg(CAST("B"."email" AS text) || '[' || CAST("B"."repetitions" AS text) || ']', ', ') as "reps" 
		FROM "B" GROUP BY "B"."Author";"""
	rows = make_query(query)
	rows = [list(row[:1] + tuple([x for x in row[1].split(", ")],)) for row in rows if row[0] != str(-1)]
	for row in range(len(rows)):
		for col in range(1, len(rows[row])):
			if len(rows[row][col]) > len("***LINK***") and rows[row][col][:len("***LINK***")] == "***LINK***":
				rows[row][col] = rows[row][col][len("***LINK***"):]	

	for row in range(len(rows)):
		rows[row] = (rows[row][0],) + tuple(set(rows[row][1:]))
	gen_csv_from_tuples(filename, ["IdAuthor", "email"], rows)
	print("[+] Finished generating Email dataset in %d seconds" % (time.time() - tic))

def extract_user_to_skype_csv(filename):
	tic = time.time()
	print("[+] Generating Skype dataset")
	query= """WITH "A" AS (SELECT
  		CAST("Post"."Author" AS text) || '[' || CAST("Post"."Site" AS text) || ']' as "Author", 
  		regexp_matches( "Content", 'skype\s*:\s*([a-zA-Z0-9:\.]{1,37})', 'g') AS "skype"
  		FROM "Post" WHERE "Content" ~ 'skype\s*:\s*([a-zA-Z0-9:\.]{1,37})'),
		"B" AS (SELECT "Author", lower("skype"[1]) as "skype", count(*) as "repetitions" FROM "A" GROUP BY "Author", "skype" )
		SELECT "B"."Author",
		string_agg(CAST("B"."skype" AS text) || '[' || CAST("B"."repetitions" AS text) || ']', ', ') as "reps" 
		FROM "B" GROUP BY "B"."Author";"""
	rows = make_query(query)
	rows = [list(row[:1] + tuple([x for x in row[1].split(", ")],)) for row in rows if row[0] != str(-1)]
	for row in range(len(rows)):
		for col in range(1, len(rows[row])):
			if rows[row][col][-1] == '.':
				rows[row][col] = rows[row][col][:-1]
				
	for row in range(len(rows)):
		rows[row] = (rows[row][0],) + tuple(set(rows[row][1:]))

	gen_csv_from_tuples(filename, ["IdAuthor", "skype"], rows)
	print("[+] Finished generating Skype dataset in %d seconds" % (time.time() - tic))

def extract_user_to_btc_csv(filename):
	tic = time.time()
	print("[+] Generating Bitcoin dataset")
	query= """WITH "A" AS (SELECT
  		CAST("Post"."Author" AS text) || '[' || CAST("Post"."Site" AS text) || ']' as "Author",
  		regexp_matches( "Content", '\y([13][a-km-zA-HJ-NP-Z1-9]{25,34})\y', 'g') AS "btc"
  		FROM "Post" WHERE "Content" ~ '([13][a-km-zA-HJ-NP-Z1-9]{25,34})'),
		"B" AS (SELECT "Author", "btc"[1] as "btc", count(*) as "repetitions" FROM "A" GROUP BY "Author", "btc" )
		SELECT "B"."Author",
		string_agg(CAST("B"."btc" AS text) || '[' || CAST("B"."repetitions" AS text) || ']', ', ') as "reps" 
		FROM "B" GROUP BY "B"."Author";"""

	rows = make_query(query)
	rows = [list(row[:1] + tuple([x for x in row[1].split(", ")],)) for row in rows if row[0] != str(-1)]
	for row in range(len(rows)):
		for col in range(1, len(rows[row])):
			if rows[row][col][-1] == '.':
				rows[row][col] = rows[row][col][:-1]
				
	for row in range(len(rows)):
		rows[row] = (rows[row][0],) + tuple(set(rows[row][1:]))
	gen_csv_from_tuples(filename, ["IdAuthor", "btc"], rows)
	print("[+] Finished generating Bitcoin dataset in %d seconds" % (time.time() - tic))



def get_user_posts(user,connector=None):
	user_id = user[0]
	site_id = user[1]
	#print("[- -] Extracting from DB")
	#query = """SELECT "IdPost", "Content", "Timestamp" FROM "Post" WHERE "Author" = %d AND "Site" = %d;""" % (user_id, site_id)
	query = """SELECT  "Content" FROM "Post" WHERE "Author" = %d AND "Site" = %d;""" % (user_id, site_id)
	rows = make_query(query,conn=connector)
	rows = [row[0] for row in rows]
	return rows
	#print("[+ +] Done extracting from DB")



def get_user_timestamps(user):
	user_id = user[0]
	site_id = user[1]
	#print("[- -] Extracting from DB")
	query = """SELECT "Timestamp" FROM "Post" WHERE"Author" = %d AND "Site" = %d;""" % (user_id, site_id)
	rows = make_query(query)
	rows = [(row[0].weekday(), row[0].hour, row[0].minute) for row in rows]
	dic_result = dict()

	minute_intervals = 10
	residual_prob = 1 / minute_intervals
	total_intervals = 60 // 10
	# O(28)
	for i in range(7):
		for j in range(24):
			for z in range(total_intervals):
				dic_result[(i,j,z)] = 0.0
	# O(n)

	for row in rows:
		hour = row[1]
		minute = row[2] // minute_intervals

		dic_result[(row[0], hour, minute)] += 1
		dic_result[(row[0], hour, (minute - 1) % total_intervals)] += residual_prob
		dic_result[(row[0], hour, (minute + 1) % total_intervals)] += residual_prob
	# O(28)
	tup = (site_id, "%d[%d]" % (user_id, site_id),) + tuple(["%s|%s|%s[%s]" % (k[0], k[1], k[2], v) for k, v in dic_result.items() if v > 0])
	return tup

def extract_timestamps(members, filename='user_to_timestamp.csv'):
	p = mp.Pool(12)
	result = p.map(get_user_timestamps, members)
	result = [x[1:] for x in result if x[0] != 4] # We manually exclude forum 4 because it is "undefined"
	gen_csv_from_tuples(filename, ["IdAuthor", "0,0"], result)
	p.close()

def do_extract_timestamps(members):
	num_users = len(members)
	tic = time.time()
	print("Extracted %d users" % (num_users))
	dirname = 'timestamp_files/'
	create_dir(dirname)
	divs = 100000
	i = 0
	total_size = 0
	while total_size < num_users:
		print("User Timestamps", "[%d Users Processed]" %(total_size), "[%0.3f Percentage]" % ((total_size / len(members)) * 100), get_ram(), get_elapsed_time(tic))
		extract_timestamps(members[total_size: total_size+divs], dirname + "user_to_timestamp.csv_%d" % (i)) 
		i += 1
		total_size += divs
		print(i, total_size, time.time() - tic )
	join_all_results(dirname+"user_to_timestamp.csv")


def get_n_grams(content, n):
	#content = content.replace(" ", "_")
	stop_words = set(stopwords.words('english'))	
	string = re.sub(r'\*{3}IMG\*{3}.*?\*{3}IMG\*{3}', ' ', content)
	string = re.sub(r'\*{3}CITING\*{3}.*?\*{3}CITING\*{3}', ' ', string)
	string = re.sub(r'\*{3}IFRAME\*{3}.*?\*{3}IFRAME\*{3}', ' ', string)
	string = re.sub(r'\*{3}LINK\*{3}.*?\*{3}LINK\*{3}', ' ', string)
	string = re.sub(r'\*{3}ATTACHMENT\*{3}.*?\*{3}ATTACHMENT\*{3}', ' ', string)
	string = re.sub(r'\*{3}CODE\*{3}.*?\*{3}CODE\*{3}', ' ', string)

	string = re.sub(r'\d+', ' num ', string)
	string = re.sub(r'\$', ' dol ', string)
	content = re.sub(r'\€', ' eur ', string)

	content = content.lower()

	content = "".join([x + ("_" * ((n - len(x)) % n)) for x in content.split(" ") if len(x) > 0 and x not in stop_words and x.isalpha()])
	return [content[i:i+n] for i in range(0, len(content), n)]

def get_all_trigrams(content):
	return get_n_grams(content, 3)


def do_extract_user_trigrams(member):
	global database_connections
	currentProcess=mp.current_process()
	if not currentProcess in database_connections:
		connector = psycopg2.connect(user=getpass.getuser(), database='crimebb')
		database_connections[currentProcess]=connector
	else:
		connector=database_connections[currentProcess]
	posts = get_user_posts(member,connector) 
	merged_posts = "\n".join(posts)
	if len(merged_posts)<600:
		return tuple()
	dict_results = dict()
	for trigram in get_all_trigrams(merged_posts):
		if trigram in dict_results:
			dict_results[trigram] += 1
		else: 
			dict_results[trigram] = 1

	list_trigrams = ["%s[%s]" % (member[0], member[1])] + ["%s[%d]" % (k, v) for k, v in dict_results.items()]
	return tuple(list_trigrams)

def do_extract_users_trigrams_mp(members):
	dict_trigrams = dict()
	tic = time.time()
	batch_size = 20000
	dirname = 'trigram_files/'
	create_dir(dirname)
	
	# We assign an index to each element in the order they are stored in the file.
	# Additionally, we remove those trigrams written only by one user.
	#dict_trigrams = {x[0]:i for i, x in enumerate(read_csv_list(dirname+"trigrams.csv")[1:]) if int(x[1]) > 1}
	print(len(dict_trigrams))
	# We make the computations in batches of information so that we do not fill the memory. 
	# Those batches are stored little by little.
	#partial_function = partial(do_extract_user_trigrams, dict_trigrams)
	for i in range(0, len(members), batch_size):
		print("Users Trigrams", "[%d Users Processed]" %(i), "[%0.3f Percentage]" % ((i / len(members)) * 100), get_ram(), get_elapsed_time(tic))
		batch = members[i:i+batch_size]
		p = mp.Pool(16)
		users_trigrams = p.map(do_extract_user_trigrams, batch)
		p.close()
		# We only store lengths bigger than 1
		gen_csv_from_tuples(dirname + "user_to_trigram.csv_%d"%(i), ["IdAuthor", "IdTrigrams"], [x for x in users_trigrams if len(x) > 1] )

	#fs = FeatureScore(None, None, None, None, None, None, None,None, None,None, None)
	join_all_results(dirname+"user_to_trigram.csv")
	return dict_trigrams

def get_trigrams_from_member(member):
	posts = get_user_posts(member) 
	merged_posts = "\n".join(posts)
	list_trigrams = get_all_trigrams(merged_posts)
	return list_trigrams

def do_extract_trigrams(members):
	# This function extracts the trigrams from users. For each user we extract all the trigrams and then we add them to the dictionary of trigrams.
	# The dictionary of trigrams counts all the trigrams that exist and with each new appearance it adds information of how many times it appeared.

	dict_trigrams = dict()
	tic = time.time()
	dirname = 'trigram_files/'
	create_dir(dirname)
	for i, member in enumerate(members):
		if i % 1000 == 0:
			print("[%d Users Processed]"%(i), "[%d Trigrams]" % (len(dict_trigrams)), "[%0.3f Percentage]" % ((i / len(members)) * 100), get_ram(), get_elapsed_time(tic))
			if i % 10000 == 0:
				print("Saving data...")
				gen_csv_from_tuples(dirname + "trigrams.csv", ["trigram", "usage"], [(k,v) for k,v in dict_trigrams.items()])
				
		for elem in get_trigrams_from_member(member):
			if elem in dict_trigrams:
				dict_trigrams[elem] += 1
			else:
				dict_trigrams[elem] = 1
	print("Saving data...")
	gen_csv_from_tuples(dirname + "trigrams.csv", ["trigram", "usage"], [(k,v) for k,v in dict_trigrams.items()])
	return dict_trigrams

def simplify_list():
	dirname = 'trigram_files/'
	lst = read_csv_list(dirname+"user_to_trigrams_complex.csv")[1:]
	lst =  [(x[0],)+tuple([int(y.split(':')[0]) for y in x[1:]]) for x in lst ]
	gen_csv_from_tuples(dirname+'user_to_trigrams.csv', ['user','trigrams_#'], lst)


def do_extract_trigrams_mp(members):
	dict_trigrams = dict()
	tic = time.time()
	batch_size = 10000
	dirname = 'trigram_files/'
	create_dir(dirname)
	for i in range(0, len(members), batch_size):
		p = mp.Pool(12)
		list_trigrams = p.map(get_trigrams_from_member, members[i:i+batch_size])
		print("Trigrams","[%d Users Processed]" %(i), "[%d Trigrams]" % (len(dict_trigrams)), "[%0.3f Percentage]" % ((i / len(members)) * 100), get_ram(), get_elapsed_time(tic))

		for l in list_trigrams:
			for elem in l:
				if elem in dict_trigrams:
					dict_trigrams[elem] += 1
				else:
					dict_trigrams[elem] = 1
		p.close()
	print("Saving data...")
	gen_csv_from_tuples(dirname + "trigrams.csv", ["trigram", "usage"], [(k,v) for k,v in dict_trigrams.items()])
	return dict_trigrams

def extract_metrics_for_user(member):
	global database_connections
	currentProcess=mp.current_process()
	if not currentProcess in database_connections:
		connector = psycopg2.connect(user=getpass.getuser(), database='crimebb')
		database_connections[currentProcess]=connector
	else:
		connector=database_connections[currentProcess]
	posts = get_user_posts(member,connector) 	
	if member[0] == -1:
		return ["%s[%s]" % (member[0], member[1])]
	len_posts = [len(x) for x in get_user_posts(member,connector=connector)]
	num_posts = len(len_posts)
	return tuple(["%s[%s]" % (member[0], member[1]), num_posts] + len_posts)

def extract_chars_per_user(members):
	dirname = 'num_files/'
	filename = 'user_to_num.csv'
	tic = time.time()
	batch_size = 24000
	create_dir(dirname)
	for i in range(0, len(members), batch_size):
		p = mp.Pool(16)
		lst_pr = p.map(extract_metrics_for_user, members[i:i+batch_size])
		print("Chars per User", "[%d Users Processed]" %(i), "[%0.3f Percentage]" % ((i / len(members)) * 100), get_ram(), get_elapsed_time(tic))
		p.close()
		# We only store lengths bigger than 1
		gen_csv_from_tuples(dirname + filename + "_%d"%(i), ["IdAuthor", "IdTrigrams"], lst_pr)

	join_all_results(dirname+filename)

def simplify():
	import adhoc_removal
	from functional import seq
	files = ['trigram_files/user_to_trigram.csv', 'timestamp_files/user_to_timestamp.csv']
	keep_users = adhoc_removal.keep_users
	for file in files:
		a = read_csv_list(file)
		print("Initial length: %d" % (len(a)))
		a = seq(a).filter(lambda x: x[0] in keep_users).filter(lambda x: len(x) > 1)
		a = [tuple(x) for x in a]
		print("Final length: %d" % (len(a)))
		gen_csv_from_tuples(file+'_simple', ['IdAuthor', 'Features'], a)

def extract_more_features():
	
	# Get all user different users
	query = """ SELECT "IdMember", "Site" FROM "Member" WHERE "IdMember" != -1;"""
	members = make_query(query)
	members = [(int(x[0]), int(x[1]))for x in members]

	do_extract_timestamps(members)
	#do_extract_trigrams_mp(members)
	#do_extract_users_trigrams_mp(members)
	#extract_chars_per_user(members)


def extract_user_trigrams():
	query = """ SELECT "IdMember", "Site" FROM "Member" WHERE "IdMember" != -1;"""
	members = make_query(query)
	members = [(int(x[0]), int(x[1]))for x in members if not int(x[1]) in nonEnglishForums]
	do_extract_users_trigrams_mp(members)



def create_directories_and_datasets():
	create_dir('email_files')
	extract_user_to_email_csv("email_files/user_to_email.csv")
	create_dir('btc_files')
	extract_user_to_btc_csv("btc_files/user_to_btc.csv")
	create_dir('ip_files')
	extract_user_to_ip_csv("ip_files/user_to_ip.csv")
	create_dir('skype_files')
	extract_user_to_skype_csv("skype_files/user_to_skype.csv")
	create_dir('link_files')
	extract_user_to_link_csv("link_files/user_to_link.csv")
	extract_user_trigrams()

def create_directories_and_datasets_1():
	create_dir('btc_files')
	extract_user_to_btc_csv("btc_files/user_to_btc.csv")
	create_dir('email_files')
	extract_user_to_email_csv("email_files/user_to_email.csv")
	create_dir('ip_files')
	extract_user_to_ip_csv("ip_files/user_to_ip.csv")
	create_dir('skype_files')
	extract_user_to_skype_csv("skype_files/user_to_skype.csv")

def create_directories_and_datasets_2():
	create_dir('link_files')
	extract_user_to_link_csv("link_files/user_to_link.csv")
	
def get_user_site(user):
	pos_bracket = user.find("[")
	user_id = int(user[:pos_bracket])
	site_id = int(user[pos_bracket + 1:-1])
	return user_id, site_id

def generate_user_dataset(user, uind, total):
	print("[-] Going for user %d/%d - %s" %(uind, total,  user))
	tic = time.time()
	user_id, site_id = get_user_site(user)
	print("[- -] Extracting from DB")
	query = """SELECT "IdPost", "Content" FROM "Post" WHERE "Author" = %d AND "Site" = %d;""" % (user_id, site_id)
	rows = make_query(query)
	rows = [(row[0], row[1]) for row in rows]
	print("[+ +] Done extracting from DB")
	#a = string.ascii_lowercase
	#b = math.ceil(float(len(rows)) / float(len(a)) )
	#names = ["".join(elem) for iter in [it.product(a, repeat=i) for  i in range(1,b + 1)] for elem in iter]
	directory = 'Author/' + user + "/"
	create_dir(directory)
	print("[- -] Generating files for user, total: %d" % (len(rows)))
	for ind, content in enumerate(rows):
		filename = str(user_id) + "-" + str(site_id) + "-" + str(content[0])
		with open(directory + filename + ".txt", 'w+') as file:  
			file.write(content[1])
	print("[+ +] Generating files for user, total: %d" % (len(rows)))
	print("[+] Going for user %d - %s" %(uind, user))

def generate_directories_for_users():
	print("[>] Creating dir")
	create_dir("Author/")
	print("[>] Reading user csv list")
	lst_users = read_csv_list("weighted_average.csv")[1:]

	#lst_users = [(x[0], x[1] for x in lst_users if float(x[2]) < 0.35)
	ev_set = set()
	for entry in lst_users:
		if float(entry[2]) >= 0.35:
			break
		ev_set.add(entry[0])
		ev_set.add(entry[1])
	#status.create_numbar(100, len(ev_set))
	for ind, user in enumerate(ev_set):
		#status.update_numbar(ind, len(ev_set))
		generate_user_dataset(user, ind, len(ev_set))
	#status.end_numbar()

def swap_files(a, b):
	temp = a+"_temp"
	os.rename(a, temp)
	os.rename(b, a)
	os.rename(temp, b)

def main():
	if len(sys.argv) < 2:
		print("""Usage: python3 dataset_generators.py <option>
	<option>:
		'datasets': generate datasets and directories
		'authorfolders': generate folders for users
		""")
		return

	if sys.argv[1] == 'authorfolders':
		print("[>>] Doing all")
		generate_directories_for_users()

	elif sys.argv[1] == 'datasets':
		print("[>>] NONE")
		create_directories_and_datasets()
		#extract_more_features()
	elif sys.argv[1] == 'morefeatures':
		print("[--] Extracting more features")
		extract_more_features()
		simplify()
	elif sys.argv[1] == 'simplify':
		simplify()
		print("Swapping files")
		files = ['trigram_files/user_to_trigram.csv', 'timestamp_files/user_to_timestamp.csv']
		for i in files:
			swap_files(i, i+"_simple")
	elif sys.argv[1] == 'trigrams':
		# create_dir('link_files')
		# extract_user_to_link_csv("link_files/user_to_link.csv")
		extract_user_trigrams()
	elif sys.argv[1]=='user_metrics':
		query = """ SELECT "IdMember", "Site" FROM "Member" WHERE "IdMember" != -1;"""
		members = make_query(query)
		members = [(int(x[0]), int(x[1]))for x in members]		
		extract_chars_per_user(members)
if __name__ == "__main__":
	join_all_results('trigram_files/user_to_trigram.csv')
	main()


