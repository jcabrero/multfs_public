from common_utils import gen_csv_from_tuples, read_csv_list, read_csv_list2, read_csv_dict, make_query, pickle_object, unpickle_object, send_mail
from common_utils import create_dir
#only on linux 
from common_utils import get_ram, get_elapsed_time
import os

def get_username(user):
	pos_bracket = user.find("[")
	user_id = int(user[:pos_bracket])
	site_id = int(user[pos_bracket + 1:-1])
	#print("[- -] Extracting from DB")
	query = """SELECT "Username", "RegistrationDate", "LastPostDate" FROM "Member" WHERE "IdMember" = %d AND "Site" = %d;""" % (user_id, site_id)
	rows = make_query(query)
	username = rows[0][0]
	regdate = rows[0][1]
	lastpostdate = rows[0][2]
	return username, regdate, lastpostdate

def get_user_posts(user):
	pos_bracket = user.find("[")
	user_id = int(user[:pos_bracket])
	site_id = int(user[pos_bracket + 1:-1])
	#print("[- -] Extracting from DB")
	query = """SELECT "IdPost", "Content" FROM "Post" WHERE "Author" = %d AND "Site" = %d;""" % (user_id, site_id)
	rows = make_query(query)
	rows = [(row[0], row[1]) for row in rows]
	return rows
	#print("[+ +] Done extracting from DB")

def make_summary_file(directory, string):
	file2 = directory + "000-SUMMARY.txt"
	print(file2)
	with open(file2, 'w+') as file: 
		file.write(string)

def get_coincidences_for_pair(u1, u2, dictios_of_users, user_inds, value_inds):
	lst_coins = []
	for user_ind, value_ind, dictio_of_users in zip(user_inds, value_inds, dictios_of_users):

		if u1 in user_ind and u2 in user_ind:
			print("BOTH")
			uind1, uind2 = user_ind[u1], user_ind[u2]
			vals_1, vals_2 = set(dictio_of_users[uind1]), set(dictio_of_users[uind2])
			inverse_value_ind = {v:k for k, v in value_ind.items()}
			lst_coins += [inverse_value_ind[i] for i in vals_1.intersection(vals_2)]
	return lst_coins

def gen_post_coincidences(lst_coincidences, user1, user2, directory):
	user1_posts = get_user_posts(user1)
	user2_posts = get_user_posts(user2)
	for elem in lst_coincidences:
		directory_elem = elem.replace("/", "").replace(":", "").replace("=", "").replace("\\", "").replace("?", "")[:150]
		filename = directory + directory_elem + ".txt"
		with open(filename, 'w+') as file:
			for (idpost1, post1), (idpost2, post2) in zip(user1_posts, user2_posts):
				if elem in post1:
					file.write("\n[>>>][1]BEGINPOST: ID[%d][1]\n"%(idpost1))
					file.write(post1)
					file.write("\n[<<<][1]ENDPOST: ID[%d][1]\n"%(idpost1))
				if elem in post2:
					file.write("\n[>>>][2]BEGINPOST: ID[%d][2]\n"%(idpost2))
					file.write(post2)
					file.write("\n[<<<][2]ENDPOST: ID[%d][2]\n"%(idpost2))

def gen_coincidences(do):
	directory = "multfs_users/"
	lst_users = read_csv_list("multfs.csv")[1:]
	create_dir(directory)
	dictios_of_users = []
	value_inds = []
	user_inds = []
	
	pairs = lst_users[:5]

	for _id in do:
		dictio_of_users = unpickle_object(_id + "_files/clean_dictio_of_users.pkl")
		dictios_of_users.append(dictio_of_users)
		print(len(dictio_of_users))
		value_ind = unpickle_object(_id + "_files/clean_value_ind.pkl")
		value_inds.append(value_ind)
		user_ind = unpickle_object(_id + "_files/clean_user_ind.pkl")
		user_inds.append(user_ind)

	for index, (u1, u2, _, _, _, _) in enumerate(pairs):
		print("Going for %d" % (index), u1, u2)
		uname1, rg1, lp1 = get_username(u1)
		uname2, rg2, lp2 = get_username(u2)
		directory2 = directory + "%s(%s)-%s(%s)/" %(u1, uname1, u2, uname2)
		create_dir(directory2)
		coins = get_coincidences_for_pair(u1, u2, dictios_of_users, user_inds, value_inds)
		print(coins)
		gen_post_coincidences(coins, u1, u2, directory2)

def gen_latex_post_coincidences(lst_coincidences, user1, user2, file):
	user1_posts = get_user_posts(user1)
	user2_posts = get_user_posts(user2)
	print("GOING FOR USER1 POSTS:", len(user1_posts))
	file.write("\\subsection{USER1: %s}\n"%(user1))
	for (idpost, post) in user1_posts:
		appears = False
		for coin in lst_coincidences:
			if len(coin) == 3:
				coin = coin.replace("_", "") # For trigrams we remove blank space
			if coin in post:
				appears = True
				post = post.replace(coin, "\\textbf{\\color{red} %s}" % (coin))
		if appears:
			file.write("\\subsubsection{[1]POST: ID[%d]}\n" %(idpost))
			file.write(post)
			file.write("\n")
	print("GOING FOR USER1 POSTS:", len(user2_posts))
	file.write("\\subsection{USER2: %s}\n"%(user2))
	for (idpost, post) in user2_posts:
		appears = False
		for coin in lst_coincidences:
			if len(coin) == 3:
				coin = coin.replace("_", "")
			if coin in post:
				appears = True
				post = post.lower().replace(coin, "\\textbf{\\color{red} %s}" % (coin))
		if appears:
			file.write("\\subsubsection{[2]POST: ID[%d]}\n" %(idpost))
			file.write(post)
			file.write("\n")
def gen_latex_coincidences(do,specific_users=[]):
	directory = "multfs_users/"
	lst_users = read_csv_list("multfs.csv")[1:]
	create_dir(directory)
	dictios_of_users = []
	value_inds = []
	user_inds = []
	file = open('analysis.tex', 'w+')
	header = """\\documentclass[12pt]{article}

\\usepackage[utf8]{inputenc}
\\usepackage[T1]{fontenc}
\\usepackage[USenglish]{babel}

\\usepackage{xcolor} % Colors
\\usepackage{tabularx} % Other type of columns
\\usepackage{caption} 
\\usepackage{hyperref}
\\renewcommand{\\baselinestretch}{1.3}


\\usepackage{minted}

\\title{Found pairs of users}
\\author{-}
\\date{}
\\begin{document}

\\maketitle\n"""

	footer = """
\\end{document}"""

	file.write(header)
	# If not specific users are given, take the first 5. Otherwise, take those from the list
	if specific_users is None:
		pairs = lst_users[:5]
	else:
		pairs=[]
		for tuple_list in lst_users:
			(u1, u2, _, _, _, _)=tuple_list
			for specific_user in specific_users:
				if (u1==specific_user and u2 in specific_users) or (u2==specific_user and u1 in specific_users):
					pairs.append(tuple_list)

	for _id in do:

		dictio_of_users = unpickle_object(_id + "_files/clean_dictio_of_users.pkl")
		dictios_of_users.append(dictio_of_users)
		
		value_ind = unpickle_object(_id + "_files/clean_value_ind.pkl")
		value_inds.append(value_ind)
		user_ind = unpickle_object(_id + "_files/clean_user_ind.pkl")
		user_inds.append(user_ind)
		print("ID: %s" %(_id), len(dictio_of_users), len(user_ind))

	for index, (u1, u2, _, _, _, _) in enumerate(pairs):
		#file1name = "tex/%d.tex"%(index)
		#file1 = open(file1name, 'w+')
		print("Going for %d" % (index), u1, u2)
		uname1, rg1, lp1 = get_username(u1) 
		uname2, rg2, lp2 = get_username(u2)
		file.write("\\section{%s(%s)-%s(%s)} \n" %(u1, uname1, u2, uname2))
		#file.write("\\include{%s}\n" % (file1name))
		coins = get_coincidences_for_pair(u1, u2, dictios_of_users, user_inds, value_inds)
		#print("COINCIDENCES", coins)
		gen_latex_post_coincidences(coins, u1, u2, file)

	file.write(footer)
	file.close()

if __name__ == "__main__":
	do = ['trigram', 'skype', 'email', 'btc', 'ip', 'link']
	gen_latex_coincidences(do,lst_users=["797785[26]","907277[8]","881531[26]","488120[0]","415390[26]","847489[26]","628821[26]","630183[0]","1098178[8]","847925[26]","321034[26]","547363[26]"]) # To create a latex file with the coincidences
	#gen_coincidences(do) # To create the folders