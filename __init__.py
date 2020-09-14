from dataset_generators import create_directories_and_datasets, create_dir, create_directories_and_datasets_1, create_directories_and_datasets_2, generate_directories_for_users
from extract_class import FeatureScore, MultFSScore
from multfs import MultFS, MultFSJoin
#from adhoc_removal import *
import adhoc_removal_v2
from common_utils import gen_csv_from_tuples, read_csv_list, make_query
import sys, time, os
import multiprocessing as mp
import shutil
import zipfile

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def generate_args(do):
	args = {}
	for identifier in do:
		args[identifier] = {}
		args[identifier]['user_removal'] = None
		args[identifier]['value_removal'] = None

	args['ip']['value_removal'] = [adhoc_removal_v2.ip_value_removal]
	args['link']['value_removal'] = [adhoc_removal_v2.link_value_removal_keep_params, adhoc_removal_v2.link_value_removal_2]
	args['trigram']['value_removal'] = [adhoc_removal_v2.trigram_value_removal, adhoc_removal_v2.trigram_non_european_chars_removal]



	args['skype']['user_removal'] = [adhoc_removal_v2.skype_user_removal]
	args['email']['user_removal'] = [adhoc_removal_v2.email_user_removal]
	args['link']['user_removal'] = [adhoc_removal_v2.link_user_removal]
	args['ip']['user_removal'] = [adhoc_removal_v2.ip_user_removal]
	args['btc']['user_removal'] = []
	args['trigram']['user_removal'] = [adhoc_removal_v2.general_user_removal]

	args['skype']['rarity_bound'] = 4
	args['email']['rarity_bound'] = 0
	args['link']['rarity_bound'] = 15
	args['ip']['rarity_bound'] = 23
	args['btc']['rarity_bound'] = 15
	args['trigram']['rarity_bound'] = 0
	return args

def extract_features(args, do):		
	lst_ms = []
	for identifier, dictio in args.items():
		if identifier in do:
			ms = MultFSScore(identifier, user_removal=dictio['user_removal'], value_removal=dictio['value_removal'])
			lst_ms.append(ms)

	for ms in lst_ms:
		ms.compute()

	return

def get_dataset_sizes(args, do):		
	sizes = []
	for identifier, dictio in args.items():
		if identifier in do:
			ms = MultFSScore(identifier, user_removal=dictio['user_removal'], value_removal=dictio['value_removal'])
			sizes.append((identifier,) + ms.get_size_clean())

	return sizes

def join_all(args, do):
	msj = MultFSJoin()
	for identifier, dictio in args.items():
		if identifier in do:
			msj.add_identifier(identifier)

	msj.compute()
	return

def copy_file(src, dest):
	print("Copying  %s to %s" % (src, dest), end='\r')
	shutil.copy(src, dest)
	print("[END] Copying  %s to %s" % (src, dest))

def zip_dir(directory):
	print("Zipping directory: %s" %(directory), end='\r')
	zipf = zipfile.ZipFile('multfs_results.zip', 'w', zipfile.ZIP_DEFLATED)
	for root, dirs, files in os.walk(directory):
		for file in files:
			zipf.write(os.path.join(root, file))
	zipf.close()
	print("[END] Zipping directory: %s" %(directory))

def main():
	init_string = """Usage: python3 __init__.py <option>
	<option>:
		'multfs': do all processing (comprises 'compute' of all features and 'join'.)
		'compute': do processing of one specific feature
		'join': join the features and compute multfs only.
		'export': generate a folder for exporting data from database"""
	if len(sys.argv) < 2:
		print(init_string)
		return

	#args = generate_args_dict()	
	if sys.argv[1] == 'multfs':
		do = ['ip', 'link','trigram', 'skype', 'email', 'btc']
		
		
		args = generate_args(do)
		extract_features(args, do)
		join_all(args, do)
		
	elif sys.argv[1] == 'compute':
		if len(sys.argv) < 3:
			print('Usage: python __init__.py compute <feature>')
			exit(-2)

		do = ['ip', 'link','trigram', 'skype', 'email', 'btc']
		args = generate_args(do)
		feature = str(sys.argv[2])
		
		if not feature in do:
			print('Feature %s not supported' % (feature))
			exit(-2)
		ms = MultFSScore(feature, user_removal=args[feature]['user_removal'], value_removal=args[feature]['value_removal'])
		ms.compute()
		
	elif sys.argv[1] == 'sizes':
		do = ['ip', 'link','trigram', 'skype', 'email', 'btc']
		
		
		args = generate_args(do)
		sizes  = get_dataset_sizes(args, do)
		print("{" + ", ".join(["'%s': (%d, %d)" % (s[0], s[1], s[2]) for s in sizes]) + "}")


	elif sys.argv[1] == 'join':
		do = ['ip', 'link','trigram', 'skype', 'email', 'btc']
		args = generate_args(do)
		join_all(args, do)	
		
	elif sys.argv[1] == 'export':
		do = ['ip', 'link','trigram', 'skype', 'email', 'btc']
		basedir = "multfs_results/"
		create_dir(basedir)
		copy_file('common_utils.py', basedir + 'common_utils.py')
		copy_file('db_credentials_example.py', basedir + 'db_credentials.py' )
		copy_file('mail_credentials_example.py', basedir + 'mail_credentials.py' )
		copy_file('multfs.csv', basedir + 'multfs.csv')
		copy_file('user_post_coincidences.py', basedir + 'user_post_coincidences.py')
		files_to_copy = ['clean_user_ind.pkl', 'clean_value_ind.pkl', 'clean_dictio_of_users.pkl']
		for feature in do:
			srcdir = feature + "_files/" 
			dstdir = basedir + feature + "_files/"
			create_dir(dstdir)
			for file in files_to_copy:
				copy_file(srcdir + file, dstdir + file)


		zip_dir(basedir)
		print("Removing temporal directory", end='\r')
		shutil.rmtree(basedir)
		print("[END] Removing temporal directory")

	else:
		print(init_string)

if __name__ == "__main__":
	main()
