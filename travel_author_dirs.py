import os

def preprocessing(text):
	#Esto procesa cada uno de los posts de cada usuario.
	print(text)
	#Poner codigo aqui.
	return text

def process_post(post_file):
	text = None
	with open(post_file, 'r') as f:
		text = f.read()
	print(post_file)
	text = preprocessing(text)
	with open(post_file, 'w+') as f:
		f.write(text)


def process_dir(directory):
	list_posts = os.listdir(directory)
	print(len(list_posts))
	for post_file in list_posts:
		process_post(directory+post_file)
		return
def process_dirs():
	directory = "Author/"
	list_dirs= os.listdir(directory)
	for directory2 in list_dirs:
		process_dir(directory+directory2+"/")
		return

def main():
	process_dirs()
	
if __name__ == "__main__":
	main()