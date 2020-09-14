import sys

def calculate_size(num_users, num_ips):
	print("USERS X ELSE", num_users * num_ips * 1 / (1024 ** 3), "GB")
	print("USERS X USERS", num_users * num_users * 4 / (1024 ** 3), "GB")
def main():
	if len(sys.argv) < 2:
		print("ERROR IN ARGUMENTS (num_users) (num_ips)")
	else:
		calculate_size(int(sys.argv[1]), int(sys.argv[2]))
if __name__ == "__main__":
	main()



# num_users = 73195
# num_ips = 1127768
# print(num_users * num_ips * 4 / (1024 ** 3), "GB")
# print(num_users * num_users * 4 / (1024 ** 3), "GB")
# num_users = 19928
# num_ips = 1103497
# print(num_users * num_ips * 1 / (1024 ** 3), "GB")
# print(num_users * num_users * 4 / (1024 ** 3), "GB")
# num_users = 2354
# num_ips = 1048493
# print(num_users * num_ips * 4 / (1024 ** 3), "GB")
# print(num_users * num_users * 4 / (1024 ** 3), "GB")
# num_users = 7132
# num_ips = 1079278
# print(num_users * num_ips * 4 / (1024 ** 3), "GB")
# print(num_users * num_users * 4 / (1024 ** 3), "GB")
# num_users = 4515
# num_ips = 699889
# print(num_users * num_ips * 4 / (1024 ** 3), "GB")
# print(num_users * num_users * 4 / (1024 ** 3), "GB")
# num_users = 74931
# num_ips = 1610185
# print(num_users * num_ips * 1 / (1024 ** 3), "GB")
# print(num_users * num_users * 4 / (1024 ** 3), "GB")
# num_users = 54000
# num_ips = 37
# print(num_users * num_ips * 1 / (1024 ** 3), "GB")
# print(num_users * num_users * 4 / (1024 ** 3), "GB")
