import time
import sys

last_percentage = 0
toolbar_width = 40
numbar_width = 40
def create_status_bar(length = 40):
	global toolbar_width
	toolbar_width = length

	# setup toolbar
	sys.stdout.write("[%s]" % (" " * toolbar_width))
	sys.stdout.flush()
	sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

def update_status_bar(point, total_length):
	global last_percentage
	percentage = int(point * toolbar_width / total_length)
	for i in range (last_percentage, percentage):
		sys.stdout.write("-")
		sys.stdout.flush()
	last_percentage = percentage

def end_status_bar():
	global last_percentage
	last_percentage = 0
	sys.stdout.write("\n")

def create_numbar(length = 40, total_length=40):
	global numbar_width
	numbar_width = length
	numbar = "[%s] 0/%d" % (" " * numbar_width, total_length)
	# setup toolbar
	sys.stdout.write(numbar)
	sys.stdout.flush()
	sys.stdout.write("\b" * (len(numbar))) # return to start of line, after '['

def update_numbar(point, total_length):
	percentage = int(point * numbar_width / total_length)
	numbar = "[%s] %d/%d" % (("-" * percentage) + (" " * (numbar_width - percentage)), point, total_length)
	sys.stdout.write(numbar)
	sys.stdout.flush()
	sys.stdout.write("\b" * (len(numbar))) # return to start of line, after '['

def end_numbar():

	sys.stdout.write("\n")