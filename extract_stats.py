import numpy as np
import random
import string
from collections import Counter
import time

from dataset_generators import create_dir
from common_utils import read_csv_list, gen_csv_from_tuples
from extract_class import MultFSScore


import sys, traceback

class Suppressor(object):

    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, type, value, traceback):
        sys.stdout = self.stdout
        if type is not None:
            print("ERROR")
            # Do normal exception handling

    def write(self, x): pass

def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def random_combination(users, values):
    distribution = np.array(list(range(len(values))))
    # We input a gaussian distribution with most likely at 0 and diminishing to higher numbers
    num_elements = np.rint(np.abs(np.random.normal(0, distribution.std(), len(users)))).astype(np.int32)
    # Contains the number of values that each user is going to be combined with.
    #num_elements = np.random.randint(0, np.round(0.20 * len(values)), len(users))    
    lst = []
    for i, user in enumerate(users):
        # Now we generate num_elements[user] random elements, which are the indexes which are considered.
        value_indexes = np.random.randint(0, len(values), num_elements[i])
        user_values = values[value_indexes]

        lst.append(user_values)
    
    return lst

def create_dummy_dataset(num_users, num_values, user_len = 11, value_len = 11):

    uinds = np.array(list(range(num_users)))
    vinds = np.array(list(range(num_values)))
    combinations = random_combination(uinds, vinds)
    
    users = get_random_string(num_users * user_len)
    values = get_random_string(num_values * value_len)
    split = lambda long_string, n: [long_string[index : index + n] for index in range(0, len(long_string), n)]

    users = split(users, user_len)
    values = split(values, value_len)
    switched = [Counter(row) for row in combinations]
    dataset = [("usr-" + users[i] + "[15]", ) + tuple(["val-" + values[k] + "[%d]"%(v) for k, v in switched[i].items()]) for i in range(num_users)]
    return dataset
    
def create_dummy_file_dataset(identifier, num_users, num_values, user_len = 5, value_len = 6):

    dirname = identifier + "_files/"
    filename = "user_to_" + identifier + ".csv"
    create_dir(dirname)

    dataset = create_dummy_dataset(num_users, num_values, user_len, value_len)

    gen_csv_from_tuples(dirname + filename, ['User', 'Values'], dataset)

def separate(x):
	pos = len(x) - x[::-1].find('[')
	y = x[:pos - 1]
	z = x[pos: - 1]
	return y, z

def basic_score(values1, values2):
    i = 0
    values1 = [separate(v) for v in values1]
    values2 = [separate(v) for v in values2]
    for value1, amount1 in values1:
        for value2, amount2 in values2:
            if value1 == value2:
                i += (int(amount1) * int(amount2))
    
    return i

def basic_solve_find_solutions(filepath):
    data = read_csv_list(filepath)
    lst = []
    for i, row1 in enumerate(data):
        user1, values1 = row1[0], row1[1:]
        for j, row2 in enumerate(data[i + 1:]):
            user2, values2 = row2[0], row2[1:]
            score = basic_score(values1, values2)
            lst.append((user1, user2, score))
    return lst

def test_basic(identifier):
    dirname = identifier + "_files/"
    filename = "user_to_" + identifier + ".csv"
    filepath = dirname + filename
    tic = time.time()

    lst = basic_solve_find_solutions(filepath)
    gen_csv_from_tuples(dirname + 'basic_result.csv', ['User1', 'User2', 'Score'], lst)
    toc = time.time()
    return toc - tic 
def test_vector(identifier):
    ms = MultFSScore(identifier)
    tic = time.time()
    
    ms.compute()
    toc = time.time()
    return toc - tic

def test(num_users, num_values):
    identifier = "test_%d_%d" % (num_users, num_values)
    create_dummy_file_dataset(identifier, num_users, num_values)
    tv = test_vector(identifier)
    print("vector", tv)
    tb = test_basic(identifier)
    print("basic", tb)
    return tb, tv


def test_only_vector(num_users, num_values):
    identifier = "test_%d_%d" % (num_users, num_values)
    
    i = True
    while (i):
        try:
            create_dummy_file_dataset(identifier, num_users, num_values)
            tv = test_vector(identifier)
            i = False
        except Exception as e:
            print("ERROR IN %d - %d" % (num_users, num_values))
    print("vector", tv)
    return tv

def launch_vector_stress_test():
    u  = np.rint(np.power(2, np.array(list(range(4, 17))))).astype(np.int32)
    #u  = np.rint(np.power(2, np.array(list(range(4, 12))))).astype(np.int32)
    v = np.rint(np.power(2, np.array(list(range(4, 17))))).astype(np.int32)
    #v = np.rint(np.power(2, np.array(list(range(4, 12))))).astype(np.int32)
    lst = []
    combinations = []
    for _u in u:
        for _v in v:
            combinations.append((_u, _v))
    
    combinations = sorted(combinations, key=lambda x: x[0] + x[1])

    for _u, _v in combinations:
        print("CURRENTLY TESTING:", _u, _v)
        tv = test_only_vector(_u, _v)
        lst.append((_u, _v, tv))
        gen_csv_from_tuples('performance_test_results_vector.csv', ['Users', 'Values', 'Vector'], lst)


def launch_tests():
    u  = np.rint(np.power(2, np.array(list(range(11, 12))))).astype(np.int32)
    #u  = np.rint(np.power(2, np.array(list(range(4, 12))))).astype(np.int32)
    v = np.rint(np.power(2, np.array(list(range(10, 12))))).astype(np.int32)
    #v = np.rint(np.power(2, np.array(list(range(4, 12))))).astype(np.int32)
    lst = []
    for _u in u:
        for _v in v:
            print("CURRENTLY TESTING:", _u, _v)
            tb, tv = test(_u, _v)
            lst.append((_u, _v, tb, tv))
            gen_csv_from_tuples('performance_test_results.csv', ['Users', 'Values', 'Basic', 'Vector'], lst)




#launch_tests()
launch_vector_stress_test()   
