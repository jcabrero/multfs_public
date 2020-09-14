import numpy as np
from common_utils import read_csv_list

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
print("Loading file data.")
lst = read_csv_list('num_files/user_to_num.csv')[1:]
print("Loaded file data.")
x = np.array([int(x[1]) for x in lst])
y = [np.array([int(y) for y in x[2:]]) for x in lst]

print("Created axis")
z = np.array([i.mean() for i in y if len(i) > 0])

xg = [np.quantile(x, i) for i in np.arange(0,1.01, 0.01)]
zg = [np.quantile(z, i) for i in np.arange(0,1.01, 0.01)]

print("XG", xg)
print("ZG", zg)


sns.set(style="whitegrid")
#ax = sns.boxplot(x=x)
ax = sns.distplot(x);
#plt.savefig('your_figure.png')
ax.figure.savefig("image.png")