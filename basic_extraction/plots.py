import matplotlib
matplotlib.use('Agg')
import seaborn as sns

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from common_utils import gen_csv_from_tuples, read_csv_list, make_query
import holoviews 
import holoviews as hv
from holoviews import opts, dim
from scipy import stats

def plot_scoring_data(scoring_data):
	data = [x[-1] for x in scoring_data] 
	sns.kdeplot(data, shade=True)


def plot_scoring_data2():
	lst = read_csv_list("results_ip_comp.csv")[1:]
	data = [float(x[-1]) for x in lst] 
	print(data[:10])
	#sns_plot = sns.kdeplot(list(range(20)), shade=True)
	sns_plot = sns.distplot(data, kde=False, rug=True);
	fig = sns_plot.get_figure()
	fig.savefig("output.png")
	#sns.plt.show()

def plot_network_chartx():
	i = 0
	lst = read_csv_list("results_ip_comp.csv")[1:1000]
	lst = [x for x in lst if float(x[-1]) < float(10000)]
	print(len(lst))
	df = pd.DataFrame({ 'from': [x[0] for x in lst], 'to': [x[1] for x in lst]})
	# Build your graph
	#G=nx.from_pandas_dataframe(df, 'from', 'to')
	G = nx.from_pandas_edgelist(df, 'from', 'to')
	plt.figure(figsize=(50,50))
	node_color = [100 * G.degree(node) for node in G]
	node_size =  [1000 * G.degree(node) for node in G]
	#pos = nx.spring_layout(G, k=0.04)

	graph = nx.draw_spring(G, k=0.14, with_labels=True, node_size=node_size, 
		node_color=node_color, node_shape="o", 
		alpha=0.5, linewidths=4, font_size=25, 
		font_color="black", font_weight="bold", 
		width=2, edge_color="grey")
	plt.savefig("graphs/Graph_IP.png", format="PNG")

def plot_network_charty():
	i = 0
	lst = read_csv_list("results_skype_comp.csv")[1:1000]
	lst = [x for x in lst if float(x[-1]) < float(10000)]
	print(len(lst))
	df = pd.DataFrame({ 'from': [x[0] for x in lst], 'to': [x[1] for x in lst]})
	# Build your graph
	#G=nx.from_pandas_dataframe(df, 'from', 'to')
	G = nx.from_pandas_edgelist(df, 'from', 'to')
	plt.figure(figsize=(50,50))
	node_color = [100 * G.degree(node) for node in G]
	node_size =  [1000 * G.degree(node) for node in G]
	#pos = nx.spring_layout(G, k=0.04)

	graph = nx.draw_spring(G, k=0.14, with_labels=True, node_size=node_size, 
		node_color=node_color, node_shape="o", 
		alpha=0.5, linewidths=4, font_size=25, 
		font_color="black", font_weight="bold", 
		width=2, edge_color="grey")
	plt.savefig("graphs/Graph_Skype.png", format="PNG")


def plot_network_chartz():
	i = 0
	lst = read_csv_list("results_email_comp.csv")[1:1000]
	lst = [x for x in lst if float(x[-1]) < float(10000)]
	print(len(lst))
	df = pd.DataFrame({ 'from': [x[0] for x in lst], 'to': [x[1] for x in lst]})
	# Build your graph
	#G=nx.from_pandas_dataframe(df, 'from', 'to')
	G = nx.from_pandas_edgelist(df, 'from', 'to')
	plt.figure(figsize=(50,50))
	node_color = [100 * G.degree(node) for node in G]
	node_size =  [1000 * G.degree(node) for node in G]
	#pos = nx.spring_layout(G, k=0.04)

	graph = nx.draw_spring(G, k=0.14, with_labels=True, node_size=node_size, 
		node_color=node_color, node_shape="o", 
		alpha=0.5, linewidths=4, font_size=25, 
		font_color="black", font_weight="bold", 
		width=2, edge_color="grey")
	plt.savefig("graphs/Graph_Email.png", format="PNG")
def plot_network_chart3():
	i = 0
	lst = read_csv_list("results_ip_comp.csv")[1:1000]
	lst = [x for x in lst if float(x[-1]) < float(10000)]
	print(len(lst))
	df = pd.DataFrame({ 'from': [x[0] for x in lst], 'to': [x[1] for x in lst]})
	# Build your graph
	#G=nx.from_pandas_dataframe(df, 'from', 'to')
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
	plt.savefig("Graph_spring.png", format="PNG")
	graph = nx.draw_spectral(G, with_labels=True, node_size=node_size, 
		node_color=node_color, node_shape="o", 
		alpha=0.5, linewidths=4, font_size=25, 
		font_color="black", font_weight="bold", 
		width=2, edge_color="grey")
	plt.savefig("Graph_spectral.png", format="PNG")
	#graph = nx.draw_planar(G, with_labels=True, node_size=node_size, 
		#node_color=node_color, node_shape="o", 
		#alpha=0.5, linewidths=4, font_size=25, 
		#font_color="black", font_weight="bold", 
		#width=2, edge_color="grey")
	#plt.savefig("Graph_planar.png", format="PNG")
	graph = nx.draw_shell(G, with_labels=True, node_size=node_size, 
		node_color=node_color, node_shape="o", 
		alpha=0.5, linewidths=4, font_size=25, 
		font_color="black", font_weight="bold", 
		width=2, edge_color="grey")
	plt.savefig("Graph_shell.png", format="PNG")

def plot_connection_graph():
	lst = read_csv_list("results_ip_comp.csv")[1:1000]
	lst = [x for x in lst if float(x[-1]) < float(10000)]
	links = pd.DataFrame({ 'source': [x[0] for x in lst], 'target': [x[1] for x in lst]})
	chord = hv.Chord(links).select(value=(5, None))
	chord.opts(
		opts.Chord(cmap='Category20', edge_cmap='Category20', 
			edge_color=dim('source').str(), labels='name',
			node_color=dim('index').str()))
def plot_histogram():
	df = sns.load_dataset('iris')
	# Make default histogram of sepal length
	sns.distplot( df["sepal_length"] )
	print(df["sepal_length"])
	plt.savefig("graph.png", format="PNG")
	#sns.plt.show()

def main():
	plot_histogram()
	#plot_connection_graph()
	#see_different()
	#plot_network_chartx()
	#plot_network_charty()
	#plot_network_chartz()
	#We extract a user to ip csv, with variable length which represents ips and repetitions of the forme
	#plot_scoring_data2()
	

if __name__ == "__main__":
	
	main()