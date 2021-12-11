from tqdm import tqdm
import pandas as pd
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from tabulate import tabulate
import numpy as np


def date_parser(date):
    '''
    :param date: date as a timestamp
    :return: date as "YEAR-month-day" format
    '''
    return datetime.fromtimestamp(date).strftime('%Y-%m-%d')


def drop_rows_int(df):
    '''
    :param df: dataframe to analyze
    :return: index of the dataframe where the date column is lower than start
    '''
    start = '2014-09-01'
    index = []
    for i, row in tqdm(df.iterrows()):
        if start > date_parser(row[2]):
            index.append(i)

    return index


def create_graph(G, df, type, self_loops=True):
    '''
    create/modify a graph from a dataframe
    :param G: networkx graph to be created/modified
    :param df: dataframe to add to the graph
    :param type: one of a2q, c2a or c2q
    :param self_loops: if it should contains self_loops or not
    '''
    for _, row in tqdm(df.iterrows()):
        if (row[0] != row[1]) or self_loops:
            if G.has_edge(row[0], row[1]):
                if G.has_edge(row[0], row[1], key=type):
                    G[row[0]][row[1]][type]['time_list'].append(date_parser(row[2]))
                    G[row[0]][row[1]][type]['weight'] += 1
                else:
                    G.add_edge(row[0], row[1], key=type, weight=1, time_list=[date_parser(row[2])])
            else:
                G.add_edge(row[0], row[1], key=type, weight=1, time_list=[date_parser(row[2])])


def create_final_graph(G, key=None):
    '''
    :param G: graph which contains as key one of a2q, c2a, c2q and
              as attributes list of dates and weights (len of list of dates)
    :param key: if there is a key, create a graph belonging to that key
    :return: graph which contains edges with list of dates and
             weights (len of list of dates)
             it is a merged initial graph G
    '''
    new_g = nx.DiGraph()
    for u, v, k, data in G.edges(data=True, keys=True):
        w = data['weight']
        time_list = data['time_list']
        if key is None or key == k:
            if new_g.has_edge(u, v):
                new_g[u][v]['weight'] += w
                new_g[u][v]['time_list'] += time_list
            else:
                new_g.add_edge(u, v, weight=w, time_list=time_list)
    return new_g


# 2 Implementation of the BackEnd
# functionality 1
def functionality_1(G, key):
    '''
    :param G: Graph with keys
    :param key: one of a2q, c2a, c2q
    :return: - the graph is directed,
             - number of users,
             - number of answers/comments,
             - average number of links per user
             - density degree of the graph,
             - whether the graph is sparse or dense
    '''
    # if there is an edge (u,v) and another (v,u),
    # then the graph is undirected
    directed = True
    set_users = set()
    total_links = 0
    density = 0
    type = None
    average_links = 0

    for edge in list(G.edges): # u->v
        u = edge[0]
        v = edge[1]
        k = edge[2]
        if (k == key):
            if directed and G.has_edge(v, u, key=key): # v->u with key=key
                directed = False
            set_users.add(u)
            set_users.add(v)
            total_links += 1

    num_users = len(set_users)

    if num_users > 0:
        average_links = total_links/num_users
        density = average_links*(1/(num_users-1))

    if abs(total_links-num_users) < abs(total_links-(num_users**2)):
        type = 'Sparse'
    else:
        type = 'Dense'

    return directed, num_users, total_links, average_links, density, type


def interval_time(G, time_inter):
    '''
    :param G: initial graph, with u, v and weights and time_list as attributes
    :param time_inter: [start_date, end_date]
    :return: new_G: graph with only u, v and weight, which is the length of all
                    the dates inside the time interval
    '''
    start = time_inter[0]
    end = time_inter[1]
    new_G = nx.DiGraph()
    for u, v, data in tqdm(G.edges(data=True)):
        time_list = data['time_list']
        # leave only the dates inside the time interval
        new_time_list = [time for time in time_list if start <= time <= end]
        # new weight, which is the length of the new_time_list
        w = len(new_time_list)
        # if the weight differ from 0, then the edge exists
        if w != 0:
            new_G.add_edge(u, v, weight=w)
    return new_G

# functionality 2
# functionality 3
# functionality 4

# 3 Implementation of the FrontEnd
# visualization 1
def visualization_1(G, key):
    '''
    :param G: Graph with keys
    :param key: one of a2q, c2a, c2q
    :return: - table of functionality_1,
             - plot of the density distribution of the input graph
    '''
    directed, num_users, total_links, average_links, density, type = functionality_1(G, key)
    table = [['Directed', 'Number of users', 'Number of answers/comments',
              'Average number of links per user', 'Density degree of the graph', 'Sparse or dense?'],
             [directed, num_users, total_links, average_links, density, type]]
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

    new_G = create_final_graph(G, key=key)
    nodes_g = list(new_G.nodes)
    df_g = nx.to_pandas_edgelist(new_G, nodelist=nodes_g)

    in_degree = df_g.target.to_list()
    dict_in = dict.fromkeys(nodes_g, 0)
    for node in in_degree:
        dict_in[node] += 1
    in_deg_freq = Counter(sorted(list(dict_in.values())))
    y_in1 = np.array(list(in_deg_freq.values()))
    y_in = y_in1 / sum(y_in1)
    x_in = list(in_deg_freq.keys())

    out_degree = df_g.source.to_list()
    dict_out = dict.fromkeys(nodes_g, 0)
    for node in out_degree:
        dict_out[node] += 1
    out_deg_freq = Counter(sorted(list(dict_out.values())))
    y_out1 = np.array(list(out_deg_freq.values()))
    y_out = y_out1 / sum(y_out1)
    x_out = list(out_deg_freq.keys())

    plt.figure(figsize=(13, 6))
    plt.bar(x_in, y_in)
    plt.title("In Degree distribution")
    plt.xlabel("in-degree")
    plt.ylabel("Density distribution in-degree")
    plt.xlim(-1, 40)
    plt.ylim(0, 0.6)
    plt.show()
    plt.close()

    plt.figure(figsize=(13, 6))
    plt.bar(x_out, y_out)
    plt.title("Out Degree distribution")
    plt.xlabel("out-degree")
    plt.ylabel("Density distribution out-degree")
    plt.xlim(-1, 40)
    plt.ylim(0, 0.6)
    plt.show()
    plt.close()
