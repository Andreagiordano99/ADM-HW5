from tqdm import tqdm
import pandas as pd
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from tabulate import tabulate
import numpy as np
import math


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


def dijkstra(df, source, target, G):
    '''
    :param df: dataframe of the graph: source, target, weight
    :param source: source node
    :param target: target node
    :return: final_weight, path
    '''
    # init the distance dictionary with the source node as key and as value None (parent node) and 0 (distance)
    distance_par = {source: (None, 0)}
    visited = set()
    # store the current node
    curr_node = source

    while curr_node != target:
        visited.add(curr_node)
        dest_list = df[df['source']==curr_node].target.to_list()
        # taking only the weight of curr_node
        curr_weight = distance_par[curr_node][1]
        for node in dest_list:
            weight = G[curr_node][node]['weight'] + curr_weight
            if node not in distance_par:
                distance_par[node] = (curr_node, weight)
            else:
                node_weight = distance_par[node][1]
                if node_weight > weight:
                    distance_par[node] = (curr_node, weight)

        # create a list of nodes to visit
        next_dest_list = {n: distance_par[n] for n in distance_par if n not in visited}

        # check if there are nodes to visit
        if not next_dest_list:
            return 'No possible path'

        # next curr node is the one in next_dest_list (nodes still to visit) with the lowest weight
        curr_node = min(next_dest_list, key=lambda k: next_dest_list[k][1])

    # now we wont to compute the list of the shortest path from source to target
    short_path = []
    # at the beginning, curr_node is the target node, since the first while is finished
    final_weight = distance_par[curr_node][1]
    while curr_node is not None:
        short_path.append(curr_node)
        par_node = distance_par[curr_node][0]
        curr_node = par_node

    # reverse the list now
    short_path = short_path[::-1]

    return [final_weight, short_path]

# functionality 2


# functionality 3


# functionality 4
def functionality_4(G, time1, time2, u1, u2):
    '''
    :param G: directed graph
    :param time1: first interval
    :param time2: second interval
    :param u1: unique user of time1
    :param u2: unique user of time2
    :return: minimum links to cut, weight and set of edges cutted
    '''
    # creating G1 and G2
    G1 = interval_time(G, time1)
    G2 = interval_time(G, time2)

    for u, v, weight in G1.edges(data='weight'):
        G1[u][v]['weight'] = round(1/weight, 2)
    for u, v, weight in G2.edges(data='weight'):
        G2[u][v]['weight'] = round(1/weight, 2)

    G12 = nx.DiGraph()
    list_g1 = list(G1.edges(data='weight'))
    list_g2 = list(G2.edges(data='weight'))
    for edge1 in list_g1:
        G12.add_edge(edge1[0], edge1[1], weight=edge1[2])
    for edge2 in list_g2:
        if G12.has_edge(edge2[0], edge2[1]):
            G12[edge2[0]][edge2[1]]['weight'] += edge2[2]
        else:
            G12.add_edge(edge2[0], edge2[1], weight=edge2[2])

    list_g12 = list(G12.edges(data='weight'))

    for edge12 in list_g12:
        if G1.has_edge(edge12[0], edge12[1]):
            G1[edge12[0]][edge12[1]]['weight'] = edge12[2]
        if G2.has_edge(edge12[0], edge12[1]):
            G2[edge12[0]][edge12[1]]['weight'] = edge12[2]

    list_n12 = list(G12.nodes)
    df_g12 = nx.to_pandas_edgelist(G12, nodelist=list_n12)
    path = dijkstra(df_g12, u1, u2, G12)

    if path == 'No possible path':
        return 'Nodes are not connected'
    else:
        path = path[1]
    weight = 0
    num_links = 0
    edge = (path[0], path[1])
    all_links = set()

    while path:
        min_weight = math.inf
        for i in range(len(path)-1):
            if G12[path[i]][path[i+1]]['weight'] < min_weight:
                min_weight = G12[path[i]][path[i+1]]['weight']
                edge = (path[i], path[i+1])
        weight += min_weight
        num_links += 1
        all_links.add(edge)
        G12.remove_edge(edge[0], edge[1])
        index_df = df_g12[(df_g12.source == edge[0]) & (df_g12.target == edge[1])].index
        # Delete these row indexes from dataFrame
        df_g12.drop(index_df, inplace=True)
        path = dijkstra(df_g12, u1, u2, G12)
        if path == 'No possible path':
            break
        else:
            path = path[1]

    return ['Minimum links to cut: ' + str(num_links) + '\nTotal weight: ' + str(weight), all_links]




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

    # in degree node: the edge come into the node
    in_degree = df_g.target.to_list()
    dict_in = dict.fromkeys(nodes_g, 0)
    # count the in-degree frequency for every node
    for node in in_degree:
        dict_in[node] += 1
    # sort ascending the in-degree frequency
    in_deg_freq = Counter(sorted(list(dict_in.values())))
    y_in1 = np.array(list(in_deg_freq.values()))
    y_in = y_in1 / sum(y_in1)
    x_in = list(in_deg_freq.keys())

    # out degree node: the edge come out of the node
    out_degree = df_g.source.to_list()
    dict_out = dict.fromkeys(nodes_g, 0)
    # count the out-degree frequency for every node
    for node in out_degree:
        dict_out[node] += 1
    # sort ascending the out-degree frequency
    out_deg_freq = Counter(sorted(list(dict_out.values())))
    y_out1 = np.array(list(out_deg_freq.values()))
    y_out = y_out1 / sum(y_out1)
    x_out = list(out_deg_freq.keys())

    plt.figure(figsize=(13, 6))
    plt.bar(x_in, y_in)
    plt.title("In Degree distribution frequency")
    plt.xlabel("in-degree")
    plt.ylabel("Density distribution in-degree")
    plt.xlim(-1, 40)
    plt.ylim(0, 0.6)
    plt.show()
    plt.close()

    plt.figure(figsize=(13, 6))
    plt.bar(x_out, y_out)
    plt.title("Out Degree distribution frequency")
    plt.xlabel("out-degree")
    plt.ylabel("Density distribution out-degree")
    plt.xlim(-1, 40)
    plt.ylim(0, 0.6)
    plt.show()
    plt.close()


# visualization 2


# visualization 3


# visualization 4
def visualization_4(G, time1, time2, u1, u2):
    list_edges = functionality_4(G, time1, time2, u1, u2)[1]
    G_edge = nx.Graph()
    G_edge.add_edges_from(list_edges)
    # nx.draw(G_edge, with_labels=True)

    left_nodes = []
    right_nodes = []
    for edge in list_edges:
        left_nodes.append(edge[0])
        right_nodes.append(edge[1])


    # set the position according to column (x-coord)
    pos = {n: (0, i) for i, n in enumerate(left_nodes)}
    pos.update({n: (1, i + 0.5) for i, n in enumerate(right_nodes)})

    options = {
        "font_size": 10,
        "node_size": 3000,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 5,
        "width": 5,
    }
    nx.draw_networkx(G_edge, pos, **options)

    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show()
