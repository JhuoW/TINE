import tqdm
import multiprocessing
import collections
import numpy as np
import os
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dataset = "blogCatalog"
train_filename = '../../data/' + dataset + '/bc_edgelist.txt'
cache_filename = "../cache/" + dataset + ".pkl"
tfidf_file = '../../data/' + dataset + '/' + 'bc_tfidf.txt'
new_filename = '../../data/blogCatalog/bc_new_embedding.txt'
embed_file = '../../data/blogCatalog/bc_embedding.txt'
n_embed = 100


def str_list_to_float(str_list):
    return [float(item) for item in str_list]


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def str_list_to_int(str_list):
    return [int(item) for item in str_list]


def read_edges_from_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        edges = [str_list_to_int(line.split()) for line in lines]
    return edges


def read_edges(filename):
    graph = {}
    nodes = set()
    train_edges = read_edges_from_file(filename)

    for edge in train_edges:
        nodes.add(edge[0])
        nodes.add(edge[1])
        if graph.get(edge[0]) is None:
            graph[edge[0]] = []
        if graph.get(edge[1]) is None:
            graph[edge[1]] = []
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])

    return nodes, len(nodes), graph


def construct_trees(nodes, graph):
    """use BFS algorithm to construct the BFS-trees

    Args:
        nodes: the list of nodes in the graph
    Returns:
        trees: dict, root_node_id -> tree, where tree is a dict: node_id -> list: [father, child_0, child_1, ...]
    """

    trees = {}
    for root in tqdm.tqdm(nodes):  # 每次循环遍历一个节点 得到以该节点为根的BFS Tree
        trees[root] = {}  # 以root为根的树
        trees[root][root] = [root]
        used_nodes = set()
        # deque为两端都可以操作的序列
        queue = collections.deque([root])
        while len(queue) > 0:
            cur_node = queue.popleft()
            used_nodes.add(cur_node)
            for sub_node in graph[cur_node]:  # 遍历当前节点的neighboor
                if sub_node not in used_nodes:
                    trees[root][cur_node].append(sub_node)
                    trees[root][sub_node] = [cur_node]
                    queue.append(sub_node)
                    used_nodes.add(sub_node)
    return trees


def read_embeddings(filename, n_node, n_embed):
    """read pretrained node embeddings
    """
    with open(filename, "r") as f:
        lines = f.readlines()[1:]  # skip the first line
        embedding_matrix = np.random.rand(n_node, n_embed)
        origin_matrix = np.random.rand(n_node, n_embed + 1)
        index = np.random.rand(n_node, 1)
        i = 0
        for line in lines:
            emd = line.split()
            # embedding矩阵 第emd[0]行的元素 第i行即为id为i的节点的embedding
            embedding_matrix[i, :] = str_list_to_float(emd[1:])
            origin_matrix[i, 0] = int(emd[0])
            origin_matrix[i, 1:] = str_list_to_float(emd[1:])
            index[i] = int(emd[0])
            i = i + 1
        return embedding_matrix, index, origin_matrix


def getTFIDF():
    nodes, n_node, graph = read_edges(train_filename)  # hepth 1038 {440:[50,103,...],50:[440,222,257]}
    root_nodes = [i for i in nodes]  # 将训练集中每个节点设为跟节点的序列
    trees = None
    if os.path.isfile(cache_filename):
        print("reading BFS-trees from cache...")
        pickle_file = open(cache_filename, 'rb')
        trees = pickle.load(pickle_file)
        pickle_file.close()
    else:
        print("constructing BFS-trees...")  # 若cache不存在 则构建BFS tree
        pickle_file = open(cache_filename, 'wb')
        trees = construct_trees(root_nodes, graph)
        pickle.dump(trees, pickle_file)
        pickle_file.close()  # 将BFS树读入pickle
    # print(len(trees))  # 1032棵BFS树
    # print(len(trees[0].keys()))  # 0为根节点的树有800个节点
    nodes = {}  # n_node个元素，每个元素为为节点i为root的树的节点序列

    trees_list = list(trees.keys())
    for i in trees_list:
        # print(trees[i])
        node = list(trees[i].keys())
        nodes[i] = node

    # print(nodes[853])  # test: 0,853
    #     # f.write(str(nodes))
    #     # for i in range(len(nodes)):
    #     #     print(nodes[853])
    #     #     break

    node_list = []
    for i in nodes:
        node = [' '.join(str(x) for x in nodes[i])]
        node_list.extend(node)

    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    X = vectorizer.fit_transform(node_list)
    # X.toarray()
    # print(X.shape)
    # print(vectorizer.get_feature_names())
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)
    node = vectorizer.get_feature_names()
    weight = tfidf.toarray()  # 对应的tfidf矩阵  # 1038*1038  共1038个文本（树） 1038个节点
    # np.savetxt(tfidf_file, weight)
    # for i in range(len(weight)):
    #     print("-------第", i , "棵树文本的词语tf-idf权重------")
    #     for j in range(len(node)):
    #         print(node[j], weight[i][j])
    # np.savetxt(tfidf_file, weight.sum(axis=0))
    Tweight = weight.sum(axis=0)
    node = np.array(node)
    Tnode = []
    for i in node:
        Tnode.append(int(i))
    Tnode = np.array(Tnode).reshape([n_node, 1])
    Tweight = Tweight.reshape([n_node, 1])
    A = np.hstack((Tnode, Tweight))
    # print(A.shape)
    f = open(tfidf_file, 'w+')
    for i in A:
        f.write(str(int(i[0])) + ' ' + str(i[1]) + '\n')

    tfidf = {}
    for i in A:
        j = int(i[0])
        k = i[1]
        tfidf[j] = k

    embedding_matrix, index, origin_matrix = read_embeddings(embed_file, n_node, n_embed)

    newtfidf = []
    for i in index:
        newtfidf.append(tfidf[int(i)])
    newtfidf = np.array(newtfidf).reshape(n_node, 1)
    newtfidf = np.log10(newtfidf)  # 归一化
    print(newtfidf)
    A = np.hstack((embedding_matrix, newtfidf))
    A = np.hstack((index, A))
    f = open(new_filename, 'w+')
    print(A.shape)
    embedding_list = A.tolist()
    embedding_str = [" ".join([str(x) for x in emb[0:]]) + "\n"
                     for emb in embedding_list]
    lines = [str(n_node) + ' ' + str(n_embed + 1) + '\n'] + embedding_str
    f.writelines(lines)


if __name__ == '__main__':
    getTFIDF()
