import numpy as np
import os
import pickle


class OpTree:
    def __init__(self, train_filename, cache_filename, txt_emb_filename):
        self.filename = train_filename
        self.cache_filename = cache_filename
        self.txt_emb_filename = txt_emb_filename

    def read_edges_from_file(self):
        with open(self.filename, "r") as f:
            lines = f.readlines()
            edges = [self.str_list_to_int(line.split()) for line in lines]
        return edges

    def str_list_to_float(self, str_list):
        return [float(item) for item in str_list]

    def sigmoid(self, x):
        s = 1 / (1 + np.exp(-x))
        return s

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))  # for computation stability
        return e_x / e_x.sum()

    def str_list_to_int(self, str_list):
        return [int(item) for item in str_list]

    def read_edges(self):
        graph = {}
        nodes = set()
        train_edges = self.read_edges_from_file()

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

    def getNodeTree(self):
        trees = None
        if os.path.isfile(self.cache_filename):
            print("reading BFS-trees from cache...")
            pickle_file = open(self.cache_filename, 'rb')
            trees = pickle.load(pickle_file)
            pickle_file.close()
            print("have getten BFS-trees")
        nodes = {}  # n_node个元素，每个元素为为节点i为root的树的节点序列
        trees_list = list(trees.keys())
        for i in trees_list:
            # print(trees[i])
            node = list(trees[i].keys())
            nodes[i] = node
        return nodes

    def getNodeText(self):
        txt2vec = {}
        with open(self.txt_emb_filename) as f:
            for line in f.readlines()[1:]:
                l = [float(i) for i in line.strip().split()]
                idx = int(l[0])
                node_txt = l[1:]
                txt2vec[idx] = node_txt
        return txt2vec

    def getTreeTheme(self):
        treeTheme = {}  # 每棵树的Theme embedding
        nodeTrees = self.getNodeTree()
        txt2vec = self.getNodeText()
        for i in nodeTrees:
            node_list = nodeTrees[i]  # [0,279,245,...]
            txt = []
            for j in node_list:
                txt.append(txt2vec[j])
            txt = np.array(txt)
            txt = np.sum(txt, 0).reshape((100, 1))
            treeTheme[i] = txt
        return treeTheme, nodeTrees

    def getNodeTheme(self):
        nodeTheme = {}
        nodes, n_node, graph = self.read_edges()
        treeTheme, nodeTrees = self.getTreeTheme()
        for i in nodes:
            theme = []
            for j in nodeTrees:
                if i in nodeTrees[j]:
                    theme.append(treeTheme[j])
            # 先用softmax实验
            theme = np.tanh(np.sum(np.squeeze(np.array(theme), axis=(2,)), 0).reshape((1, 100)))
            nodeTheme[i] = theme.tolist()[0]
        return nodeTheme
