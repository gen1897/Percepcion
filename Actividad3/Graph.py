# https://sandipanweb.wordpress.com/2018/02/25/graph-based-image-segmentation-in-python/
# https://www.cis.upenn.edu/~jshi/GraphTutorial/Tutorial-ImageSegmentationGraph-cut1-Shi.pdf
import math
import OptimizedUnionFind as uf
import sys
import cv2
import random as rand
import numpy as np


class Graph():

    def __init__(self, file_path):
        self.file_output = None
        # Graphs
        self.graph = []
        self.sorted_graph = None
        # Images
        self.image = cv2.imread(file_path)
        self.np_image = np.asarray(self.image, dtype=float)
        self.filter_image = cv2.GaussianBlur(self.np_image, (5, 5), 0.5)
        # Channels
        self.b, self.g, self.r = cv2.split(self.filter_image)
        # Shape
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]
        self.num_node = self.width * self.height
        # Parameters
        self.sigma = 0.5
        self.k = 500
        self.min_size = 50
        # Optimizer
        self.upset = None

    ###### FUNCIONES DEL GRAPH ######
    def get_diff(self, x1, y1, x2, y2):
        r = (self.r[y1, x1] - self.r[y2, x2]) ** 2
        g = (self.g[y1, x1] - self.g[y2, x2]) ** 2
        b = (self.b[y1, x1] - self.b[y2, x2]) ** 2
        return math.sqrt(r + g + b)

    def get_threshold(self, size):
        return (self.k / size)

    def create_edge(self, x1, y1, x2, y2):
        def vertex_id(x, y): return y * self.width + x

        w = self.get_diff(x1, y1, x2, y2)
        return (vertex_id(x1, y1), vertex_id(x2, y2), w)

    def build_graph(self):
        for y in range(self.height):
            for x in range(self.width):
                if x < self.width - 1:
                    self.graph.append(self.create_edge(x, y, x + 1, y))
                if y < self.height - 1:
                    self.graph.append(self.create_edge(x, y, x, y + 1))
                if x < self.width - 1 and y < self.height - 1:
                    self.graph.append(self.create_edge(x, y, x + 1, y + 1))
                if x < self.width - 1 and y > 0:
                    self.graph.append(self.create_edge(x, y, x + 1, y - 1))

        self.sorted_graph = sorted(self.graph, key=lambda x: x[2])

    def remove_small_component(self):
        for edge in self.sorted_graph:
            u = self.ufset.find(edge[0])
            v = self.ufset.find(edge[1])

            if u != v:
                if self.ufset.size_of(u) < self.min_size or self.ufset.size_of(v) < self.min_size:
                    self.ufset.merge(u, v)

    def segment_graph(self):
        self.ufset = uf.OptimizedUnionFind(self.num_node)
        threshold = [self.get_threshold(1)] * self.num_node

        for edge in self.sorted_graph:
            u = self.ufset.find(edge[0])
            v = self.ufset.find(edge[1])
            w = edge[2]

            if u != v:
                if w <= threshold[u] and w <= threshold[v]:
                    self.ufset.merge(u, v)
                    parent = self.ufset.find(u)
                    threshold[parent] = w + \
                        self.get_threshold(self.ufset.size_of(parent))

    def generate_image(self):
        def random_color(): return (int(rand.random() * 255),
                                    int(rand.random() * 255), int(rand.random() * 255))
        color = [random_color() for i in range(self.width * self.height)]
        save_img = np.zeros((self.height, self.width, 3), np.uint8)

        for y in range(self.height):
            for x in range(self.width):
                color_idx = self.ufset.find(y * self.width + x)
                save_img[y, x] = color[color_idx]

        return save_img


    def run(self):
        self.build_graph()
        self.segment_graph()
        self.remove_small_component()
        self.file_output = self.generate_image()
