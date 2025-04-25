# from __future__ import print_function
import time
# from gensim.models import Word2Vec
from . import walker
import numpy as np


class Node2vec(object):

    def __init__(self, graph, path_length, num_paths, p=1.0, q=1.0, dw=False, **kwargs):
        # workers貌似没用上
        kwargs["workers"] = kwargs.get("workers", 1) # 如果没有这个key，就会返回1。
        if dw:
            kwargs["hs"] = 1
            p = 1.0
            q = 1.0

        self.graph = graph
        if dw:  # deepwalk
            self.walker = walker.BasicWalker(graph, workers=kwargs["workers"])
        else:  # node2vec
            self.walker = walker.Walker(
                graph, p=p, q=q, workers=kwargs["workers"])
            print("Preprocess transition probs...")
            self.walker.preprocess_transition_probs()

        self.walks = self.walker.simulate_walks(
            num_walks=num_paths, walk_length=path_length)


    def get_walks(self):

        return self.walks
