# from __future__ import print_function
import time
# from gensim.models import Word2Vec
from . import walker
import numpy as np

class Node2vec(object):

    def __init__(self, graph, path_length, num_paths, p=1.0, q=1.0, dw=False, device=None, use_gpu=False, **kwargs):
        kwargs["workers"] = kwargs.get("workers", 1)
        if dw:
            kwargs["hs"] = 1
            p = 1.0
            q = 1.0

        self.graph = graph
        self.path_length = path_length
        self.num_paths = num_paths
        if dw:  # deepwalk
            self.walker = walker.BasicWalker(
                graph, workers=kwargs["workers"], device=device, use_gpu=use_gpu
            )
        else:  # node2vec
            self.walker = walker.Walker(
                graph, p=p, q=q, workers=kwargs["workers"], device=device, use_gpu=use_gpu
            )
            print("Preprocess transition probs...")
            self.walker.preprocess_transition_probs()

        self.walks = self.walker.simulate_walks(
            num_walks=num_paths, walk_length=path_length)

    def get_walks(self):
        """Return walks aligned to graph node indices."""
        walks = getattr(self, "walks", None)
        if walks is None:
            walks = self.walker.simulate_walks(
                num_walks=self.num_paths, walk_length=self.path_length
            )
            self.walks = walks

        # Some walkers skip isolated nodes; align results so index == node id
        # and fill missing rows with -1 to keep shapes stable for callers.
        n_nodes = self.graph.number_of_nodes()
        if isinstance(walks, np.ndarray) and walks.size > 0 and walks.shape[0] != n_nodes:
            full_walks = np.full(
                (n_nodes, walks.shape[1], walks.shape[2]), -1, dtype=walks.dtype
            )
            for i in range(walks.shape[0]):
                start_node = walks[i, 0, 0]
                if 0 <= start_node < n_nodes:
                    full_walks[start_node] = walks[i]
            walks = full_walks

        return walks