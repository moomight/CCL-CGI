import os

import h5py
import networkx as nx
import numpy as np
import torch

from utils.node2vec import Node2vec


def create_subgraphs_randomwalk(
    dataset,
    adj,
    n_graphs,
    n_neighbors,
    idx,
    p=1.5,
    q=1.2,
    use_cancer_ppi=False,
    sp_dir=None,
    dataset_name="CCL-CGI",
):
    print(f"Creating random walk subgraphs for dataset: {dataset} with p={p}, q={q}...")
    my_graph = nx.Graph()
    edge_index_begin, edge_index_end = np.where(adj > 0)
    edge_index = np.array([edge_index_begin, edge_index_end]).transpose().tolist()
    n_nodes = adj.shape[0]
    pmat = np.ones(shape=(n_nodes, n_nodes), dtype=int) * np.inf
    print(n_nodes)
    print(len(edge_index_end))
    my_graph = nx.from_numpy_array(adj)
    my_graph = my_graph.to_directed()
    walks = Node2vec(
        graph=my_graph,
        path_length=n_neighbors,
        num_paths=n_graphs,
        p=p,
        q=q,
        workers=6,
        dw=False,
    ).get_walks()

    new_walks = np.zeros(shape=(n_nodes, n_graphs, n_neighbors), dtype=int)
    for i in range(walks.shape[0]):
        new_walks[walks[i][0][0], :, :] = walks[i]

    walks = new_walks

    if sp_dir is None:
        sp_dir = os.path.join(os.getcwd(), "sp", dataset_name)

    if use_cancer_ppi:
        save_path = os.path.join(sp_dir, dataset + "_sp_withCancer.h5")
    else:
        save_path = os.path.join(sp_dir, dataset + "_sp.h5")

    if os.path.exists(save_path):
        use_cached_sp = True
    else:
        use_cached_sp = False

    if use_cached_sp is False:
        print("Computing shortest path distances...")

        def _torch_all_pairs_shortest_paths(adj_matrix: np.ndarray, batch_size: int = 256, device: str = None):
            n = adj_matrix.shape[0]

            edges = np.where(adj_matrix > 0)
            edge_index = np.vstack(edges)
            values = np.ones(edge_index.shape[1], dtype=np.float32)

            device = device or ("cuda" if torch.cuda.is_available() else "cpu")

            indices = torch.tensor(edge_index, dtype=torch.long, device=device)
            values_t = torch.tensor(values, dtype=torch.float32, device=device)
            adj_sparse = torch.sparse_coo_tensor(indices, values_t, (n, n), device=device).coalesce()

            result = torch.full((n, n), -1, dtype=torch.int32, device=device)

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_nodes = torch.arange(start, end, device=device, dtype=torch.long)
                batch_size_actual = batch_nodes.shape[0]

                frontier = torch.zeros((batch_size_actual, n), device=device, dtype=torch.bool)
                frontier[torch.arange(batch_size_actual, device=device), batch_nodes] = True
                visited = frontier.clone()
                dist = torch.full((batch_size_actual, n), -1, device=device, dtype=torch.int32)
                dist[torch.arange(batch_size_actual, device=device), batch_nodes] = 0

                step = 0
                while frontier.any():
                    neighbors = torch.sparse.mm(frontier.float(), adj_sparse).to(torch.bool)
                    neighbors = neighbors & (~visited)
                    step += 1

                    if neighbors.any():
                        dist[neighbors] = step

                    visited = visited | neighbors
                    frontier = neighbors

                result[start:end] = dist

            return result.cpu().numpy()

        pmat = None
        if torch.cuda.is_available():
            try:
                print("Attempting GPU-accelerated shortest paths with torch...")
                pmat = _torch_all_pairs_shortest_paths(adj, batch_size=256, device="cuda")
                print("GPU shortest path computation completed using torch.")
            except Exception as exc:
                print(f"GPU computation failed: {exc}")

        if pmat is None:
            try:
                print("Using torch on CPU for shortest path computation...")
                pmat = _torch_all_pairs_shortest_paths(adj, batch_size=64, device="cpu")
                print("CPU torch shortest path computation completed.")
            except Exception as exc:
                print(f"CPU torch shortest path computation failed: {exc}")

        if pmat is None:
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import floyd_warshall

            print("Using scipy (CPU) Floyd-Warshall algorithm...")

            sparse_adj = csr_matrix(adj)

            pmat = floyd_warshall(sparse_adj, directed=False, unweighted=True)

            pmat[pmat == np.inf] = -1
            pmat = pmat.astype(int)

            print(f"CPU shortest path computation completed. Matrix shape: {pmat.shape}")

        new_file = h5py.File(save_path, "w")
        new_file.create_dataset(
            name="sp",
            shape=(n_nodes, n_nodes),
            data=pmat,
            compression="gzip",
            compression_opts=4,
            chunks=(1024, 1024),
        )
        new_file.close()
        print(f"Saved shortest path matrix to {save_path}")
    else:
        print(f"Loading cached shortest path matrix from {save_path}")
        f = h5py.File(save_path, "r")
        pmat = f["sp"][:]
        pmat[pmat == np.inf] = -1
        f.close()

    subgraphs_list = []
    for id in range(n_nodes):
        sub_subgraph_list = []
        for g in range(n_graphs):
            node_feature_id = np.array(walks[id, g, :], dtype=int)

            attn_bias = np.concatenate([np.expand_dims(i[node_feature_id, :][:, node_feature_id], 0) for i in [pmat]])

            sub_subgraph_list.append(attn_bias)

        subgraphs_list.append(sub_subgraph_list)

    return walks, np.array(subgraphs_list)
