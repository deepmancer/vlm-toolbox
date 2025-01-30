import networkx as nx


def create_nx_graph(nodes_df, edges_df):
    G = nx.DiGraph()
    for _, row in nodes_df.iterrows():
        G.add_node(row['idx'])
    
    for _, row in edges_df.iterrows():
        G.add_edge(row['idx'], row['parent_idx'])
        # G.add_edge(row['parent_idx'], row['idx'])

    return G

def find_smallest_subtree(graph, node_ids, to_prune_roots_cnt=0):
    if not node_ids:
        return nx.Graph()
        
    subtree = nx.Graph()
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            lca = nx.lowest_common_ancestor(graph, node_ids[i], node_ids[j])
            path1 = nx.shortest_path(graph, node_ids[i], lca)
            path2 = nx.shortest_path(graph, node_ids[j], lca)
            nx.add_path(subtree, path1)
            nx.add_path(subtree, path2)

    if to_prune_roots_cnt:
        root = 0
        pruned_cnt = 0

        while True:
            if not subtree.nodes:
                return None
    
            neighbors = list(subtree.neighbors(root))
            if len(neighbors) > 1 or pruned_cnt == to_prune_roots_cnt:
                return subtree
            subtree.remove_node(root)
            root = neighbors[0]
            pruned_cnt += 1

    return subtree
