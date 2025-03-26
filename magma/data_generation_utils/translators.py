import numpy as np
from collections import deque
from .data_utils import (
    translate_source_node, 
    translate_unweighted_graph,
    datapoints_list_to_dict,
    remove_padding_from_hints
)

def translate_outputs(alg, outputs, final_d=None):
    """
    Translate the algorithm outputs into a human-readable string.
    """
    outputs_dict = datapoints_list_to_dict(outputs)
    if alg == "bfs":
        list_out_preds = outputs_dict["pi"]["data"][0]
        return bfs_translate_output(list_out_preds)
    if alg == "dfs":
        return f"Connected Components: {final_d}"
    elif alg in ["dka", "bfd"]:
        raise NotImplementedError(f"No hint translation implemented for {alg}")
    elif alg in ["dijkstra", "floyd_warshall"]:
        return f"Distances: {final_d}"
    elif alg == "mst_prim":
        return f"MST Edges: {final_d}"
    else:
        raise NotImplementedError(f"No output translation implemented for {alg}")

def translate_hints(alg, neg_edges, edgelist_lookup, hints, source=None):
    """
    Translate hints based on the algorithm type.
    """
    hints = remove_padding_from_hints(hints)
    hints_dict = datapoints_list_to_dict(hints)
    if alg == "bfs":
        list_reach_h = preprocess_hint_matrix(alg, hints_dict["reach_h"]["data"])
        list_pred_h = preprocess_hint_matrix(alg, hints_dict["pi_h"]["data"])
        return bfs_translate_reach_pred_h(neg_edges, edgelist_lookup, list_reach_h, list_pred_h)
    elif alg == "dfs":
        list_pred_h = preprocess_hint_matrix(alg, hints_dict["pi_h"]["data"])
        list_color_h = preprocess_hint_matrix(alg, hints_dict["color"]["data"])
        list_source_h = preprocess_hint_matrix(alg, hints_dict["s"]["data"])
        return dfs_translate_list_h(neg_edges, edgelist_lookup, list_pred_h, list_color_h, list_source_h)
    elif alg == "floyd_warshall":
        dist_matrix = hints_dict["D"]["data"]
        return fw_translate_hints(dist_matrix)
    elif alg == "dijkstra":
        return translate_dijkstra_hints(hints_dict, source)
    elif alg == "mst_prim":
        return translate_mst_prim_hints(hints_dict, source)
    else:
        raise NotImplementedError(f"No hint translation implemented for {alg}")

def translate_inputs(alg, inputs):
    """
    Translate the input datapoints into algorithm-specific inputs.
    Returns a tuple: (algorithm, edge list, source).
    """
    inputs_dict = datapoints_list_to_dict(inputs)
    if alg == "bfs":
        edge_list = translate_unweighted_graph(inputs_dict["adj"]["data"])
        source = translate_source_node(inputs_dict["s"]["data"])
        return alg, edge_list, source
    elif alg in ["floyd_warshall", "dijkstra", "mst_prim"]:
        adj_matrix = np.squeeze(inputs_dict["adj"]["data"])
        weights = np.squeeze(inputs_dict["A"]["data"])
        edge_set = set()
        edge_list = []
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix[i])):
                if adj_matrix[i][j] == 1 and weights[i][j] != 0 and i != j:
                    edge = (i, j, float(weights[i][j]))
                    reverse_edge = (j, i, float(weights[j][i]))
                    if reverse_edge not in edge_set:
                        edge_list.append(edge)
                        edge_set.add(edge)
        source = "" if alg == "floyd_warshall" else translate_source_node(inputs_dict["s"]["data"])
        return alg, edge_list, source
    elif alg == "dfs":
        adj_matrix = np.squeeze(inputs_dict["adj"]["data"])
        weights = np.squeeze(inputs_dict["A"]["data"])
        edge_set = set()
        edge_list = []
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix[i])):
                if adj_matrix[i][j] == 1 and weights[i][j] != 0 and i != j:
                    edge = (i, j)
                    reverse_edge = (j, i)
                    if reverse_edge not in edge_set:
                        edge_list.append(edge)
                        edge_set.add(edge)
        return alg, edge_list, ""
    else:
        raise NotImplementedError(f"No input translation implemented for {alg}")
    
def preprocess_hint_matrix(alg, matrix_h):
    """
    Process hint matrices for graph-based algorithms.
    For BFS/DFS, unflatten the 2D list representation.
    """
    if alg in ["bfs", "dfs"]:
        return [unflat[0] for unflat in matrix_h.astype(int).tolist()]
    elif alg in ["dka", "bfd"]:
        raise NotImplementedError(f"No hint translation implemented for {alg}")
    else:
        raise NotImplementedError(f"No hint translation implemented for {alg}")

def fw_translate_hints(distance_matrix):
    """
    Translate hints for Floyd–Warshall.
    Returns a tuple of (list of hint strings, final edge list).
    """
    hints = []
    N = distance_matrix.shape[0]
    final_edge_list = []
    for i in range(1, N):
        hints.append(f"Queue: {list(range(i - 1, N + 1))}\nDequeue {i - 1}")
        current_dist_matrix = distance_matrix[i, 0]
        edge_list = []
        for j in range(N):
            for k in range(j + 1, N):
                if current_dist_matrix[j, k] != 0:
                    edge_list.append((j, k, current_dist_matrix[j, k]))
        hints.append(f"Distances: {edge_list}")
        final_edge_list = edge_list  # Update final edge list at each step
    return hints, final_edge_list

def translate_dijkstra_hints(hints_dict, source):
    """
    Translate hint dictionary for Dijkstra's algorithm.
    Returns a tuple of (list of hint strings, final distance list).
    """
    d = hints_dict["d"]["data"]
    mark = hints_dict["mark"]["data"]
    in_queue = hints_dict["in_queue"]["data"]
    u = hints_dict["u"]["data"]

    hints = []
    N = d.shape[0]
    nodes = d.shape[2]
    final_distances = []

    for i in range(N):
        priority_queue = [(j, d[i, 0, j]) for j in range(nodes) if in_queue[i, 0, j] == 1]
        priority_queue = sorted(priority_queue, key=lambda x: x[1] if x[1] != 0 else float('inf'))
        unvisited_nodes = [j for j in range(nodes) if mark[i, 0, j] == 0]
        visited_nodes = [j for j in range(nodes) if mark[i, 0, j] == 1]

        hints.append(
            f"Step {i}:\nPriority Queue: {priority_queue}\nUnvisited Nodes: {unvisited_nodes}\nVisited Nodes: {visited_nodes}"
        )

        if not (mark[i, 0].any() or in_queue[i, 0].any() or u[i, 0].any()):
            hints.append("Queue is empty. Algorithm terminates.")
            break
        else:
            final_distances = [(source, j, d[i, 0, j]) for j in range(nodes) if d[i, 0, j] != 0]
            hints.append(f"Distances: {final_distances}")
    return hints, final_distances

def translate_mst_prim_hints(hints_dict, source):
    """
    Translate hint dictionary for MST (Prim's algorithm).
    Returns a tuple of (list of hint strings, final MST edge list).
    """
    key = hints_dict["key"]["data"]
    pi_h = hints_dict["pi_h"]["data"]
    mark = hints_dict["mark"]["data"]
    in_queue = hints_dict["in_queue"]["data"]
    u = hints_dict["u"]["data"]

    hints = []
    N = key.shape[0]
    nodes = key.shape[2]
    final_mst_edges = []

    for i in range(N):
        priority_queue = sorted([j for j in range(nodes) if in_queue[i, 0, j] == 1])
        unvisited_nodes = [j for j in range(nodes) if mark[i, 0, j] == 0]
        visited_nodes = [j for j in range(nodes) if mark[i, 0, j] == 1]

        hints.append(
            f"Step {i}:\nPriority Queue: {priority_queue}\nUnvisited Nodes: {unvisited_nodes}\nVisited Nodes: {visited_nodes}"
        )

        if not (mark[i, 0].any() or in_queue[i, 0].any() or u[i, 0].any()):
            hints.append("Queue is empty. Algorithm terminates.")
            break
        else:
            mst_edges = [
                (int(min(pi_h[i, 0, j], j)), int(max(pi_h[i, 0, j], j)), key[i, 0, j])
                for j in range(nodes) if pi_h[i, 0, j] != j
            ]
            mst_edges = [(i, j, w) for i, j, w in mst_edges if i < j]
            hints.append(f"MST Edges: {mst_edges}")
            final_mst_edges = mst_edges
    return hints, final_mst_edges

def dfs_translate_list_h(neg_edges, edgelist_lookup, list_pred_h, list_color_h, list_source_h):
    """
    Translate hints for DFS.
    Returns a tuple of (list of hint strings, final connected components as tuples).
    """
    reach_stack = []
    hints = []
    final_groupings = []
    visited = set()
    current_component = []

    def get_local_neighborhood(node, edgelist):
        """Local helper to compute neighborhood."""
        neighbors = set()
        for edge in edgelist:
            if edge[0] == node:
                neighbors.add(edge[1])
            elif edge[1] == node:
                neighbors.add(edge[0])
        return list(neighbors)

    for i in range(len(list_color_h)):
        colors = list_color_h[i]
        try:
            source_node = list_source_h[i].index(1.0)
        except Exception as e:
            raise ValueError(e)
        for j, node_color in enumerate(colors):
            if node_color == [0, 1, 0] and j not in visited:
                reach_stack.append(j)
                visited.add(j)
                if not current_component:
                    current_component = [source_node]
                current_component.append(j)
                hint_str = (
                    f"Stack: {reach_stack}, "
                    f"Pop Node: {reach_stack[-1]}, "
                    f"1-hop Neighborhood of {reach_stack[-1]}: {get_local_neighborhood(reach_stack[-1], edgelist_lookup)}."
                )
                hints.append(hint_str)
                formatted_components = ', '.join(
                    f"({', '.join(map(str, sorted(set(component))))})"
                    for component in final_groupings + [current_component]
                )
                hints.append(f"Connected Components: [{formatted_components}]")
            elif node_color == [0, 0, 1] and reach_stack:
                reach_stack.pop()

        if not reach_stack and current_component:
            final_groupings.append(current_component)
            current_component = []

    # Filter and merge connected components
    final_groupings = [
        sorted(set(component))
        for component in final_groupings
        if any(get_local_neighborhood(node, edgelist_lookup) for node in component)
    ]
    merged_groupings = []
    for component in final_groupings:
        added = False
        for mg in merged_groupings:
            if any(node in mg for node in component):
                mg.update(component)
                added = True
                break
        if not added:
            merged_groupings.append(set(component))
    final_groupings = [sorted(mg) for mg in merged_groupings]
    formatted_components = [tuple(component) for component in final_groupings]
    return hints, formatted_components

def bfs_translate_output(list_pred):
    """
    Translate BFS predecessor array (list_pred) into a human-readable string.

    list_pred is typically an array where list_pred[node] = predecessor(node),
    and a node is considered reachable if its predecessor != itself.
    """
    list_out_idxs = [str(node_idx) 
                     for node_idx, pred_idx in enumerate(list_pred) 
                     if pred_idx != node_idx]
    return f"Reachable Nodes: [{', '.join(list_out_idxs)}]"


def bfs_translate_reach_pred_h(neg_edges, edgelist_lookup, list_reach_h, list_pred_h):
    """
    Translate BFS 'hint' matrices (list_reach_h, list_pred_h) into BFS steps,
    one node-dequeue per hint.

    Each entry in list_reach_h[level] is a binary array indicating which
    nodes become newly 'reached' at BFS level = level. Similarly, each
    entry in list_pred_h[level] gives the 'predecessor' for each node at that level.

    We reconstruct a BFS queue from these levels so that we can emit BFS step-by-step hints:
      1) The current contents of the BFS queue
      2) The node we dequeue (pop) from the queue
      3) Its unvisited neighbors
      4) The cumulative set of reachable nodes

    This version ensures each BFS "dequeue" event becomes its own separate hint
    in the returned 'hints' list (rather than bundling an entire BFS level
    into a single hint).
    """
    dict_reach_h = {}
    neighborhood_h = {}
    visited = set()
    reach_h_queue = []   # BFS "levels"; each is a set/list of nodes discovered at that level
    
    # 1) Collect BFS levels from (list_reach_h, list_pred_h)
    for reach_arr, pred_arr in zip(list_reach_h, list_pred_h):
        # If both arrays are all-zero, there's nothing to process for that level
        if sum(reach_arr) == 0 and sum(pred_arr) == 0:
            continue
        
        level_queue = set()
        
        for node_idx, (reach_flag, pred_node_idx) in enumerate(zip(reach_arr, pred_arr)):
            if pred_node_idx not in dict_reach_h:
                dict_reach_h[pred_node_idx] = set()
                neighborhood_h[pred_node_idx] = set()
            
            # If node_idx is newly reached in this level:
            if reach_flag == 1:
                # If its predecessor is different from itself, store that edge
                if node_idx != pred_node_idx:
                    dict_reach_h[pred_node_idx].add((node_idx, pred_node_idx))
                    neighborhood_h[pred_node_idx].add(node_idx)
                
                # Mark it visited & queue for BFS
                if node_idx not in visited:
                    level_queue.add(node_idx)
                    visited.add(node_idx)
        
        # Append the newly discovered nodes of this BFS level
        if level_queue:
            reach_h_queue.append(sorted(level_queue))
    
    # 2) Generate BFS step-by-step hints, one node per “hint”
    hints = []
    reachable_nodes = set()
    
    if not reach_h_queue:
        # If there were no discovered levels at all, just return empty or some fallback
        return hints
    
    # Start the actual BFS queue with the first discovered BFS level
    bfs_queue = deque(reach_h_queue[0])
    # Mark those first-level nodes as reachable
    for n in reach_h_queue[0]:
        reachable_nodes.add(n)
    
    level_idx = 0
    # We walk through each BFS level
    while level_idx < len(reach_h_queue):
        # For BFS level `level_idx`, we have a set/list of newly discovered nodes:
        level_nodes = reach_h_queue[level_idx]

        # For each node discovered at this level, we create a separate BFS step/hint
        for _ in level_nodes:
            if not bfs_queue:
                # If the queue empties out before we finish, we break
                break

            current_hint_lines = []
            current_hint_lines.append(f"Queue: {list(bfs_queue)}")

            # Dequeue the front of the queue
            current_source = bfs_queue.popleft()
            
            # “Unvisited neighborhood” from our BFS-hint data structure
            neighbors = sorted(neighborhood_h.get(current_source, []))
            
            current_hint_lines.append(
                f"Dequeue: {current_source}\n"
                f"Unvisited neighborhood of {current_source}: {neighbors}"
            )
            
            # Enqueue neighbors that haven't been marked reachable yet
            for nbr in neighbors:
                if nbr not in reachable_nodes:
                    bfs_queue.append(nbr)
                    reachable_nodes.add(nbr)

            # Append this single-node BFS expansion as one “hint”
            hints.append("\n".join(current_hint_lines))
            hints.append(f"Reachable Nodes: {sorted(reachable_nodes)}")
        
        level_idx += 1
    
    return hints