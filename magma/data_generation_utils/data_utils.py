import argparse
import os
import json
import dill
import yaml
import random
import numpy as np
import collections
from typing import List
from .clrs_modules import probing
from globals import REASONING_STRATEGIES, FORMATTED_ALGORITHMS, OUTPUT_FORMATS, COT_PROMPT

Feedback = collections.namedtuple('Feedback', ['features', 'outputs'])

# =============================================================================
# Utility Functions
# =============================================================================

def iterate_sampler(sampler, batch_size):
    """Generator that yields batches from the sampler."""
    while True:
        yield sampler.next(batch_size)


def datapoint_to_dict(dp):
    """Convert a datapoint object to a dictionary."""
    return {"name": dp.name, "location": dp.location, "data": dp.data}


def datapoints_list_to_dict(dp_list):
    """Convert a list of datapoints into a dictionary keyed by datapoint name."""
    return {dp.name: datapoint_to_dict(dp) for dp in dp_list}


def hash_edgelist(edgelist):
    """Compute a canonical hash for a given edgelist."""
    canonical_edges = sorted([str(sorted(edge)) for edge in edgelist])
    return hash(",".join(canonical_edges))


def get_neighborhood(node, edgelist):
    """Return the sorted list of nodes adjacent to the given node."""
    neighbors = []
    for e in edgelist:
        if e[0] == node:
            neighbors.append(e[1])
        elif e[1] == node:
            neighbors.append(e[0])
    return sorted(neighbors)


def get_reachable_nodes(node, edgelist, visited_nodes):
    """Return reachable nodes from 'node' that have not been visited."""
    reachable = []
    for e in edgelist:
        if e[0] == node and e[1] not in visited_nodes:
            reachable.append(e[1])
        elif e[1] == node and e[0] not in visited_nodes:
            reachable.append(e[0])
    return sorted(reachable)


def translate_source_node(source_list):
    """Extract and return the source node index from a one-hot encoded list."""
    return int(np.nonzero(source_list.flatten())[0][0])


def translate_unweighted_graph(adj_matrix):
    """
    Convert an adjacency matrix (assumed to be unweighted) into an edge list.
    Only one direction is kept (i.e. i < j) to avoid duplicates.
    """
    adj_matrix = adj_matrix.squeeze()
    rows, cols = adj_matrix.shape
    edge_list = []
    for i in range(rows):
        for j in range(i + 1, cols):
            if adj_matrix[i][j] >= 1:
                edge_list.append((i, j))
    return edge_list

def remove_padding_from_hints(hints: List[probing.DataPoint]) -> List[probing.DataPoint]:
    """
    Removes trailing zero-padding from a single-sample (batch_size=1) set of hint DataPoints.

    Assumptions:
      - Each DataPoint dp in `hints` is shaped [T_max, 1, ...].
      - We detect how far the BFS/DFS, etc. actually ran by locating
        the largest time index that is not all zeros across any DataPoint.

    Returns:
      A new list of DataPoints in which each dp.data is sliced to remove trailing zeros.
      The shape remains [T_eff, 1, ...], where T_eff <= T_max.
    """

    if not hints:
        return hints  # Empty list, no work to do

    # Check that we're dealing with batch_size = 1
    batch_dim = hints[0].data.shape[1]
    if batch_dim != 1:
        raise ValueError(
            f"remove_padding_from_hints currently only handles batch_size=1, but got {batch_dim}."
        )

    # Find the largest time index (across all DataPoints) that isn't all zeros.
    T_max = hints[0].data.shape[0]
    last_nonzero = 0

    for dp in hints:
        data_2d = dp.data[:, 0, ...]  # shape [T_max, ...] (strip off the batch dim)
        # We'll call a row "all zero" if its entire slice is zero.
        sums = np.sum(np.abs(data_2d), axis=tuple(range(1, data_2d.ndim)))
        nonzero_rows = np.where(sums != 0)[0]
        if len(nonzero_rows) > 0:
            candidate = nonzero_rows[-1]  # the last time index with nonzero data
            if candidate > last_nonzero:
                last_nonzero = candidate

    # Slice each DataPoint's data up to `last_nonzero + 1`
    new_hints = []
    for dp in hints:
        data_sliced = dp.data[: last_nonzero + 1, 0:1, ...]  # keep batch dim=1
        new_dp = probing.DataPoint(dp.name, dp.location, dp.type_, data_sliced)
        new_hints.append(new_dp)

    return new_hints

# =============================================================================
# File I/O and Serialization Functions
# =============================================================================

def dump_yml(outfile, data):
    yaml.dump(data, open(outfile, 'w'), default_flow_style=False)


def load_json(filepath):
    return json.load(open(filepath, 'r'))


def write_json(outfile, data):
    json.dump(data, open(outfile, "w"))


def write_pickle(outfile, data):
    dill.dump(data, open(outfile, 'wb'))


def write_clrs_format(outfile, data):
    write_pickle(outfile, data)


def json_to_string(data):
    prompt = data.get("prompt", "")
    messages = data.get("messages", [])
    output_string = prompt + "\n\n"
    for message in messages:
        content = message.get("content", "")
        output_string += content + "\n"
    return {"content": output_string.strip()}

# =============================================================================
# Content Formatting Functions
# =============================================================================

def write_clrs_chat_format(data):
    chat_data = []
    for idx in data:
        chat_data.append({
            "traj_id": f"{idx}",
            "prompt": data[idx]["inputs"],
            "messages": [
                {
                    "role": "system",
                    "content": data[idx]["inputs"],
                },
                {
                    "role": "assistant",
                    "content": data[idx]["outputs"],
                }
            ],
        })
    return chat_data


def write_chat_format(reasoning_type, reasoning_strategy, data_sect, data):
    chat_data = []
    for idx, item in data.items():
        algorithm = item["inputs"][0]
        edge_list = item["inputs"][1]
        source_node = item["inputs"][2]
        hints_list = item["hints"]
        
        messages = []
        
        # Build system message (initial prompt) in a structured way.
        system_message_parts = []
        system_message_parts.append(
            f"Please perform the {FORMATTED_ALGORITHMS[algorithm]['name']} algorithm for {FORMATTED_ALGORITHMS[algorithm]['goal']} on the following undirected graph:"
        )
        system_message_parts.append(
            f"Edgelist: [{','.join(str(tuple(edge)) for edge in edge_list)}]."
        )
        if source_node != "":
            system_message_parts.append(f"Source Node: {source_node}.")
        system_message_parts.append(FORMATTED_ALGORITHMS[algorithm]['output_format'])
        system_message_parts.append(REASONING_STRATEGIES[reasoning_type][reasoning_strategy])
        
        # Optionally include the first hint if provided.
        if hints_list and hints_list[0].strip() and reasoning_type == "Intermediate_Steps_W_Hints_Format":
            system_message_parts.append(hints_list[0].strip())
        
        system_message_parts.append(FORMATTED_ALGORITHMS[algorithm]['instruction'])
        system_message = "\n".join(system_message_parts)
        
        # If chain-of-thought is part of the strategy, prepend the COT prompt.
        if "cot" in reasoning_strategy:
            system_message = f"{COT_PROMPT}\n" + system_message
        
        messages.append({"role": "system", "content": system_message})
        init_prompt = system_message

        # Process remaining hints (starting at index 1)
        for i, hint in enumerate(hints_list[1:], start=1):
            # For Input-Output format, skip intermediate steps except the last.
            if reasoning_type == "Input-Output_Format" and i < len(hints_list) - 1:
                continue

            role = "assistant" if i % 2 == 1 else "user"
            if role == "assistant":
                content = hint.strip() + "\n"
            elif role == "user":
                if reasoning_type == "Intermediate_Steps_W_Hints_Format":
                    content = hint.strip() + "\nPlease perform the next step. " + FORMATTED_ALGORITHMS[algorithm]['instruction'] + "\n"
                elif reasoning_type == "Intermediate_Steps_Format":
                    content = "Please perform the next step. " + FORMATTED_ALGORITHMS[algorithm]['instruction'] + "\n"
                else:
                    content = ""
            # For the last hint (except for bfs), use the output provided.
            if i == len(hints_list) - 1 and algorithm not in ["bfs"]:
                content = item["outputs"]
            messages.append({"role": role, "content": content})
            
        datapoint = {"algorithm": algorithm, "traj_id": str(idx), "prompt": init_prompt, "messages": messages}
        chat_data.append(datapoint)
    return chat_data

# =============================================================================
# Directory and Selection Functions
# =============================================================================

def resolve_output_dirs(output_dir, output_formats):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    clrs_data_dir = os.path.join(output_dir, "clrs_data")
    clrs_text_data_dir = os.path.join(output_dir, "clrs_text_data")
    clrs_text_no_hints_data_dir = os.path.join(output_dir, "clrs_text_no_hints_data")
    llm_data_dir = os.path.join(output_dir, "magma_data")
    for d in [clrs_data_dir, clrs_text_data_dir, clrs_text_no_hints_data_dir, llm_data_dir]:
        if not os.path.exists(d):
            os.mkdir(d)
    magma_formatted_data_dirs = {}
    for output_format in output_formats:
        magma_formatted_data_dir = os.path.join(llm_data_dir, output_format)
        magma_formatted_data_dirs[output_format] = magma_formatted_data_dir
        if not os.path.exists(magma_formatted_data_dir):
            os.mkdir(magma_formatted_data_dir)
        for reasoning_type in REASONING_STRATEGIES:
            for reasoning_strategy in REASONING_STRATEGIES[reasoning_type]:
                reasoning_strategy_dir = os.path.join(magma_formatted_data_dir, reasoning_strategy)
                if not os.path.exists(reasoning_strategy_dir):
                    os.mkdir(reasoning_strategy_dir)
    return clrs_data_dir, clrs_text_data_dir, clrs_text_no_hints_data_dir, magma_formatted_data_dirs


def count_random_selections(pool, n):
    """
    Randomly selects indices from the given pool n times and counts the number 
    of times each index was selected.
    """
    k = len(pool)
    counts = [0] * k
    for _ in range(n):
        idx = random.randint(0, k - 1)
        counts[idx] += 1
    return counts

# =============================================================================
# Argument Parsing
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Jointly constructs data for the MAGMA benchmark and CLRS/CLRS-Text benchmarks for comparison.")
    parser.add_argument("algorithm", type=str, choices=['bfs', 'dfs', 'dijkstra', 'floyd_warshall', 'mst_prim', 'all'], 
                        help="Algorithm must be one of: 'bfs', 'dfs', 'dijkstra', 'floyd_warshall', or 'mst_prim'.")
    parser.add_argument("-graph_sizes", "--graph_sizes", nargs="+", type=int,
                        default=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 50],
                        help="Number of nodes present in the graphs generated. Default behavior sets num_samples to the number of training datapoints.")
    parser.add_argument("-num_samples", "--num_samples", type=int, default=-1,
                        help="Number of data samples to generate.")
    parser.add_argument("-neg_edges", "--neg_edges", type=bool, default=True,
                        help="Include negative edges.")
    parser.add_argument("-seed", "--seed", type=int, default=100898,
                        help="Random seed used in constructing the algorithm sampler.")
    parser.add_argument("-output_dir", "--output_dir", type=str,
                        default="/local/ataylor2/algorithmic_reasoning/PBR",
                        help="Output directory. Will create folders named after the algorithm for which data is generated.")
    parser.add_argument("-output_formats", "--output_formats", type=list, default=["cot_analysis"],
                        choices=OUTPUT_FORMATS, help="Output format for dataset")
    parser.add_argument("-ood_generation", "--ood_generation", type=bool, default=True,
                        help="Generate mixed graph size data with out-of-distribution testing.")
    args = parser.parse_args()
    return args