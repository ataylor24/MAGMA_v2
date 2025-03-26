# =============================================================================
# Constants and Global Variables
# =============================================================================
FOUNDATION_MODELS = ['gpt-4o', 'o1', 'o1-mini', 'o3-mini', 'deepseek-reasoner']
OPENAI_MODELS = ['gpt-4o', 'o1', 'o1-mini', 'o3-mini']

OUTPUT_FORMATS = [
    "magma", "cot_analysis", "llama", "llama-instruct", "mistral", "mistral-instruct", "gpt4o",
    "gpt_o1", "gpt_o3", "deepseek_llama", "deepseek_qwen", "Deepseek_r1"
]

_OOD_TRAIN_LENGTHS = [6, 7, 8, 9, 10, 11]
_OOD_EVAL_LENGTHS = [12, 13, 14, 15, 16, 17, 20, 25]

SAMPLE_SPACE_SIZE = 1000

REASONING_STRATEGIES = {
    "Chain_of_Examples_Format": {},
    "Intermediate_Steps_W_Hints_Format": {
        "IO": "",
        "IO_cot": "",
        "IO_cod": "",
        "IS": "",
        "IS_cot": "",
        "IS_cod": "",
        },
    "Intermediate_Steps_Format": {
        "IO": "",
        "IO_cot": "",
        "IO_cod": "",
        "IS": "",
        "IS_cot": "",
        "IS_cod": "",
        },
    "Input-Output_Format": {
        "IO": "",
        "IO_cot": "",
        "IO_cod": "",
    }
}

COT_PROMPT = (
    f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    f"The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    f"The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
    f"i.e., <think> reasoning process here </think> <answer> answer here </answer>."
)

FORMATTED_ALGORITHMS = {
    "bfs": {
        "name": "Breadth-first Search",
        "goal": "reachability",
        "instruction": "List all known reachable nodes.",
        "output_format": "; Output Format: Reachable Nodes: [node1, node2, ...]",
        "eval_output_format": "Reachable Nodes:"
    },
    "dfs": {
        "name": "Depth-first Search",
        "goal": "reachability",
        "instruction": "List all known connected components.",
        "output_format": "; Output Format: Connected Components: [(node1, node2), (node3, node5), ...]",
        "eval_output_format": "Connected Components:"
    },
    "dijkstra": {
        "name": "Dijkstra's",
        "goal": "shortest-path",
        "instruction": "List the current distances between the source node and each other node.",
        "output_format": "; Output Format: Distances: [(node1, node2, weight), ...]",
        "eval_output_format": "Distances:"
    },
    "floyd_warshall": {
        "name": "Floyd-Warshall",
        "goal": "shortest-path",
        "instruction": "List the shortest path distances between all pairs of nodes.",
        "output_format": "; Output Format: Distances: [(node1, node2, weight), ...]",
        "eval_output_format": "Distances:"
    },
    "mst_prim": {
        "name": "Prim MST",
        "goal": "minimum spanning tree",
        "instruction": "List the current edgelist of the minimum spanning tree.",
        "output_format": "; Output Format: MST Edges: [(node1, node2, weight), ...]",
        "eval_output_format": "MST Edges:"
    }
}

TRAIN_TEST_SPLIT = {
    -1: [100, 50],
    3: [4, 4],
    4: [42, 22],
    5: [800, 100]
}
