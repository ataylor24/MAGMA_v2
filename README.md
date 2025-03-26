MAGMA v2

Model-Agnostic Graph Modeling & Augmentation

A major refactor of the original MAGMA library, MAGMA v2 provides a unified, extensible framework for sampling and augmenting graphs — both synthetic and from real-world datasets — to significantly boost structural diversity for graph-based research and machine learning.

⸻

🚀 New in v2
	•	Completely rewritten codebase for modularity, clarity, and performance
	•	Flexible graph sampler API supporting a wide range of synthetic generators:
	  •	Erdős–Rényi (ER)
	  •	Barabási–Albert (BA)
	  •	Stochastic Block Model (SBM)
	  •	Watts–Strogatz (WS)
	  •	Random Geometric Graph (RGG)
	  •	Custom parameterizable variants
	•	Real-world graph ingestion from common benchmark datasets (e.g., Cora, Citeseer, PubMed, ogbn-arxiv)
	•	Graph-source auto-mixing for improved evaluation set diversity
	•	Augmentation utilities for maximizing GED, edge perturbation, node feature noise, subgraph extraction, and more
	•	Improved logging and CLI support
⸻

📦 Installation

# Install via pip (PyPI)
pip install magma-v2

# Or install from source
git clone https://github.com/ataylor24/MAGMA_v2.git
cd MAGMA_v2
pip install -e .



⸻

🔧 Quickstart

from magma import GraphSampler

# Generate a synthetic Erdős–Rényi graph
sampler = GraphSampler(
    generator="erdos_renyi", n_nodes=1000, p=0.01
)
G = sampler.sample()

# Sample a real-world graph (Cora citation network)
real_sampler = GraphSampler.from_dataset("cora")
G_real = real_sampler.sample()

# Augment by randomly rewiring 10% of edges
G_aug = sampler.augment(G, method="edge_rewire", fraction=0.1)



⸻

🎛️ API Reference

GraphSampler

Initialization

def __init__(
    self,
    generator: str,
    n_nodes: int = None,
    **generator_kwargs,
)

Parameter	Type	Description
generator	str	Name of the synthetic generator ("erdos_renyi", "ba", etc.) or None for real-world dataset sampling
n_nodes	int	Number of nodes (synthetic only)
generator_kwargs	dict	Generator-specific parameters

Class Methods

@classmethod
GraphSampler.from_dataset(name: str) -> GraphSampler

Load and prepare a sampler for a real-world dataset.

Sampling

GraphSampler.sample() -> networkx.Graph

Return a NetworkX graph instance.

Augmentation

GraphSampler.augment(
    graph: networkx.Graph,
    method: str,
    fraction: float = 0.1,
    **kwargs,
) -> networkx.Graph

Supported methods: "edge_perturb", "edge_rewire", "node_mask", "subgraph".

⸻

📖 Examples & Tutorials

Check the examples/ directory for Jupyter notebooks demonstrating:
	•	Synthetic graph generation pipelines
	•	Real-world dataset loading and preprocessing
	•	Augmentation strategies for GNN robustness

⸻

🤝 Contributing

Contributions, issues, and feature requests are welcome!
	1.	Fork the repo
	2.	Create a feature branch (git checkout -b feature/your-feature)
	3.	Commit your changes (git commit -am 'Add new feature')
	4.	Push to the branch (git push origin feature/your-feature)
	5.	Open a Pull Request

Please follow our coding standards and run pytest before submitting.

⸻

📜 License

MIT License © 2025 Alexander Taylor
