# MAGMA v2: Multistep Algorithmic Reasoning Benchmark

## Paper
**Are Large-Language Models Graph Algorithmic Reasoners?** [[pdf](https://arxiv.org/pdf/2410.22597)]

## Summary

**MAGMA v2** evaluates large language models (LLMs) on five classical graph algorithms—BFS, DFS, Dijkstra’s, Floyd–Warshall, and Prim’s MST—with an emphasis on **multistep reasoning** and **intermediate-step accuracy**. Despite progress in LLMs, structured tasks like graph algorithms remain challenging. This benchmark highlights where LLMs excel and where they fall short.

---

## Features
- **Five Graph Algorithms**: BFS, DFS, Dijkstra’s, Floyd–Warshall, and MST (Prim).
- **Intermediate Steps**: Measure chain-of-thought accuracy, not just final outputs.
- **Multiple Sources**: Synthetic and real-world graph data.
- **Flexible Prompting**: Chain-of-thought or instruction-based queries.
- **Modular & Extensible**: Easy to add new tasks or adapt data generation.

---

## Installation

1. **Clone the Repo**:
   ```bash
   git clone https://github.com/ataylor24/MAGMA_v2.git
   cd MAGMA_v2
   ```
2. **Install** (editable mode recommended):
   ```bash
   pip install -e .
   ```

**Conda users**:
```bash
conda env create --file environment.yml
conda activate nar2
pip install -e .
```

---

## Data Generation

**Standard usage for all algorithms**:
```bash
python magma_generation.py all
```
This runs BFS, DFS, Dijkstra’s, Floyd–Warshall, and Prim’s MST with default settings in `globals.py`. Adjust via CLI flags as needed.

### Key Defaults (in `globals.py`)
- **`TRAIN_TEST_SPLIT`** maps graph sizes to train/test counts.
- **`_OOD_TRAIN_LENGTHS`** & **`_OOD_EVAL_LENGTHS`** for out-of-distribution sizes.
- **`OUTPUT_FORMATS`**: e.g. `cot_analysis`, `magma`, etc.
- **`FORMATTED_ALGORITHMS`**: instructions/output formatting for each algorithm.
- **`COT_PROMPT`**: chain-of-thought prompt template.

### Example Usage
```bash
python sample_data.py bfs \
  --graph_sizes 5 7 10 \
  --seed 1234 \
  --ood_generation False \
  --output_dir /path/to/output \
  --output_formats cot_analysis
```
- `bfs` can be replaced with `dfs`, `dijkstra`, `floyd_warshall`, `mst_prim`, or `all`.
- `graph_sizes` sets node counts.
- `ood_generation` enables out-of-distribution sampling.
- `output_formats` toggles data format (chain-of-thought, etc.).

---

## Performance Metrics
- **Exact Match Accuracy**: Matches final solution exactly.
- **F1 Score**: Partial correctness.
- **Intermediate Steps**: Evaluates step-by-step reasoning.
- **Final Step**: Only the final result.
- **Trajectory**: Entire chain-of-thought.
- **Independent**: Each step treated independently.

---

## Contributing
1. **Fork** the repo.
2. **Create** a branch.
3. **Commit** changes.
4. **Push** the branch.
5. **Open** a Pull Request.

Please run `pytest` before submitting.

---

## Reproducibility
- **Seed**: `100898` ensures consistent data generation.
- Other defaults in code or config.

---

## License
MIT License © 2025 Alexander Taylor

---

## Acknowledgements
- Data adapted from [CLRS benchmark](https://github.com/google-deepmind/clrs)
- Model training adapted from [Hugging Face Alignment Handbook](https://github.com/huggingface/alignment-handbook.git)

---

## Contact
Questions/feedback? Open an issue or email `ataylor2@cs.ucla.edu`.

---

**Thank you for using MAGMA v2!** Accelerate research into LLM-based graph algorithmic reasoning.

