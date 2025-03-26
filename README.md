# MAGMA v2: Multistep AlGorithMic ReAsoning Benchmark

## Paper

**Are Large-Language Models Graph Algorithmic Reasoners?** [[pdf](https://arxiv.org/pdf/2410.22597)]

## Summary

**MAGMA v2** extends and refines the original MAGMA benchmark to evaluate how large language models (LLMs) perform on classical graph algorithms with explicit intermediate-step reasoning. Building on lessons learned from the first iteration, MAGMA v2 focuses on improved data generation, structured prompting, and comprehensive metrics to uncover the strengths and limitations of LLMs on fundamental graph problems:

- **Breadth-First Search (BFS)**
- **Depth-First Search (DFS)**
- **Dijkstra’s Shortest Path**
- **Floyd–Warshall All-Pairs Shortest Paths**
- **Prim’s Minimum Spanning Tree**

LLMs often struggle with multi-step structured tasks—especially those involving graph-based reasoning. MAGMA v2 quantifies their performance through intermediate steps and final results, providing insights into the models’ capabilities and deficiencies.

> **We are actively updating this benchmark!** Please reach out to us at `ataylor2@cs.ucla.edu` or open an issue with any update requests or bug reports.

---

## Features

- **Comprehensive Benchmark:** Evaluates LLM performance on five classical graph algorithms.
- **Intermediate Steps Evaluation:** Focuses on the accuracy of chain-of-thought or step-by-step reasoning.
- **Multiple Graph Sources:** Integrates both synthetic graphs and real-world graph datasets.
- **Advanced Prompting Techniques:** Explores advanced algorithmic instructions, chain-of-thought, and self-check strategies.
- **Easy Extensibility:** Add new algorithms, datasets, or evaluation routines with minimal code changes.

---

## Installation & Package Setup

### Prerequisites
- **Python 3.10 or higher**
- (Optional) **Conda** for environment management

### Clone the Repository

```bash
git clone https://github.com/ataylor24/MAGMA_v2.git
cd MAGMA_v2
```

### Install or Set Up the Package

You can install MAGMA v2 as a standard Python package:

```bash
pip install -e .
```

This enables an “editable” install, so any local changes to the code are immediately available.

Alternatively, if you prefer to use **Conda**, you can:

1. Create an environment using the included `environment.yml`:
   ```bash
   conda env create --file environment.yml
   ```
2. Activate the environment:
   ```bash
   conda activate nar2
   ```
3. Install the package:
   ```bash
   pip install -e .
   ```

---

### Prerequisites
- **Python 3.10 or higher**
- (Optional) **Conda** for environment management

### Clone the Repository

```bash
git clone https://github.com/ataylor24/MAGMA_v2.git
cd MAGMA_v2
```

### Create a Conda Environment

We provide an example environment file (`environment.yml`) to ensure reproducible setups:

```bash
conda env create --file environment.yml
```

### Activate the Conda Environment

```bash
conda activate nar2
```

---

## Training Baseline Models

We include scripts in the `run_scripts/` directory as a reference for how to train and evaluate baseline models on MAGMA v2:

```bash
bash run_scripts/bfs_CoT.sh
```

Feel free to modify the scripts for different algorithms (DFS, Dijkstra, etc.) or custom hyperparameters.

---

## Running Inference with Trained Models

To evaluate a trained model on a particular algorithm, use the scripts in `inference_scripts/`:

```bash
bash inference_scripts/bfs_CoT.sh
```

By default, these scripts assume you have a checkpoint and configuration file with the same naming conventions as in `run_scripts/`.

---

## Configuration

You can customize the model training and inference settings in `configuration_example/config_qlora.yaml` (or similar config files). This includes:

- Model architecture and size
- Optimizer, learning rate, and batch size
- Training schedules and epochs
- Evaluation metrics and intervals

---

## Performance Metrics

MAGMA v2 supports multiple metrics for evaluating LLM reasoning:

- **Exact Match Accuracy**  
  Checks if the final output exactly matches the expected solution (primary metric in the paper).
- **F1 Score**  
  Measures partial correctness of the final output, giving partial credit.

For each of the above, we also track:

- **Intermediate Steps Accuracy**  
  Evaluates correctness of step-by-step reasoning.
- **Final Step Accuracy**  
  Evaluates correctness solely on the concluding step.
- **Trajectory Accuracy**  
  Evaluates correctness over the entire chain-of-thought (all steps + final).
- **Independent Accuracy**  
  Evaluates performance of each inference step independently, ignoring the chain-of-thought.

---

## Contributing

We welcome contributions and improvements:

1. **Fork** the repository.
2. **Create a new branch** (`git checkout -b feature-branch`).
3. **Commit your changes** (`git commit -am 'Add new feature'`).
4. **Push** to your branch (`git push origin feature-branch`).
5. Open a **Pull Request**.

Check the `data_generation/` folder for more details on generating tasks, and please run `pytest` to verify correctness before submitting.

---

## Reproducibility

- **Seed:** 100898 (used for consistent graph generation and model initialization)
- **BFS Llama3 hyperparameters**: `_r_` and `_alpha_` = 8
- Other baseline data generation and training use default settings.

---

## License

This project is licensed under the [MIT License](LICENSE). © 2025 Alexander Taylor

---

## Acknowledgements

- **Data** partially adapted from the [CLRS benchmark](https://github.com/google-deepmind/clrs).
- **Model training** adapted from the [Hugging Face Alignment Handbook](https://github.com/huggingface/alignment-handbook.git).

---

## Contact Information

For questions, issues, or feedback, please open an issue or reach out via email at `ataylor2@cs.ucla.edu`.

---

Thank you for using **MAGMA v2**! We hope this benchmark furthers the understanding and capabilities of large language models in structured, algorithmic reasoning tasks.

