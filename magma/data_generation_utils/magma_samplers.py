# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Sampling utilities."""

import abc
import collections
import inspect
import types

from typing import Any, Callable, List, Optional, Tuple
from absl import logging

from . import clrs_modules
from .clrs_modules import probing
from .clrs_modules import specs
import jax
import numpy as np
import networkx as nx
import random

from torch_geometric.datasets import Planetoid, PPI
from torch_geometric.utils import to_networkx 
from Graph_Sampling import SRW_RWF_ISRW, MHRW, TIES
from globals import SAMPLE_SPACE_SIZE

_Array = np.ndarray
_DataPoint = probing.DataPoint
Trajectory = List[_DataPoint]
Trajectories = List[Trajectory]


Algorithm = Callable[..., Any]
Features = collections.namedtuple('Features', ['inputs', 'hints', 'lengths'])
FeaturesChunked = collections.namedtuple(
    'Features', ['inputs', 'hints', 'is_first', 'is_last'])
Feedback = collections.namedtuple('Feedback', ['features', 'outputs'])

##############################
# Load Real-World Graph Datasets
##############################

# 1. Cora (small citation network)
cora_dataset = Planetoid(root='/tmp/Cora', name='Cora')
cora_data = cora_dataset[0]
G_cora = to_networkx(cora_data)
print("Cora - Nodes:", G_cora.number_of_nodes(), "Edges:", G_cora.number_of_edges())

# 2. CiteSeer (small citation network)
citeseer_dataset = Planetoid(root='/tmp/CiteSeer', name='CiteSeer')
citeseer_data = citeseer_dataset[0]
G_citeseer = to_networkx(citeseer_data)
print("CiteSeer - Nodes:", G_citeseer.number_of_nodes(), "Edges:", G_citeseer.number_of_edges())

# 3. PubMed (small-medium citation network)
pubmed_dataset = Planetoid(root='/tmp/PubMed', name='PubMed')
pubmed_data = pubmed_dataset[0]
G_pubmed = to_networkx(pubmed_data)
print("PubMed - Nodes:", G_pubmed.number_of_nodes(), "Edges:", G_pubmed.number_of_edges())

# 4. PPI (collection of graphs; using the first graph)
ppi_dataset = PPI(root='/tmp/PPI', split='train')
ppi_data = ppi_dataset[0]
G_ppi = to_networkx(ppi_data)
print("PPI - Nodes:", G_ppi.number_of_nodes(), "Edges:", G_ppi.number_of_edges())

datasets = {
    'Cora': G_cora,
    'CiteSeer': G_citeseer,
    'PubMed': G_pubmed,
    'PPI': G_ppi,
}

class Sampler(abc.ABC):
  """Sampler abstract base class."""

  def __init__(
      self,
      algorithm: Algorithm,
      spec: specs.Spec,
      num_samples: int,
      *args,
      seed: Optional[int] = None,
      **kwargs,
  ):
    """Initializes a `Sampler`.

    Args:
      algorithm: The algorithm to sample from
      spec: The algorithm spec.
      num_samples: Number of algorithm unrolls to sample. If positive, all the
        samples will be generated in the constructor, and at each call
        of the `next` method a batch will be randomly selected among them.
        If -1, samples are generated on the fly with each call to `next`.
      *args: Algorithm args.
      seed: RNG seed.
      **kwargs: Algorithm kwargs.
    """

    # Use `RandomState` to ensure deterministic sampling across Numpy versions.
    self._rng = np.random.RandomState(seed)
    self._spec = spec
    self._num_samples = num_samples
    self._algorithm = algorithm
    self._args = args
    self._kwargs = kwargs

    if num_samples < 0:
      logging.warning('Sampling dataset on-the-fly, unlimited samples.')
      # Just get an initial estimate of max hint length
      self.max_steps = -1
      for _ in range(10):
        data = self._sample_data(*args, **kwargs)
        _, probes = algorithm(*data)
        _, _, hint = probing.split_stages(probes, spec)
        for dp in hint:
          assert dp.data.shape[1] == 1  # batching axis
          if dp.data.shape[0] > self.max_steps:
            self.max_steps = dp.data.shape[0]
    else:
      logging.info('Creating a dataset with %i samples.', num_samples)
      (self._inputs, self._outputs, self._hints,
       self._lengths) = self._make_batch(num_samples, spec, 0, algorithm, *args,
                                         **kwargs)

  def _make_batch(self, num_samples: int, spec: specs.Spec, min_length: int,
                  algorithm: Algorithm, *args, **kwargs):
    """Generate a batch of data."""
    inputs = []
    outputs = []
    hints = []

    for _ in range(num_samples):
      data = self._sample_data(*args, **kwargs)
      _, probes = algorithm(*data)
      inp, outp, hint = probing.split_stages(probes, spec)
      inputs.append(inp)
      outputs.append(outp)
      hints.append(hint)
      if len(hints) % 1000 == 0:
        logging.info('%i samples created', len(hints))

    # Batch and pad trajectories to max(T).
    inputs = _batch_io(inputs)
    outputs = _batch_io(outputs)
    hints, lengths = _batch_hints(hints, min_length)
    return inputs, outputs, hints, lengths

  def next(self, batch_size: Optional[int] = None) -> Feedback:
    """Subsamples trajectories from the pre-generated dataset.

    Args:
      batch_size: Optional batch size. If `None`, returns entire dataset.

    Returns:
      Subsampled trajectories.
    """
    if batch_size:
      if self._num_samples < 0:  # generate on the fly
        inputs, outputs, hints, lengths = self._make_batch(
            batch_size, self._spec, self.max_steps,
            self._algorithm, *self._args, **self._kwargs)
        if hints[0].data.shape[0] > self.max_steps:
          logging.warning('Increasing hint length from %i to %i',
                          self.max_steps, hints[0].data.shape[0])
          self.max_steps = hints[0].data.shape[0]
      else:
        if batch_size > self._num_samples:
          raise ValueError(
              f'Batch size {batch_size} > dataset size {self._num_samples}.')

        # Returns a fixed-size random batch.
        indices = self._rng.choice(self._num_samples, (batch_size,),
                                   replace=True)
        inputs = _subsample_data(self._inputs, indices, axis=0)
        outputs = _subsample_data(self._outputs, indices, axis=0)
        hints = _subsample_data(self._hints, indices, axis=1)
        lengths = self._lengths[indices]

    else:
      # Returns the full dataset.
      assert self._num_samples >= 0
      inputs = self._inputs
      hints = self._hints
      lengths = self._lengths
      outputs = self._outputs

    return Feedback(Features(inputs, hints, lengths), outputs)

  @abc.abstractmethod
  def _sample_data(self, length: int, *args, **kwargs) -> List[_Array]:
    pass

  def _random_sequence(self, length, low=0.0, high=1.0):
    """Random sequence."""
    return self._rng.uniform(low=low, high=high, size=(length,))

  def _random_string(self, length, chars=4):
    """Random string."""
    return self._rng.randint(0, high=chars, size=(length,))

  def _random_er_graph(self, nb_nodes, p=0.5, directed=False, acyclic=False,
                       weighted=False, low=0, high=10, integer_based=True, self_edges_weighted=False):
    """Random Erdos-Renyi graph."""

    mat = self._rng.binomial(1, p, size=(nb_nodes, nb_nodes))
    if not directed:
      mat *= np.transpose(mat)
    elif acyclic:
      mat = np.triu(mat, k=1)
      p = self._rng.permutation(nb_nodes)  # To allow nontrivial solutions
      mat = mat[p, :][:, p]
    if weighted:
      weights = self._rng.random_integers(low=low, high=high, size=(nb_nodes, nb_nodes))
      
      if not directed:
        if not integer_based:
          weights *= np.transpose(weights)
          weights = np.sqrt(weights + 1e-3)  # Add epsilon to protect underflow
        else:
          weights = np.maximum(weights, weights.T)
      
      if not self_edges_weighted:
        matrix_size = weights.shape[0]
        matrix = np.ones((matrix_size, matrix_size)) - np.eye(matrix_size)
        weights = matrix * weights
        
      mat = mat.astype(int) * weights

    return mat

  def _random_community_graph(self, nb_nodes, k=4, p=0.5, eps=0.01,
                              directed=False, acyclic=False, weighted=False,
                              low=0.0, high=1.0):
    """Random perturbed k-community graph."""
    mat = np.zeros((nb_nodes, nb_nodes))
    if k > nb_nodes:
      raise ValueError(f'Cannot generate graph of too many ({k}) communities.')
    los, his = [], []
    lo = 0
    for i in range(k):
      if i == k - 1:
        hi = nb_nodes
      else:
        hi = lo + nb_nodes // k
      mat[lo:hi, lo:hi] = self._random_er_graph(
          hi - lo, p=p, directed=directed,
          acyclic=acyclic, weighted=weighted,
          low=low, high=high)
      los.append(lo)
      his.append(hi)
      lo = hi
    toggle = self._random_er_graph(nb_nodes, p=eps, directed=directed,
                                   acyclic=acyclic, weighted=weighted,
                                   low=low, high=high)

    # Prohibit closing new cycles
    for i in range(k):
      for j in range(i):
        toggle[los[i]:his[i], los[j]:his[j]] *= 0

    mat = np.where(toggle > 0.0, (1.0 - (mat > 0.0)) * toggle, mat)
    p = self._rng.permutation(nb_nodes)  # To allow nontrivial solutions
    mat = mat[p, :][:, p]
    return mat

  def _random_bipartite_graph(self, n, m, p=0.25):
    """Random bipartite graph-based flow network."""
    nb_nodes = n + m + 2
    s = 0
    t = n + m + 1
    mat = np.zeros((nb_nodes, nb_nodes))
    mat[s, 1:n+1] = 1.0  # supersource
    mat[n+1:n+m+1, t] = 1.0  # supersink
    mat[1:n+1, n+1:n+m+1] = self._rng.binomial(1, p, size=(n, m))
    return mat
  
  def _random_barabasi_albert_graph(self, n, m=2):
    if n <= m:
        m = max(1, n - 1)
    return nx.barabasi_albert_graph(n, m)

  def _random_watts_strogatz_graph(self, n, k=4, p=0.3):
      if k >= n:
          k = n - 1
      return nx.watts_strogatz_graph(n, k, p)

  def _random_stochastic_block_model_graph(self, n, num_blocks=2, intra_p=0.5, inter_p=0.2):
      sizes = [n // num_blocks] * num_blocks
      remainder = n % num_blocks
      for i in range(remainder):
          sizes[i] += 1
      p_matrix = [[intra_p if i == j else inter_p for j in range(num_blocks)] for i in range(num_blocks)]
      return nx.stochastic_block_model(sizes, p_matrix)
  
  def _ensure_edge_weights(self, G, weight_range=(0, 1)):
    """Ensure edge weights are present and within a given range."""
    for u, v, data in G.edges(data=True):
      if 'weight' not in data:
        data['weight'] = random.uniform(*weight_range)
    return G
    
  def _random_real_world_graph(self, name, G, n, method, weighted=False, weight_range=(0,10), p=0.5, k=3):
    """Random real-world graph."""
    def sample_subgraph(G, n, method='random_node', **kwargs):
      """
      Sample an n-sized subgraph from graph G using one of several methods.
      """
      if n > G.number_of_nodes():
          raise ValueError("n must be less than or equal to the total number of nodes in G.")
      
      if method == "rwss":
        sampler = SRW_RWF_ISRW.SRW_RWF_ISRW()
        return sampler.random_walk_sampling_simple(G, n)
      elif method == "rwswfb":
        sampler = SRW_RWF_ISRW.SRW_RWF_ISRW()
        return sampler.random_walk_sampling_with_fly_back(G, n, sampler.fly_back_prob)
      elif method == "rwigs":
        sampler = SRW_RWF_ISRW.SRW_RWF_ISRW()
        return sampler.random_walk_induced_graph_sampling(G, n)
      elif method == "mhrw":
        seed = kwargs.get("seed")
        sampler = MHRW.MHRW()
        return sampler.mhrw(G, seed, n)
      elif method == "ties":
        sampler = TIES.TIES()
        return sampler.ties(G, n, n / len(G.nodes()))
      else:
          raise ValueError("Invalid sampling method.")
        
    G = G.to_undirected() if isinstance(G, nx.DiGraph) else G
    
    # Ensure we are not passing an empty graph
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        raise ValueError(f"[ERROR] Input graph {name} is empty!")
    
    subG = sample_subgraph(G, n, method=method, p=p, k=k, seed=next(iter(G.nodes())))
    if subG.number_of_nodes() < n:
        raise ValueError(f"[ERROR] Sampling method {method} failed to sample {n} nodes from {name}. The subgraph has only {subG.number_of_nodes()} nodes.")
    subG = nx.to_numpy_array(subG)
    label = f"RealWorld_{name}_{method}_{n}"
    if weighted:
      subG = self._ensure_edge_weights(subG, weight_range=weight_range)
    return subG
         
  def _random_synthetic_graph(self, n, method, weighted=False, weight_range=(0,10), p=0.3):
    """Random synthetic graph."""
    def sample_subgraph(n, method='random_node', **kwargs):
      """
      Sample an n-sized subgraph from graph G using one of several methods.
      """
      if method == 'ER':
          G_syn = self._random_er_graph(n, p=p)
      elif method == 'BA':
          G_syn = self._random_barabasi_albert_graph(n, m=2)
      elif method == 'WS':
          G_syn = self._random_watts_strogatz_graph(n, k=4, p=0.3)
      elif method == 'SBM':
          G_syn = self._random_stochastic_block_model_graph(n, num_blocks=2, intra_p=0.5, inter_p=0.2)
      return G_syn
    G_syn = sample_subgraph(n, method=method, p=p)
    
    if method != 'ER':
      G_syn = nx.to_numpy_array(G_syn.to_undirected())
    
    if weighted:
      G_syn = self._ensure_edge_weights(G_syn, weight_range=weight_range)
    label = f"Synthetic_{method}_{n}"
    return G_syn

  def _select_diverse_graphs(candidate_list, k=3):
    """
    Given a list of candidates [(label, graph), ...], greedily select k graphs
    that are as diverse as possible based on GED. The algorithm starts with an 
    arbitrary candidate and then iteratively picks the candidate that maximizes 
    the minimum GED to those already selected.
    """
    def compute_ged(G1, G2, timeout=1.0):
      """
      Compute approximate graph edit distance (GED) between G1 and G2.
      If GED cannot be computed in time, return a large value.
      """
      try:
        ged = nx.graph_edit_distance(G1, G2, timeout=timeout)
        if ged is None:
          return float('inf')
        return ged
      except Exception as e:
        return float('inf')
      
    if not candidate_list:
      return []
    selected = [candidate_list[0]]
    remaining = candidate_list[1:]
    while len(selected) < k and remaining:
      best_candidate = None
      best_max_ged = -1
      for cand in remaining:
        # Compute the minimum GED from this candidate to all already selected graphs.
        max_ged = max(compute_ged(cand[1], sel[1]) for sel in selected)
        if max_ged > best_max_ged:
          best_max_ged = max_ged
          best_candidate = cand
        if best_candidate is None:
          break
        selected.append(best_candidate)
        remaining.remove(best_candidate)
    return selected
  
  def _sample_mixed_graph(
    self,                     
    length: int,          
    weighted=False,
    weight_range=(0, 10),
    p=0.3                      # default edge probability (or other relevant synthetic param)
  ):
    """
    Sample a mix of real-world and synthetic graphs, then select the top k diverse
    candidates per sample size based on graph edit distance.
    
    Parameters:
        real_datasets (dict): Mapping of dataset name -> networkx.Graph
        real_methods (list): List of sampling methods for real graphs.
        synthetic_methods (list): List of generation models for synthetic graphs.
        sample_size (int): integer sample size.
        k (int): Number of diverse graphs to select for each size.
        weighted (bool): Whether to assign random weights to edges.
        weight_range (tuple): Range for random edge weights (min, max).
        p (float): A default parameter for, e.g., Erdos-Renyi or other random sampling.
    
    Returns:
        dict: Dictionary keyed by sample size, each value is a list of (label, G) for the selected graphs.
    """
    
    real_methods = ['rwss', 'rwswfb', 'rwigs', 'mhrw']
    synthetic_methods = ['ER', 'BA', 'WS', 'SBM']
    real_world_pairs = list(datasets.items())
    
    if length < 0:
      raise ValueError("Please provide a valid graph size.")
    
    graph_sampling_method = random.choice([True, False])
    
    if graph_sampling_method:
      name, G = random.choice(real_world_pairs)
      method = random.choice(real_methods)
      # Wrap this in a try block to handle errors gracefully
      
      sampled_G = self._random_real_world_graph(
        name=name,
        G=G,
        n=length,
        method=method,
        weighted=weighted,
        weight_range=weight_range,
        p=p   
      )
    else:
      method = random.choice(synthetic_methods)
      sampled_G = self._random_synthetic_graph(
        n=length,
        method=method,
        weighted=weighted,
        weight_range=weight_range,
        p=p  # used in e.g. Erdos-Renyi or WS
      )

    return sampled_G
      
    
def build_sampler(
    name: str,
    num_samples: int,
    *args,
    seed: Optional[int] = None,
    **kwargs,
) -> Tuple[Sampler, specs.Spec]:
  """Builds a sampler. See `Sampler` documentation."""

  if name not in specs.SPECS or name not in SAMPLERS:
    raise NotImplementedError(f'No implementation of algorithm {name}.')
  spec = specs.SPECS[name]
  algorithm = getattr(clrs_modules, name)
  sampler_class = SAMPLERS[name]
  # Ignore kwargs not accepted by the sampler.
  sampler_args = inspect.signature(sampler_class._sample_data).parameters  # pylint:disable=protected-access
  clean_kwargs = {k: kwargs[k] for k in kwargs if k in sampler_args}
  if set(clean_kwargs) != set(kwargs):
    logging.warning('Ignoring kwargs %s when building sampler class %s',
                    set(kwargs).difference(clean_kwargs), sampler_class)
  sampler = sampler_class(algorithm, spec, num_samples, seed=seed,
                          *args, **clean_kwargs)
  return sampler, spec


class DfsSampler(Sampler):
  """DFS sampler."""

  def _sample_data(
      self,
      length: int,
      mixed_data: bool = False,
      p: Tuple[float, ...] = (0.5,),
  ):
    if not mixed_data:
      graph = self._random_er_graph(
          nb_nodes=length, p=self._rng.choice(p),
          directed=False, acyclic=False, weighted=False)
    else:
      graph = self._sample_mixed_graph(length)
    return [graph]


class BfsSampler(Sampler):
  """BFS sampler."""

  def _sample_data(
      self,
      length: int,
      mixed_data: bool = False,
      p: Tuple[float, ...] = (0.5,),
  ):
    if not mixed_data:
      graph = self._random_er_graph(
          nb_nodes=length, p=self._rng.choice(p),
          directed=False, acyclic=False, weighted=False)
    else:
      graph = self._sample_mixed_graph(length)
    source_node = self._rng.choice(graph.shape[0])
    return [graph, source_node]


class MSTSampler(Sampler):
  """MST sampler for Kruskal's algorithm."""

  def _sample_data(
      self,
      length: int,
      mixed_data: bool = False,
      p: Tuple[float, ...] = (0.2,),  # lower p to account for class imbalance
      low: float = 0.,
      high: float = 1.,
  ):
    
    if not mixed_data:
      graph = self._random_er_graph(
          nb_nodes=length,
          p=self._rng.choice(p),
          directed=False,
          acyclic=False,
          weighted=True,
          low=low,
          high=high)
    else:
      graph = self._sample_mixed_graph(length)
    return [graph]


class BellmanFordSampler(Sampler):
  """Bellman-Ford sampler."""

  def _sample_data(
      self,
      length: int,
      mixed_data: bool = False,
      p: Tuple[float, ...] = (0.5,),
      low: float = 0,
      high: float = 10,
  ):
    if not mixed_data:
      graph = self._random_er_graph(
          nb_nodes=length,
          p=self._rng.choice(p),
          directed=False,
          acyclic=False,
          weighted=True,
          low=low,
          high=high)
    else:
      graph = self._sample_mixed_graph(length)
    source_node = self._rng.choice(graph.shape[0])
    return [graph, source_node]


class FloydWarshallSampler(Sampler):
  """Sampler for all-pairs shortest paths."""

  def _sample_data(
      self,
      length: int,
      mixed_data: bool = False,
      p: Tuple[float, ...] = (0.5,),
      low: float = 0,
      high: float = 10,
  ):
    if not mixed_data:
      graph = self._random_er_graph(
          nb_nodes=length,
          p=self._rng.choice(p),
          directed=False,
          acyclic=False,
          weighted=True,
          low=low,
          high=high)
    else:
      graph = self._sample_mixed_graph(length)
    return [graph]


SAMPLERS = {
    'dfs': DfsSampler,
    'bfs': BfsSampler,
    'mst_prim': BellmanFordSampler,
    'dijkstra': BellmanFordSampler,
    'floyd_warshall': FloydWarshallSampler
}

def _batch_io(traj_io: Trajectories) -> Trajectory:
  """Batches a trajectory of input/output samples along the time axis per probe.

  Args:
    traj_io:  An i/o trajectory of `DataPoint`s indexed by time then probe.

  Returns:
    A |num probes| list of `DataPoint`s with the time axis stacked into `data`.
  """

  assert traj_io  # non-empty
  for sample_io in traj_io:
    for i, dp in enumerate(sample_io):
      assert dp.data.shape[0] == 1  # batching axis
      assert traj_io[0][i].name == dp.name

  return jax.tree_util.tree_map(lambda *x: np.concatenate(x), *traj_io)


def _batch_hints(
    traj_hints: Trajectories, min_steps: int) -> Tuple[Trajectory, List[int]]:
  """Batches a trajectory of hints samples along the time axis per probe.

  Unlike i/o, hints have a variable-length time dimension. Before batching, each
  trajectory is padded to the maximum trajectory length.

  Args:
    traj_hints: A hint trajectory of `DataPoints`s indexed by time then probe
    min_steps: Hints will be padded at least to this length - if any hint is
      longer than this, the greater length will be used.

  Returns:
    A |num probes| list of `DataPoint`s with the time axis stacked into `data`,
    and a |sample| list containing the length of each trajectory.
  """

  max_steps = min_steps
  assert traj_hints  # non-empty
  for sample_hint in traj_hints:
    for dp in sample_hint:
      assert dp.data.shape[1] == 1  # batching axis
      if dp.data.shape[0] > max_steps:
        max_steps = dp.data.shape[0]
  time_and_batch = (max_steps, len(traj_hints))

  # Create zero-filled space for the batched hints, then copy each hint
  # up to the corresponding length.
  batched_traj = jax.tree_util.tree_map(
      lambda x: np.zeros(time_and_batch + x.shape[2:]),
      traj_hints[0])
  hint_lengths = np.zeros(len(traj_hints))

  for sample_idx, cur_sample in enumerate(traj_hints):
    for i in range(len(cur_sample)):
      assert batched_traj[i].name == cur_sample[i].name
      cur_data = cur_sample[i].data
      cur_length = cur_data.shape[0]
      batched_traj[i].data[:cur_length, sample_idx:sample_idx+1] = cur_data
      if i > 0:
        assert hint_lengths[sample_idx] == cur_length
      else:
        hint_lengths[sample_idx] = cur_length
  return batched_traj, hint_lengths


def _subsample_data(
    trajectory: Trajectory,
    idx: List[int],
    axis: int = 0,
) -> Trajectory:
  """New `Trajectory` where each `DataPoint`'s data is subsampled along axis."""
  sampled_traj = []
  for dp in trajectory:
    sampled_data = np.take(dp.data, idx, axis=axis)
    sampled_traj.append(
        probing.DataPoint(dp.name, dp.location, dp.type_, sampled_data))
  return sampled_traj


def _preprocess_permutations(probes, enforce_permutations):
  """Replace should-be permutations with proper permutation pointer + mask."""
  output = []
  for x in probes:
    if x.type_ != specs.Type.SHOULD_BE_PERMUTATION:
      output.append(x)
      continue
    assert x.location == specs.Location.NODE
    if enforce_permutations:
      new_x, mask = probing.predecessor_to_cyclic_predecessor_and_first(x.data)
      output.append(
          probing.DataPoint(
              name=x.name,
              location=x.location,
              type_=specs.Type.PERMUTATION_POINTER,
              data=new_x))
      output.append(
          probing.DataPoint(
              name=x.name + '_mask',
              location=x.location,
              type_=specs.Type.MASK_ONE,
              data=mask))
    else:
      output.append(probing.DataPoint(name=x.name, location=x.location,
                                      type_=specs.Type.POINTER, data=x.data))
  return output


def process_permutations(spec, sample_iterator, enforce_permutations):
  """Replace should-be permutations with proper permutation pointer + mask."""
  def _iterate():
    while True:
      feedback = next(sample_iterator)
      features = feedback.features
      inputs = _preprocess_permutations(features.inputs, enforce_permutations)
      hints = _preprocess_permutations(features.hints, enforce_permutations)
      outputs = _preprocess_permutations(feedback.outputs, enforce_permutations)
      features = features._replace(inputs=tuple(inputs),
                                   hints=tuple(hints))
      feedback = feedback._replace(features=features,
                                   outputs=outputs)
      yield feedback

  new_spec = {}
  for k in spec:
    if (spec[k][1] == specs.Location.NODE and
        spec[k][2] == specs.Type.SHOULD_BE_PERMUTATION):
      if enforce_permutations:
        new_spec[k] = (spec[k][0], spec[k][1], specs.Type.PERMUTATION_POINTER)
        new_spec[k + '_mask'] = (spec[k][0], spec[k][1], specs.Type.MASK_ONE)
      else:
        new_spec[k] = (spec[k][0], spec[k][1], specs.Type.POINTER)
    else:
      new_spec[k] = spec[k]

  return new_spec, _iterate()


def process_pred_as_input(spec, sample_iterator):
  """Move pred_h hint to pred input."""
  def _iterate():
    while True:
      feedback = next(sample_iterator)
      features = feedback.features
      pred_h = [h for h in features.hints if h.name == 'pred_h']
      if pred_h:
        assert len(pred_h) == 1
        pred_h = pred_h[0]
        hints = [h for h in features.hints if h.name != 'pred_h']
        for i in range(len(features.lengths)):
          assert np.sum(np.abs(pred_h.data[1:int(features.lengths[i]), i] -
                               pred_h.data[0, i])) == 0.0
        inputs = tuple(features.inputs) + (
            probing.DataPoint(name='pred', location=pred_h.location,
                              type_=pred_h.type_, data=pred_h.data[0]),)
        features = features._replace(inputs=tuple(inputs),
                                     hints=tuple(hints))
        feedback = feedback._replace(features=features)
      yield feedback

  new_spec = {}
  for k in spec:
    if k == 'pred_h':
      assert spec[k] == (specs.Stage.HINT, specs.Location.NODE,
                         specs.Type.POINTER)
      new_spec['pred'] = (specs.Stage.INPUT, specs.Location.NODE,
                          specs.Type.POINTER)
    else:
      new_spec[k] = spec[k]

  return new_spec, _iterate()


def process_random_pos(sample_iterator, rng):
  """Randomize the `pos` input from a sampler.

  The `pos` input is, by default, a scalar uniformly spaced between 0 and 1
  across the nodes. The exception are string algorithms (naive_string_matcher,
  kmp_string_matcher and lcs_length), where the `pos` sequence is split into
  needle and haystack (or first and second string, for lcs_length). Here
  we replace the uniformly spaced `pos` with an ordered sequence of random
  scalars, or, for string algorithms, two ordered sequences of random scalars.

  Args:
    sample_iterator: An iterator producing samples with non-random `pos` inputs.
    rng: Numpy random generator
  Returns:
    An iterator returning the samples with randomized `pos` inputs.
  """
  def _iterate():
    while True:
      feedback = next(sample_iterator)
      inputs = feedback.features.inputs
      pos, = [x for x in inputs if x.name == 'pos']
      batch_size, num_nodes = pos.data.shape
      unsorted = rng.uniform(size=(batch_size, num_nodes))
      new_pos = []
      for i in range(batch_size):  # we check one example at a time.
        # We find if there are splits in the pos sequence, marked by zeros.
        # We know there will always be at least 1 zero, if there's no split.
        split, = np.where(pos.data[i] == 0)
        split = np.concatenate([split, [num_nodes]])
        # We construct the randomized pos by sorting the random values in each
        # split and concatenating them.
        new_pos.append(
            np.concatenate([np.sort(unsorted[i, split[j]:split[j+1]])
                            for j in range(len(split) - 1)]))
      pos.data = np.array(new_pos)
      inputs = [(pos if x.name == 'pos' else x) for x in inputs]
      features = feedback.features._replace(inputs=inputs)
      feedback = feedback._replace(features=features)
      yield feedback

  return _iterate()
