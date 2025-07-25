"""
Utility functions for sentiment analysis pipeline
"""
import asyncio
from aiolimiter import AsyncLimiter
import pickle
import os
import time
import logging
import numpy as np
from typing import List, Dict, Tuple, Set, Any, Optional
from collections import defaultdict, deque
import networkx as nx
from sklearn.model_selection import StratifiedGroupKFold



logger = logging.getLogger(__name__)

class TokenRateLimiter:
    def __init__(self, max_requests_per_sec: int, max_tokens_per_min: int):
        """
        Initialize the rate limiter.
        
        Args:
            max_requests_per_sec: Maximum requests per second.
            max_tokens_per_min: Maximum tokens allowed per minute.
        """
        self.request_limiter = AsyncLimiter(max_requests_per_sec, time_period=1)
        self.max_tokens_per_min = max_tokens_per_min
        self.tokens_used = 0
        self.lock = asyncio.Lock()
        self.last_reset = time.time()

    async def wait_for_token_availability(self, tokens_needed: int):
        """
        Wait until tokens are available.
        
        Args:
            tokens_needed: The number of tokens required for this request.
        """
        while True:
            async with self.lock:
                current_time = time.time()
                # Reset token count if a minute has passed
                if current_time - self.last_reset >= 60:
                    self.tokens_used = 0
                    self.last_reset = current_time
                
                # Check if tokens are available
                if self.tokens_used + tokens_needed <= self.max_tokens_per_min:
                    self.tokens_used += tokens_needed
                    break
            
            # Sleep briefly before re-checking
            await asyncio.sleep(0.1)

    async def run_task(self, func, tokens_needed: int, *args, **kwargs):
        """
        Run a task while respecting the rate limits.
        
        Args:
            func: The coroutine function to run.
            tokens_needed: The number of tokens needed for this request.
        """
        async with self.request_limiter:
            await self.wait_for_token_availability(tokens_needed)
            return await func(*args, **kwargs)


def save_embeddings(embeddings: np.ndarray, filepath: str):
    """Save embeddings to disk
    
    Args:
        embeddings: Embeddings array
        filepath: Path to save embeddings
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(embeddings, f)
    logger.info(f"Saved embeddings to {filepath}")


def load_embeddings(filepath: str) -> Optional[np.ndarray]:
    """Load embeddings from disk
    
    Args:
        filepath: Path to load embeddings from
        
    Returns:
        Embeddings array if file exists, None otherwise
    """
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            embeddings = pickle.load(f)
        logger.info(f"Loaded embeddings from {filepath}")
        return embeddings
    return None

def build_graphs(self, pairs: List[Tuple[str, str]], rankings: List[int]) -> Tuple[nx.DiGraph, nx.Graph]:
        """Build directed graph and equality graph from pairs and rankings

            Directed graph corresponds to the ranking relations, while
            equality graph corresponds to the equality classes.
        
        Args:
            pairs: List of document pairs
            rankings: List of rankings
            
        Returns:
            Tuple of (directed graph, equality graph)
        """
        # Create mapping from text to index
        all_texts = set()
        for doc1, doc2 in pairs:
            all_texts.add(doc1)
            all_texts.add(doc2)
        
        self.text_to_idx = {text: i for i, text in enumerate(sorted(all_texts))}
        self.idx_to_text = {i: text for text, i in self.text_to_idx.items()}
        
        # Create directed graph for ranking relations
        G_directed = nx.DiGraph()
        
        # Create undirected graph for equality relations
        G_equality = nx.Graph()
        
        # Add all nodes to both graphs
        for text in all_texts:
            node_id = self.text_to_idx[text]
            G_directed.add_node(node_id)
            G_equality.add_node(node_id)
        
        # Add edges based on rankings
        for (doc1, doc2), rank in zip(pairs, rankings):
            idx1, idx2 = self.text_to_idx[doc1], self.text_to_idx[doc2]
            
            if rank == 1:  # doc1 > doc2
                G_directed.add_edge(idx1, idx2, weight=1, rank=rank)
            elif rank == -1:  # doc1 < doc2
                G_directed.add_edge(idx2, idx1, weight=1, rank=-rank)
            elif rank == 0:  # doc1 == doc2, add to equality graph
                G_equality.add_edge(idx1, idx2, weight=1, rank=rank)
        
        graph = G_directed
        equality_graph = G_equality
        
        # Find connected components in equality graph to get equality classes
        equality_classes = list(nx.connected_components(G_equality))
        logger.info(f"Found {len(equality_classes)} equality classes")
        
        return graph, equality_graph






def create_graph_from_pairs(pairs: List[Tuple[str, str]], 
                           ranks: List[int]) -> Tuple[Dict[int, List[Tuple[int, int]]], int, Dict[str, int]]:
    """Create a directed graph from document pairs and rankings
    
    Args:
        pairs: List of document pairs
        ranks: List of rankings (1 if first is better, -1 if second is better)
        
    Returns:
        Tuple containing the graph as adjacency list, number of documents,
        and mapping from document text to index
    """
    # Map unique documents to indices
    text_to_index = {}
    idx = 0
    for pair in pairs:
        for doc in pair:
            if doc not in text_to_index:
                text_to_index[doc] = idx
                idx += 1
    
    # Create graph
    graph = defaultdict(list)
    for (doc1, doc2), rank in zip(pairs, ranks):
        idx1, idx2 = text_to_index[doc1], text_to_index[doc2]
        ## Ranks are [1, -1, 0]. If rank is 0, we skip the edge because it doesn't 
        if rank == 1:
            graph[idx1].append((idx2, 1))  # First doc is better
            graph[idx2].append((idx1, -1))  # Second doc is worse
        elif rank == -1:
            graph[idx2].append((idx1, 1))  # Second doc is better
            graph[idx1].append((idx2, -1))  # First doc is worse
    
    return graph, idx, text_to_index


def find_all_cycles(graph: Dict[int, List[Tuple[int, int]]], num_docs: int) -> List[List[int]]:
    """Find all cycles in the graph
    
    Args:
        graph: Graph as adjacency list
        num_docs: Number of documents
        
    Returns:
        List of cycles, where each cycle is a list of document indices
    """
    # Use NetworkX for cycle detection
    G = nx.DiGraph()
    
    # Add all nodes to ensure isolated nodes are included
    for i in range(num_docs):
        G.add_node(i)
    
    # Add edges
    for node, edges in graph.items():
        for target, _ in edges:
            if target > node:  # Only add each edge once
                G.add_edge(node, target)
    
    # Find cycles
    cycles = list(nx.simple_cycles(G))
    
    return cycles


def remove_cycles(cycles: List[List[int]]) -> Set[int]:
    """Identify nodes to remove to break cycles
    
    Args:
        cycles: List of cycles
        
    Returns:
        Set of nodes to remove
    """
    if not cycles:
        return set()
    
    # Count node frequency in cycles
    node_count = defaultdict(int)
    for cycle in cycles:
        for node in cycle:
            node_count[node] += 1
    
    # Find nodes that appear in most cycles
    nodes_to_remove = set()
    while cycles:
        # Find node that appears in most cycles
        best_node = max(node_count, key=node_count.get)
        nodes_to_remove.add(best_node)
        
        # Remove cycles containing this node
        new_cycles = []
        for cycle in cycles:
            if best_node not in cycle:
                new_cycles.append(cycle)
            else:
                # Decrement count for other nodes in this cycle
                for node in cycle:
                    if node != best_node:
                        node_count[node] -= 1
        
        cycles = new_cycles
        if best_node in node_count:
            del node_count[best_node]
    
    return nodes_to_remove


def prune_cycles(pairs: List[Tuple[str, str]], ranks: List[int]) -> Tuple[List[Tuple[str, str]], List[int]]:
    """Prune document pairs to remove cycles
    
    Args:
        pairs: List of document pairs
        ranks: List of rankings
        
    Returns:
        Tuple with pruned pairs and corresponding ranks
    """
    graph, num_docs, text_to_index = create_graph_from_pairs(pairs, ranks)
    cycles = find_all_cycles(graph, num_docs)
    
    if not cycles:
        return pairs, ranks
    
    logger.info(f"Found {len(cycles)} cycles among {num_docs} documents")
    
    remove_idxs = remove_cycles(cycles)
    logger.info(f"Removing {len(remove_idxs)} documents to break cycles")
    
    # Create pruned pairs and ranks
    new_pairs = []
    new_ranks = []
    for i, (pair, rank) in enumerate(zip(pairs, ranks)):
        doc1, doc2 = pair
        idx1, idx2 = text_to_index[doc1], text_to_index[doc2]
        if idx1 not in remove_idxs and idx2 not in remove_idxs:
            new_pairs.append(pair)
            new_ranks.append(rank)
    
    logger.info(f"Pruned from {len(pairs)} to {len(new_pairs)} pairs")
    return new_pairs, new_ranks


def stratified_group_kfold(X, y, groups, n_splits=5):
    """Custom implementation of StratifiedGroupKFold for older scikit-learn versions
    
    Args:
        X: Features
        y: Target variable
        groups: Group labels
        n_splits: Number of splits
        
    Returns:
        List of train/test indices for each fold
    """
    try:
        # Try to use sklearn's implementation first
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42).split(X, y, groups)
    except:
        # Fall back to custom implementation
        labels_num = np.max(y) + 1
        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
        for label, group in zip(y, groups):
            y_counts_per_group[group][label] += 1
            
        y_counts = np.zeros(labels_num)
        for group in sorted(y_counts_per_group):
            y_counts += y_counts_per_group[group]
            
        # Compute desired number of samples from each class in each fold
        fold_target = y_counts / n_splits
        
        # Split groups into folds
        groups_unique = list(set(groups))
        fold_groups = [[] for _ in range(n_splits)]
        fold_y_counts = [np.zeros(labels_num) for _ in range(n_splits)]
        
        # Sort groups by size for better balancing
        groups_and_counts = [(g, y_counts_per_group[g]) for g in groups_unique]
        groups_and_counts.sort(key=lambda x: -np.sum(x[1]))
        
        # Assign groups to folds
        for g, g_y_counts in groups_and_counts:
            # Find fold with most capacity for these samples
            best_fold = np.argmin([
                np.sum(np.maximum(0, fold_y + g_y_counts - fold_target))
                for fold_y in fold_y_counts
            ])
            fold_groups[best_fold].append(g)
            fold_y_counts[best_fold] += g_y_counts
            
        # Create indices for each fold
        all_indices = np.arange(len(y))
        fold_indices = []
        for fold_g in fold_groups:
            fold_idx = np.array([
                i for i, g in enumerate(groups) if g in fold_g
            ])
            fold_indices.append(fold_idx)
            
        # Generate train/test splits
        for i in range(n_splits):
            test_idx = fold_indices[i]
            train_idx = np.concatenate([
                fold_indices[j] for j in range(n_splits) if j != i
            ])
            yield train_idx, test_idx