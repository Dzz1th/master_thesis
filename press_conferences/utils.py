from aiolimiter import AsyncLimiter
import asyncio
import time
from collections import defaultdict

from sklearn.model_selection import StratifiedKFold

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
        
def stratified_group_kfold(X, y, group_labels, n_splits=5, shuffle=True, random_state=42):
    """
        Ensure that each group is represented in each fold in the same proportion. 
    """

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    for train_group_idx, test_group_idx in skf.split(X, group_labels):
        X_train, X_test = X[train_group_idx], X[test_group_idx]
        y_train, y_test = y[train_group_idx], y[test_group_idx] 

        yield  X_train, y_train, X_test, y_test 



def add_edge(graph, u, v):
    """Add a directed edge from u to v."""
    graph[u].append(v)

def find_all_cycles_util(v, visited, rec_stack, path, graph, all_cycles):
    """Utilize DFS to find all cycles."""
    visited[v] = True
    rec_stack[v] = True
    path.append(v)

    for neighbor in graph[v]:
        if not visited[neighbor]:
            find_all_cycles_util(neighbor, visited, rec_stack, path, graph, all_cycles)
        elif rec_stack[neighbor]:
            # Cycle detected, extract the cycle
            cycle_start_index = path.index(neighbor)
            cycle = path[cycle_start_index:]
            all_cycles.append(cycle)

    # Backtrack
    rec_stack[v] = False
    path.pop()

def find_all_cycles(graph, num_nodes):
    """Find all cycles in the graph."""
    visited = [False] * num_nodes
    rec_stack = [False] * num_nodes
    all_cycles = []

    for node in range(num_nodes):
        if not visited[node]:
            find_all_cycles_util(node, visited, rec_stack, [], graph, all_cycles)

    return all_cycles

def create_graph_from_pairs(pairs, rank):
    idx = 0
    text_to_index = {}
    for (u, v) in pairs:
        if u not in text_to_index:
            text_to_index[u] = idx
            idx += 1
        if v not in text_to_index:
            text_to_index[v] = idx
            idx += 1

    sorted_index_pairs = []
    for i, (u, v) in enumerate(pairs):
        if rank[i] == 1:
            sorted_index_pairs.append((text_to_index[u], text_to_index[v]))
        else:
            sorted_index_pairs.append((text_to_index[v], text_to_index[u]))

    graph = defaultdict(list)
    for (u, v) in sorted_index_pairs:
        add_edge(graph, u, v)

    return graph, len(text_to_index), text_to_index

def remove_cycles(cycles):
    removed_idxs = []

    def iteration(cycles):
        counter = defaultdict(int)
        for cycle in cycles:
            for el in cycle:
                counter[el] += 1

        max_el = max(counter, key=counter.get)
        cycles = [cycle for cycle in cycles if max_el not in cycle]
        return cycles, max_el
    
    while len(cycles) > 0:
        cycles, max_el = iteration(cycles)
        removed_idxs.append(max_el)

    return removed_idxs

if __name__ == "__main__":
    # Example usage
    # Let's say we have 4 documents and the following pairwise rankings:
    # doc0 < doc1, doc1 < doc2, doc2 < doc0 (this creates a cycle)
    cycles = [[69, 36, 18, 13, 12], [18, 13, 74, 29, 7, 21], [2, 69, 36, 18, 13, 74, 29, 7, 21], [40, 16, 11], [2, 69, 36, 18, 13, 74, 29, 64], [2, 69, 36, 18, 13, 74, 29, 31], [74, 29, 31, 10], [2, 69, 36, 18, 13, 74, 29, 31, 32, 52, 6, 3], [1, 2, 69, 36, 18, 13, 74, 29, 31, 32, 52, 6, 3], [74, 29, 31, 32, 52, 6, 3], [31, 32, 52, 6, 3], [29, 31, 32, 52, 56], [2, 69, 36, 18, 13, 74, 29, 31, 32, 52, 56], [13, 74, 29, 31, 32, 52, 56], [31, 32, 52], [13, 74, 29, 31, 32], [2, 71, 57], [1, 2, 71, 35], [71, 35, 37, 45, 61], [71, 35, 37, 45], [1, 2, 71, 35, 51, 34], [71, 35, 51, 34, 8, 53, 70], [9, 48, 60]]

    remove_idxs = remove_cycles(cycles)
    print(remove_idxs)