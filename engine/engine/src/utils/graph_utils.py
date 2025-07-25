from collections import defaultdict

class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        """Finds the root of x with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        """Unites the sets containing x and y using union by rank."""
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1

    def add(self, x):
        """Adds a new element as its own set."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def nodes(self):
        """Returns all nodes in the structure."""
        return list(self.parent.keys())

def ensure_transitivity(pairs):
    """
    Ensures transitivity by merging equal elements, constructing a directed graph,
    detecting cycles, and removing minimal conflicting nodes.
    """
    # Step 1: Build equivalence classes using Union-Find
    uf = UnionFind()
    for a, b, rank in pairs:
        uf.add(a)
        uf.add(b)
        if rank == 0:
            uf.union(a, b)
    
    # Create a mapping from each node to its representative
    rep = { node: uf.find(node) for node in uf.nodes() }

    # Step 2: Build the directed graph for strict relations
    graph = defaultdict(set)
    conflict_counts = defaultdict(int)

    for a, b, rank in pairs:
        A, B = rep[a], rep[b]
        if rank in (1, -1):
            if A == B:
                # Track conflicting nodes (those causing an inconsistency)
                conflict_counts[a] += 1
                conflict_counts[b] += 1
                continue  # Skip adding this conflicting strict relation

            if rank == 1:
                graph[A].add(B)
            else:
                graph[B].add(A)

    # Step 3: Remove minimal number of conflicting nodes
    removed_nodes = remove_minimum_conflicting_nodes(graph, conflict_counts)

    # Step 4: Detect and prune cycles. |= in-place OR operator for sets.
    removed_nodes |= prune_cycles(graph)

    return {
        "status": "consistent after pruning",
        "removed_nodes": removed_nodes,
        "final_graph": graph,
        "union_find": uf
    }

def remove_minimum_conflicting_nodes(graph, conflict_counts):
    """
    Removes the minimal number of nodes to resolve contradictions.
    Prioritizes keeping as many relations as possible.
    """
    removed_nodes = set()

    while conflict_counts:
        # Pick the node with the **highest number of conflicts** and **fewest strict relations**
        node_to_remove = min(conflict_counts.keys(), key=lambda x: (conflict_counts[x], len(graph[x])))

        # Remove node and update affected conflicts
        removed_nodes.add(node_to_remove)
        del conflict_counts[node_to_remove]
        graph.pop(node_to_remove, None)

        for n in list(graph.keys()):
            if node_to_remove in graph[n]:
                graph[n].discard(node_to_remove)

    return removed_nodes

def prune_cycles(graph):
    """ Removes nodes to break cycles in the directed graph """
    def find_cycle():
        visited = set()
        stack = set()
        path = []
        
        def visit(node):
            if node in stack:
                path.append(node)
                return node  # Cycle detected, return starting node
            if node in visited:
                return None
            visited.add(node)
            stack.add(node)
            path.append(node)
            for neighbor in graph[node]:
                cycle_start = visit(neighbor)
                if cycle_start is not None:
                    return cycle_start
            stack.remove(node)
            path.pop()
            return None

        for node in list(graph.keys()):
            cycle_start = visit(node)
            if cycle_start:
                return path[path.index(cycle_start):]  # Return full cycle path
        return None

    removed_nodes = set()

    while True:
        cycle = find_cycle()
        if not cycle:
            break  # No more cycles, exit

        # Pick the node with the **fewest outgoing edges** to remove the least relations
        node_to_remove = min(cycle, key=lambda x: len(graph[x]))
        removed_nodes.add(node_to_remove)

        # Remove the node from the graph
        graph.pop(node_to_remove, None)
        for n in graph:
            graph[n].discard(node_to_remove)

    return removed_nodes

def reconstruct_pairs(graph, removed_nodes, uf):
    """ Reconstructs the modified set of ranked pairs after pruning cycles and conflicts. """
    pairs = set()

    # Step 1: Restore strict order relations
    for a in graph:
        for b in graph[a]:
            pairs.add((a, b, 1))  # a > b

    # Step 2: Restore equivalence relations
    class_map = {}
    for node in uf.nodes():
        rep = uf.find(node)
        if rep not in class_map:
            class_map[rep] = []
        class_map[rep].append(node)

    for rep, group in class_map.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                pairs.add((group[i], group[j], 0))  # a = b

    return pairs

def run_tests():
    test_cases = [
        {
            "name": "Basic Transitive Relations (No Issues)",
            "pairs": [
                ('a', 'b', 1),
                ('b', 'c', 1),
                ('a', 'c', 1),  # Already transitive
                ('d', 'e', 0),
                ('e', 'f', 1)
            ]
        },
        {
            "name": "Simple Cycle Removal",
            "pairs": [
                ('x', 'y', 1),
                ('y', 'z', 1),
                ('z', 'x', 1)  # Cycle x > y > z > x
            ]
        },
        {
            "name": "Conflicting Equality with Strict Order",
            "pairs": [
                ('p', 'q', 0),  # p == q
                ('q', 'r', 1),  # q > r
                ('r', 'p', 1)   # r > p (conflict!)
            ]
        },
        {
            "name": "Complex Graph with Mixed Issues",
            "pairs": [
                ('a', 'b', 1),  # a > b
                ('b', 'c', 1),  # b > c
                ('c', 'd', 1),  # c > d
                ('d', 'a', 1),  # d > a (Cycle)
                ('e', 'f', 0),  # e == f
                ('f', 'g', 1),  # f > g
                ('g', 'h', -1), # g < h (conflict with f > g)
                ('i', 'j', 1),
                ('j', 'k', 1),
                ('k', 'i', 1)  # Cycle
            ]
        },
        {
            "name": "Large-Scale Test",
            "pairs": [
                ('n1', 'n2', 1), ('n2', 'n3', 1), ('n3', 'n4', 1),
                ('n4', 'n5', 1), ('n5', 'n6', 1), ('n6', 'n7', 1),
                ('n7', 'n8', 1), ('n8', 'n9', 1), ('n9', 'n10', 1),
                ('n10', 'n1', 1),  # Large cycle
                ('m1', 'm2', 0), ('m2', 'm3', 0), ('m3', 'm1', 0),  # Valid equivalences
                ('m3', 'n5', 1),  # Conflict
                ('o1', 'o2', 1), ('o2', 'o3', 1), ('o3', 'o4', 1),
                ('o4', 'o5', 1), ('o5', 'o1', 1)  # Another cycle
            ]
        }
    ]

    for test in test_cases:
        print(f"\n--- Running Test: {test['name']} ---\n")
        result = ensure_transitivity(test["pairs"])
        
        if result["status"] == "consistent after pruning":
            updated_pairs = reconstruct_pairs(result["final_graph"], result["removed_nodes"], result["union_find"])
            print("Updated Pairs:", updated_pairs)
            print("Removed Nodes:", result["removed_nodes"])
        else:
            print("Error:", result["message"])
        
        print("\n" + "="*50)


if __name__ == "__main__":  
    # Run the test suite
    run_tests()
