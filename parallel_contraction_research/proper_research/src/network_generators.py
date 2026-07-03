import numpy as np
import networkx as nx

def generate_1d_chain(n, d_bond=16):
    """Generates a closed 1D chain tensor network (Vector-Matrix-Matrix...-Vector)."""
    tensors = []
    edges = []
    
    tensors.append(np.random.randn(d_bond))
    edges.append(['b0'])
    
    for i in range(1, n - 1):
        tensors.append(np.random.randn(d_bond, d_bond))
        edges.append([f'b{i-1}', f'b{i}'])
        
    tensors.append(np.random.randn(d_bond))
    edges.append([f'b{n-2}'])
    
    return tensors, edges

def generate_2d_grid(rows, cols, d_bond=3):
    """Generates a closed 2D grid tensor network (PEPS-like with no physical legs)."""
    tensors = []
    edges = []
    for r in range(rows):
        for c in range(cols):
            node_edges = []
            shape = []
            
            if c > 0:
                node_edges.append(f'h_{r}_{c-1}')
                shape.append(d_bond)
            if c < cols - 1:
                node_edges.append(f'h_{r}_{c}')
                shape.append(d_bond)
            if r > 0:
                node_edges.append(f'v_{r-1}_{c}')
                shape.append(d_bond)
            if r < rows - 1:
                node_edges.append(f'v_{r}_{c}')
                shape.append(d_bond)
                
            tensors.append(np.random.randn(*shape))
            edges.append(node_edges)
    return tensors, edges

def generate_3d_grid(l, w, h, d_bond=2):
    """Generates a closed 3D grid tensor network with no physical legs."""
    tensors = []
    edges = []
    for x in range(l):
        for y in range(w):
            for z in range(h):
                node_edges = []
                shape = []
                
                # X connections
                if x > 0:
                    node_edges.append(f'x_{x-1}_{y}_{z}')
                    shape.append(d_bond)
                if x < l - 1:
                    node_edges.append(f'x_{x}_{y}_{z}')
                    shape.append(d_bond)
                # Y connections
                if y > 0:
                    node_edges.append(f'y_{x}_{y-1}_{z}')
                    shape.append(d_bond)
                if y < w - 1:
                    node_edges.append(f'y_{x}_{y}_{z}')
                    shape.append(d_bond)
                # Z connections
                if z > 0:
                    node_edges.append(f'z_{x}_{y}_{z-1}')
                    shape.append(d_bond)
                if z < h - 1:
                    node_edges.append(f'z_{x}_{y}_{z}')
                    shape.append(d_bond)
                    
                tensors.append(np.random.randn(*shape))
                edges.append(node_edges)
    return tensors, edges

def generate_random_regular(n, degree=3, d_bond=3):
    """Generates a closed Random Regular Graph tensor network."""
    G = nx.random_regular_graph(degree, n)
    tensors = []
    edges = []
    
    # Map edges to unique index names
    edge_to_name = {e: f'e_{i}' for i, e in enumerate(G.edges())}
    
    for u in G.nodes():
        node_edges = []
        shape = []
        for v in G.neighbors(u):
            e = (u, v) if (u, v) in edge_to_name else (v, u)
            node_edges.append(edge_to_name[e])
            shape.append(d_bond)
            
        tensors.append(np.random.randn(*shape))
        edges.append(node_edges)
        
    return tensors, edges

def generate_binary_tree(depth, d_bond=3):
    """Generates a closed binary tree tensor network with no physical legs."""
    tensors = []
    edges = []
    
    total_nodes = (2 ** depth) - 1
    
    for i in range(1, total_nodes + 1):
        node_edges = []
        shape = []
        
        # Parent connection
        if i > 1:
            parent = i // 2
            node_edges.append(f't_{parent}_{i}')
            shape.append(d_bond)
            
        # Children connections
        left_child = 2 * i
        right_child = 2 * i + 1
        
        if left_child <= total_nodes:
            node_edges.append(f't_{i}_{left_child}')
            shape.append(d_bond)
        if right_child <= total_nodes:
            node_edges.append(f't_{i}_{right_child}')
            shape.append(d_bond)
            
        tensors.append(np.random.randn(*shape))
        edges.append(node_edges)
        
    return tensors, edges
