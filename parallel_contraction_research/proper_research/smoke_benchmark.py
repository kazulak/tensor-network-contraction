import numpy as np
import time
import opt_einsum

def run_smoke_benchmark():
    print("Running tiny tensor network contraction smoke benchmark...")
    
    # Create a small loop tensor network:
    # Tensors: A_ij, B_jk, C_kl, D_li
    # Dimensions: 50x50
    dim = 50
    np.random.seed(42)
    A = np.random.randn(dim, dim)
    B = np.random.randn(dim, dim)
    C = np.random.randn(dim, dim)
    D = np.random.randn(dim, dim)
    
    views = [A, B, C, D]
    
    # Generate and optimize the contraction path
    expr = opt_einsum.contract_expression("ij,jk,kl,li->", *[v.shape for v in views])
    print("Contraction expression generated successfully.")
    print("Path Details:")
    print(expr)
    
    # Execute the contraction
    start = time.perf_counter()
    result = expr(A, B, C, D)
    elapsed = time.perf_counter() - start
    
    print(f"Result of contraction: {result:.6f}")
    print(f"Contraction finished in {elapsed:.6f} seconds.")
    print("Smoke benchmark PASSED.")

if __name__ == "__main__":
    run_smoke_benchmark()
