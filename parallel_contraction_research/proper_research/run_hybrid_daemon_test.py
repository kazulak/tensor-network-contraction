import os
import sys
import time
import socket
import subprocess
import numpy as np

# Ensure proper_research root in path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from src.network_generators import generate_2d_grid
from src.exporter import export_contraction_job

def get_daemon_connection(port=8080):
    """Auto-connects to or starts the persistent Julia daemon."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect(("127.0.0.1", port))
        print("Connected to existing Julia Daemon.")
        return s
    except ConnectionRefusedError:
        print("Julia Daemon not running. Auto-starting background daemon...")
        daemon_script = os.path.join(current_dir, "src", "hybrid_daemon.jl")
        
        # Start Julia daemon as a background subprocess
        # We start it with 4 threads
        subprocess.Popen(
            ["julia", "--threads=4", daemon_script, str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Poll socket until it accepts connection
        for attempt in range(30):
            time.sleep(0.2)
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect(("127.0.0.1", port))
                print(f"Daemon successfully initialized after {0.2 * (attempt+1):.1f}s.")
                return s
            except ConnectionRefusedError:
                continue
        raise RuntimeError("Could not connect to Julia daemon after 6 seconds.")

def send_job_to_daemon(job_dir, port=8080):
    """Sends the job path to the daemon and returns the parsed result."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("127.0.0.1", port))
    # Send path followed by newline
    s.sendall(f"{job_dir}\n".encode())
    
    # Read response
    response_data = []
    while True:
        chunk = s.recv(4096)
        if not chunk:
            break
        response_data.append(chunk)
    s.close()
    
    response = b"".join(response_data).decode()
    lines = response.splitlines()
    if lines[0] != "SUCCESS":
        raise RuntimeError(f"Daemon job failed: {response}")
        
    res_val = float(lines[1])
    contract_time = float(lines[2])
    total_time = float(lines[3])
    return res_val, contract_time, total_time

def shutdown_daemon(port=8080):
    """Gracefully shuts down the background Julia daemon."""
    print("\nSending SHUTDOWN signal to Julia Daemon...")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("127.0.0.1", port))
        s.sendall("SHUTDOWN\n".encode())
        response = s.recv(1024).decode()
        s.close()
        print(f"Daemon response: {response.strip()}")
    except Exception as e:
        print(f"Error shutting down daemon: {e}")

def test_persistent_daemon():
    print("=" * 80)
    print("           DEMONSTRATING PERSISTENT JULIA DAEMON (ZERO JIT LATENCY)")
    print("=" * 80)
    
    port = 8085
    # 1. Connect or auto-start
    get_daemon_connection(port)
    
    # Generate 3x3 PEPS grid contraction
    print("\nPreparing PEPS grid contraction...")
    tensors, edges = generate_2d_grid(rows=3, cols=3, d_bond=3)
    job_dir = os.path.join(current_dir, "results", "daemon_job")
    export_contraction_job(tensors, edges, target_slices=16, job_dir=job_dir)
    
    # Run 1: First call (compiling & executing)
    print("\n[Run 1] Sending job to daemon (compiling & executing)...")
    start = time.perf_counter()
    res1, contract_time1, total_time1 = send_job_to_daemon(job_dir, port)
    end = time.perf_counter()
    python_total_time1 = end - start
    print(f"  Result: {res1:.6f} | Julia Contraction: {contract_time1:.4f}s | Roundtrip: {python_total_time1:.4f}s")
    
    # Run 2: Second call (already compiled)
    print("\n[Run 2] Sending job to daemon again...")
    start = time.perf_counter()
    res2, contract_time2, total_time2 = send_job_to_daemon(job_dir, port)
    end = time.perf_counter()
    python_total_time2 = end - start
    print(f"  Result: {res2:.6f} | Julia Contraction: {contract_time2:.4f}s | Roundtrip: {python_total_time2:.4f}s")
    
    # Run 3: Third call (fully warmed up)
    print("\n[Run 3] Sending job to daemon third time...")
    start = time.perf_counter()
    res3, contract_time3, total_time3 = send_job_to_daemon(job_dir, port)
    end = time.perf_counter()
    python_total_time3 = end - start
    print(f"  Result: {res3:.6f} | Julia Contraction: {contract_time3:.4f}s | Roundtrip: {python_total_time3:.4f}s")
    
    # 4. Graceful Shutdown
    shutdown_daemon(port)
    
    # Clean up folders
    subprocess.run(["rm", "-rf", job_dir])
    
    print("\n" + "=" * 80)
    print("                         TIMING COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Run 1 (Cold)':<15} | {'Run 2 (Warm)':<15} | {'Run 3 (Hot)':<15}")
    print("-" * 80)
    print(f"{'Julia Contraction (s)':<30} | {contract_time1:<15.4f} | {contract_time2:<15.4f} | {contract_time3:<15.4f}")
    print(f"{'Python Roundtrip (s)':<30} | {python_total_time1:<15.4f} | {python_total_time2:<15.4f} | {python_total_time3:<15.4f}")
    print("-" * 80)
    speedup = python_total_time1 / python_total_time3
    print(f"Pre-compiled Daemon Speedup Factor: {speedup:.2f}x faster roundtrip!")
    print("=" * 80)

if __name__ == "__main__":
    test_persistent_daemon()
