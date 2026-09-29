"""Slicing study (counting only): FLOP overhead vs. memory reduction for random-circuit amplitude networks.
Usage: python run_study.py  ->  results.json, results.md, overhead_vs_memory.png"""
import json
import math
import random
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import cotengra as ctg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tnslice import *
warnings.filterwarnings("ignore")  # cotengra warns that kahypar/optuna are not installed (see README)
INSTANCES = {"1D 30q d20": (30, brickwork(30, 20), "small"), "2D 6x6 d12": (36, grid(6, 6, 12), "small"),
             "1D 36q d28": (36, brickwork(36, 28), "large"), "2D 8x8 d12": (64, grid(8, 8, 12), "large")}
BITS, SEEDS, DRAWS = (1, 2, 4, 6, 8), (0, 1, 2), 5     # memory reductions m (2^m smaller), path-finder seeds, random draws
STRATS = ["uniform", "memory", "greedy", "ctg-slice", "ctg-reconf"]

def study(name):
    n, layers, size = INSTANCES[name]
    ts, ins = network(n, layers, random_gates(layers, np.random.default_rng(0)), [0] * n)   # counts do not depend on gate values
    ms, back, rows, alt, t0 = masks(ins), {ctg.get_symbol(l): l for i in ins for l in i}, [], [], time.time()
    for seed in SEEDS:
        random.seed(seed), np.random.seed(seed)
        tree = find_tree(ins).subtree_reconfigure()                      # common base tree (min-FLOP) for all strategies
        path = tree.get_ssa_path()
        macs0, big0, _ = stats(path, ms)
        t2 = find_tree(ins, minimize="size").subtree_reconfigure(minimize="size")   # unsliced, memory-optimised tree
        macs2, big2, _ = stats(t2.get_ssa_path(), ms)
        alt.append(dict(instance=name, seed=seed, m=big0 - big2, log2_overhead=math.log2(macs2 / macs0)))
        for m in BITS:
            tgt = big0 - m
            def add(strategy, S, p=path):
                macs, big, _ = stats(p, ms, S)
                assert big <= tgt, (name, strategy, big, tgt)                # every strategy really reached the target
                rows.append(dict(instance=name, seed=seed, strategy=strategy, m=m, k=S.bit_count(), log2big=big,
                                 log2_overhead=math.log2(macs) + S.bit_count() - math.log2(macs0)))
            for d in range(DRAWS):
                for how in ("uniform", "memory"):
                    add(how, slice_to(path, ms, tgt, how, np.random.default_rng([seed, d])))
            add("greedy", slice_to(path, ms, tgt, "greedy"))
            mask = lambda t: sum(1 << back[s] for s in t.sliced_inds)
            add("ctg-slice", mask(tree.slice(target_size=2 ** tgt, minimize="flops")))
            if seed == 0 and (size == "small" or m <= 4):                    # the expensive strategy: one tree, small m on big nets
                t = tree.slice_and_reconfigure(target_size=2 ** tgt, minimize="flops")
                add("ctg-reconf", mask(t), t.get_ssa_path())
        print(f"{name} seed {seed}: {len(ins)} tensors, unsliced 2^{math.log2(macs0):.2f} MACs, largest intermediate 2^{big0}, {time.time() - t0:.0f}s", flush=True)
    return rows, alt

def summ(rows, name, s, m):
    """(k, log2 overhead) as the median over seed-0 runs, and the min/max log2 overhead over all seeds and draws; None if absent."""
    v = [r for r in rows if (r["instance"], r["strategy"], r["m"]) == (name, s, m)]
    v0, o = [r for r in v if r["seed"] == 0], [r["log2_overhead"] for r in v]
    return v and (np.median([r["k"] for r in v0]), np.median([r["log2_overhead"] for r in v0]), min(o), max(o))

def report(rows, alt):
    fmt = lambda x: f"{2 ** x:.3g}" if x < 60 else f"1e{x * math.log10(2):.0f}"
    out = ["| instance | strategy | " + " | ".join(f"{2 ** m}x smaller: k / overhead (min-max)" for m in (2, 4, 8)) + " |", "|---|---|---|---|---|"]
    for name in INSTANCES:
        for s in STRATS:
            c = [summ(rows, name, s, m) for m in (2, 4, 8)]
            out.append(f"| {name} | {s} | " + " | ".join(f"{x[0]:g} / {fmt(x[1])}" + ("" if s == "ctg-reconf" else f" ({fmt(x[2])}-{fmt(x[3])})") if x else "n/a" for x in c) + " |")
    out += ["", "Unsliced memory-optimised tree (no slicing), per path-finder seed: memory reduction m reached / FLOP ratio to the min-FLOP tree", ""]
    return "\n".join(out + [f"- {n}: " + ", ".join(f"m={a['m']} / {2 ** a['log2_overhead']:.3g}x" for a in alt if a["instance"] == n) for n in INSTANCES])

def figure(rows):
    sty = {"memory": ("#8a8985", "o", "--", "random among largest-tensor indices"), "greedy": ("#2a78d6", "s", "-", "greedy, fixed tree"),
           "ctg-slice": ("#eb6834", "^", "-", "cotengra tree.slice, fixed tree"), "ctg-reconf": ("#1baf7a", "D", "-", "cotengra slice_and_reconfigure")}
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.5), facecolor="#fcfcfb", sharex=True, subplot_kw=dict(facecolor="#fcfcfb"))
    for ax, name in zip(axs.T.flat, INSTANCES):                       # columns: small / large; rows: 1D / 2D
        for s, (c, mk, ls, lab) in sty.items():
            pts = [(m, summ(rows, name, s, m)) for m in BITS if summ(rows, name, s, m)]
            ax.plot([m for m, _ in pts], [2 ** x[1] for _, x in pts], color=c, marker=mk, ls=ls, lw=2, ms=6, mec="#fcfcfb", label=lab)
            ax.fill_between([m for m, _ in pts], [2 ** x[2] for _, x in pts], [2 ** x[3] for _, x in pts], color=c, alpha=0.15, lw=0)
        top = max(2 ** r["log2_overhead"] for r in rows if r["instance"] == name and r["strategy"] in ("greedy", "ctg-slice", "ctg-reconf"))
        ax.set(yscale="log", xticks=BITS, ylim=(0.8, 30 * top))       # the random baseline is clipped at the top; its m=8 value is annotated
        ax.text(0.02, 0.97, f"dashed: clipped, {2 ** summ(rows, name, 'memory', 8)[1]:.2g} at m=8", transform=ax.transAxes, va="top", fontsize=8, color="#52514e")
        ax.set_title(name, loc="left", fontsize=11, color="#0b0b0b")
        ax.grid(color="#e4e3df", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("memory reduction m (largest intermediate 2^m times smaller)", color="#52514e", fontsize=10)
    fig.supylabel("FLOP overhead = total sliced / unsliced", color="#52514e", fontsize=10)
    fig.suptitle("FLOP overhead of slicing vs. memory saved (line: path seed 0; band: min-max over seeds and draws)", y=0.985, fontsize=11)
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=4, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig("overhead_vs_memory.png", dpi=150)

if __name__ == "__main__":
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(study, INSTANCES))
    rows, alt = [r for x in res for r in x[0]], [a for x in res for a in x[1]]
    json.dump(dict(rows=rows, min_memory_tree=alt), open("results.json", "w"), indent=1)
    open("results.md", "w").write(report(rows, alt) + "\n")
    figure(rows)
    print(report(rows, alt))
