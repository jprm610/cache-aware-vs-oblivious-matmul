**English** | [Español](README.es.md)

# Iterated Matrix Multiplication Benchmark

**Course:** Computer Architecture
**Authors:** Santiago Uribe Echavarría - Juan Pablo Robledo Meza
**University:** Universidad Nacional de Colombia, Medellin campus
**Date:** May 29, 2026
**Reference hardware:** Ryzen 5 4600H (Renoir, Zen 2)

This repository implements and measures the recurrence $B_{i+1} = A \cdot B_i$ over eight different kernels (baseline, loop reordering, tiling, AVX2 vectorization and OpenMP parallelization), plus a `perf` profiling pipeline on Zen 2 hardware.

> **Note:** the per-module documentation under [`docs/`](docs/) is currently written in Spanish. The Spanish version of this README is [`README.es.md`](README.es.md).

## 0. Table of contents

1. [What this project does](#1-what-this-project-does)
2. [Reference hardware](#2-reference-hardware)
3. [Repository structure](#3-repository-structure)
4. [WSL2 setup](#4-wsl2-setup)
5. [Quickstart](#5-quickstart)
6. [Building](#6-building)
7. [Validating correctness](#7-validating-correctness)
8. [Running benchmarks](#8-running-benchmarks)
9. [Profiling with `perf`](#9-profiling-with-perf)
10. [Reproducibility and stability](#10-reproducibility-and-stability)
11. [Adapting the project to other hardware or microkernels](#11-adapting-the-project-to-other-hardware-or-microkernels)
12. [Additional documentation](#12-additional-documentation)

---

## 1. What this project does

It measures the recurrence

$$
B_{i+1} = A \cdot B_i, \quad i = 0, 1, \ldots, I-1, \quad B_0 = Z
$$

with $A \in \mathbb{R}^{m \times m}$, $Z \in \mathbb{R}^{m \times 128}$, $I = 2m/n$. Each iteration stores the first $n$ rows of the product in an output buffer.

The goal of the project is not matmul itself, but to **measure the incremental effect** of each optimization technique (loop order, tiling, register blocking, SIMD, OpenMP) over a naive baseline. All eight kernels share the same public signature (Section 1.5 of `docs/API.md`) and are evaluated with the same hardware-counter pipeline.

---

## 2. Reference hardware

All hyperparameter decisions in the project (tile sizes, OMP configuration, recursion threshold) were chosen for the Ryzen 5 $4600$H:

| Resource | Value |
|---|---|
| Microarchitecture | Zen 2 (Renoir) |
| Physical cores | $6$, organized in $2$ CCXs of $3$ cores |
| Logical threads | $12$ (SMT) |
| L1d | $32$ KiB per core |
| L2 | $512$ KiB per core |
| L3 | $4$ MiB **private per CCX** |
| SIMD ISA | AVX2 + FMA (no AVX-512) |
| Turbo frequency under AVX2 | $\sim 4.0$ GHz |

Each decision is justified in the corresponding module doc: $M_c = K_c = 256$ ([`docs/1.3) matmul_tiled_ikj.md`](<docs/1.3) matmul_tiled_ikj.md>)) makes the three active panels fill the L2; $M_C = 192,\ k_c = 384$ ([`docs/1.4) matmul_tiled_ikj_avx2.md`](<docs/1.4) matmul_tiled_ikj_avx2.md>)) leaves the active $288$ KiB $A$ panel in L2 and the $24$ KiB $B$ panel in L1d; the default OpenMP setting of $6$ threads ([`docs/1.5) matmul_tiled_ikj_omp.md`](<docs/1.5) matmul_tiled_ikj_omp.md>)) uses one thread per physical core so FMA pipes are not shared with the SMT sibling.

If you are going to run the project on another PC, see [Section 11](#11-adapting-the-project-to-other-hardware-or-microkernels) for the places to change.

---

## 3. Repository structure

```
.
|-- README.md
|-- README.es.md                          Spanish version of this README
|-- Makefile
|-- docs/                                 Project documentation (see Section 12)
|-- src/
|   |-- core/                             Modules shared by all algorithms
|   |   |-- matrix_utils.{h,c}            Aligned allocation, init_*, matrices_close, scalar_t
|   |   |-- timing.h                      Inline now_seconds() over CLOCK_MONOTONIC
|   |   `-- morton.{h,c}                  Z-order encoding and row-major <-> Morton reorganization
|   |-- microkernels/                     Header-only AVX2 tiles (static inline)
|   |   |-- kernel_avx2_morton.h          4x16, used by the Morton family
|   |   `-- kernel_avx2_tiled.h           6x16, used by the tiled_ikj family
|   |-- algorithms/
|   |   |-- naive/                        matmul_naive (ijk baseline)
|   |   |-- loops/                        matmul_loops (6 orders selectable by name)
|   |   |-- tiled_ikj/                    matmul_tiled_ikj{,_avx2,_omp}
|   |   `-- morton/                       matmul_morton{,_avx2,_omp}
|   |-- drivers/                          main programs
|   |   |-- bench/                        bench_*.c (one per kernel)
|   |   `-- validate/                     validate_*.c (one per kernel)
|   `-- tests/                            Standalone unit tests (test_*.c)
|-- scripts/
|   |-- profile_perf_zen2.sh              perf capture per cell (variant, m)
|   |-- run_perf_zen2_sweep.sh            Orchestrates the full sweep
|   |-- consolidate_perf_zen2.py          Merges groups A+B -> results/metrics.csv
|   `-- plot_metrics_perf_zen2.py         Local plots (Zen 2): GFLOPS vs m, optimization arc, cache hierarchy, OMP scaling
|-- bin/                                  Compiled binaries (gitignored)
|   |-- bench/                            bench_<kernel>_O3
|   |-- validate/                         validate_<kernel>_O0 or _O3
|   `-- tests/                            test_<module>
|-- results/                              CSVs and profiling reports (gitignored)
`-- plots/                                Generated images (gitignored)
```

---

## 4. WSL2 setup

The project **requires WSL2 with Ubuntu** for compatibility with `gprof` and `perf`. Tested on Ubuntu $22.04$ and $24.04$.

### 4.1 Install WSL2 + Ubuntu

From **PowerShell as administrator**:

```powershell
wsl --install -d Ubuntu
```

After the reboot, open Ubuntu from the Start menu and create your user. Verify:

```bash
uname -a   # should say Linux ... microsoft-standard-WSL2
```

### 4.2 Required packages

```bash
sudo apt update
sudo apt install -y build-essential gcc make binutils
```

Provides `gcc` (compiler), `make` (build), `gprof` (per-function profiling), `ar`/`objdump` (analysis).

### 4.3 perf

First attempt with the generic package:

```bash
sudo apt install -y linux-tools-generic linux-tools-common
perf --version
```

If it fails with "perf not found for kernel ...", build it from the WSL2 kernel repository:

```bash
sudo apt install -y flex bison libelf-dev libdwarf-dev libdw-dev libnuma-dev pkg-config
cd /usr/src
sudo git clone --depth=1 https://github.com/microsoft/WSL2-Linux-Kernel.git
cd WSL2-Linux-Kernel/tools/perf
sudo make
sudo cp perf /usr/local/bin/
perf --version
```

Adjust `perf_event_paranoid` so PMU counters are accessible without root (once per boot, or persistently):

```bash
sudo sysctl -w kernel.perf_event_paranoid=1                            # temporary
echo "kernel.perf_event_paranoid=1" | sudo tee /etc/sysctl.d/99-perf.conf && sudo sysctl --system   # persistent
```

On WSL2 some PMU events may return `<not supported>` (hypervisor limitation); the profiling script logs this and continues.

### 4.4 Python + matplotlib

For the plots (`make plots`):

```bash
sudo apt install -y python3 python3-pip python3-venv
python3 -m venv ~/venvs/matmul
source ~/venvs/matmul/bin/activate
pip install matplotlib numpy
```

Activate the venv (`source ~/venvs/matmul/bin/activate`) every time you open a terminal before running the plots.

### 4.5 Cachegrind (optional)

To diagnose cache misses per line of code (slower than `perf` but independent of the hypervisor):

```bash
sudo apt install -y valgrind
```

Basic usage:

```bash
valgrind --tool=cachegrind --cache-sim=yes ./bin/bench/bench_naive_O3 1024 1
cg_annotate cachegrind.out.<pid> | less
```

### 4.6 Environment check

```bash
gcc --version
make --version
gprof --version
perf --version || echo "perf is not ready"
python3 -c "import matplotlib, numpy; print('python OK')"
```

---

## 5. Quickstart

From the repository root in WSL2:

```bash
make build       # 1. builds the 20 binaries (8 bench + 8 validate + 4 tests)
make validate    # 2. runs the tests -> validate pyramid over the 8 kernels
make results     # 3. Zen 2 perf sweep -> results/metrics.csv
make plots       # 4. generates the 4 figures in plots/ (optional, needs a venv with matplotlib)
```

If the first three steps finish with `OK`, the repo is healthy and `results/metrics.csv` has one row per `(variant, m)` cell with all counters. The full sweep takes about $20$-$30$ minutes on the $4600$H; use the knobs in [Section 8.3](#83-unified-sweep-make-results) to limit its scope during development.

---

## 6. Building

### 6.1 Aggregate targets

| Target | What it does |
|---|---|
| `make build` (alias `make all`) | Builds the $20$ binaries without running anything |
| `make tests` | Builds and runs the $4$ unit tests in order |
| `make validate` | Depends on `tests`; runs the $8$ `validate_*` in order |
| `make results` | Builds the $8$ benches and launches the perf sweep $\to$ `results/metrics.csv` |
| `make plots` | Reads `results/metrics.csv` and generates $4$ figures in `plots/` (GFLOPS vs $m$, optimization arc, cache hierarchy, OMP scaling). Requires `matplotlib` |
| `make clean` | Deletes `bin/` and `build/` |
| `make distclean` | `clean` + deletes `results/*.csv` and `plots/*` |

### 6.2 Individual targets

| Kernel | Bench target | Validate target |
|---|---|---|
| `naive` | `make bench_naive_O3` | `make validate_naive` |
| `loops` | `make bench_loops_O3` | `make validate_loops` |
| `tiled_ikj` | `make bench_tiled_ikj_O3` | `make validate_tiled_ikj` |
| `tiled_ikj_avx2` | `make bench_tiled_ikj_avx2_O3` | `make validate_tiled_ikj_avx2` |
| `tiled_ikj_omp` | `make bench_tiled_ikj_omp_O3` | `make validate_tiled_ikj_omp` |
| `morton` | `make bench_morton_O3` | `make validate_morton` |
| `morton_avx2` | `make bench_morton_avx2_O3` | `make validate_morton_avx2` |
| `morton_omp` | `make bench_morton_omp_O3` | `make validate_morton_omp` |

The complete target reference, including the profiling targets (`profile_zen2`, `profile_zen2_one`, `profile_zen2_omp`, `consolidate_zen2`) and the analysis targets (`plots`), is in [`docs/0.0) makefile.md`](<docs/0.0) makefile.md>).

### 6.3 Flags per suffix

| Suffix | Variable | Flags |
|---|---|---|
| `_O0` | `BASE_CFLAGS` | `-std=c11 -Wall -Wextra -Wpedantic -O0 -g -fno-omit-frame-pointer -D_POSIX_C_SOURCE=200809L` |
| `_O3` | `CFLAGS_O3_ZEN2` | `BASE_CFLAGS` with `-O3 -march=znver2 -mavx2 -mfma` (without `-O0` or `-g`) |
| `_O3` + OMP | `CFLAGS_OMP_ZEN2` | `CFLAGS_O3_ZEN2` + `-fopenmp` |

`-O0 -g` is used where we want the code to reflect the algorithm without compiler reordering (`validate_naive`, `validate_loops`, etc.). The benches and the validates of the vectorized variants use `_O3` because the microkernels are `static inline` with intrinsics: at `-O0` the FMA instructions are not materialized and the test would measure something else.

---

## 7. Validating correctness

### 7.1 Three-level pyramid

The project verifies with three layers, from fine to coarse:

```
tests (unit)     -> individual pieces in isolation
validate         -> full algorithms against algebraic invariants + cross-check vs naive
bench            -> performance (not correctness)
```

`make validate` runs `tests` first and aborts if they fail. Layer details in [`docs/1.9) tests.md`](<docs/1.9) tests.md>).

### 7.2 How to run

```bash
make tests        # 4 unit tests over core/ and microkernels/
make validate     # tests + 8 validate_* in order, aborts at the first failure
```

Individually:

```bash
./bin/validate/validate_naive_O0           # m = 256 by default
./bin/validate/validate_naive_O0 512       # custom m
./bin/validate/validate_tiled_ikj_avx2_O3 256 384   # m, bs (only tiled_*_avx2/omp)
./bin/tests/test_morton                    # morton module tests
```

### 7.3 Covered invariants

All `validate_*` check the three basic algebraic invariants:

1. $A \cdot 0 = 0$
2. $I \cdot Z = Z$
3. $A \cdot (Z_1 + Z_2) = A \cdot Z_1 + A \cdot Z_2$

The `validate_*` other than the baseline also add cross-validation against `matmul_naive` over a sweep of sizes. Per-validate tolerances and details of the additional tests are in `docs/API.md` Section 5.2.

Expected output:

```
Validating matmul_naive at m=256, n=128
Tolerances: abs=1.0e-04, rel=1.0e-03
  [OK]   A * 0 == 0
  [OK]   I * Z == Z
  [OK]   A * (Z1+Z2) == A*Z1 + A*Z2
VALIDATION OK
```

Exit code `0` on success, `1` at the first failure, with a report of the first diverging index and values.

---

## 8. Running benchmarks

### 8.1 Common convention

**Standard CLI:**

```
bench_<kernel>_O3 <m> [num_iters] [num_runs]
```

Variants with an extra knob:

```
bench_loops_O3 <order> <m> [num_iters] [num_runs]              # order = ijk|ikj|jik|jki|kij|kji
bench_tiled_ikj_avx2_O3 <m> [num_iters] [num_runs] [bs]        # bs = k_c, default 384
bench_tiled_ikj_omp_O3  <m> [num_iters] [num_runs] [bs]
```

**Defaults:** `num_iters = min(2m/n, 4)`, `num_runs = 5`.

**Protocol:** one unmeasured warm-up run + `num_runs` measured runs; the **median** of the times is reported.

**Output:** one CSV line on `stdout`:

```
<kernel>,m,n,num_iters,median_seconds,gflops
```

For `tiled_ikj_avx2` and `tiled_ikj_omp` a `bs` column is inserted ($7$-column format).

**Morton family restriction:** $m$ must be a power of $2$ and $\geq 4$ (Z-order indexing requires exact subdivisions). The binary aborts with a clear message otherwise. The reorganization to Morton runs **once before the warm-up**, outside the timed region.

### 8.2 Examples

```bash
# Naive
./bin/bench/bench_naive_O3 1024 4 1

# Loops, ikj order
./bin/bench/bench_loops_O3 ikj 1024 4 1

# Tiled_ikj_avx2 with custom bs
./bin/bench/bench_tiled_ikj_avx2_O3 4096 4 1 256

# Tiled_ikj_omp with 6 threads and close binding
OMP_NUM_THREADS=6 OMP_PLACES=cores OMP_PROC_BIND=close \
    ./bin/bench/bench_tiled_ikj_omp_O3 4096

# Morton (m power of 2)
./bin/bench/bench_morton_avx2_O3 4096 4 1

# Morton OpenMP
OMP_NUM_THREADS=6 OMP_PLACES=cores OMP_PROC_BIND=close \
    ./bin/bench/bench_morton_omp_O3 4096
```

### 8.3 Unified sweep (`make results`)

To compare kernels and produce a consolidated CSV:

```bash
make results                                        # defaults
make results MS="1024 4096"                         # only those sizes
make results VARIANTS="tiled_ikj_avx2 morton_avx2"  # only those variants
make results ITERS_PER_RUN=0                        # I_full = 2m/n, 1 run
```

**Knobs:**

| Variable | Default | What it controls |
|---|---|---|
| `VARIANTS` | $13$ active | List of variants to run (space-separated) |
| `MS` | `1024 2048 4096 8192 16384 32768` | List of sizes |
| `ITERS_PER_RUN` | `1` | Iterations per run. `0` enables $I_{\text{full}} = 2m/n$ |
| `RUNS` | `3` (or `1` if `ITERS_PER_RUN=0`) | Measured runs for the median |

**Active variants** (the ones `run_perf_zen2_sweep.sh` runs by default): `naive`, `loop_ijk`, `loop_ikj`, `loop_jik`, `loop_jki`, `loop_kij`, `loop_kji`, `tiled_ikj`, `tiled_ikj_avx2`, `tiled_ikj_omp`, `morton`, `morton_avx2`, `morton_omp`.

**Output:** `results/metrics.csv` with one row per `(variant, m)` cell and all hardware-counter columns consolidated.

---

## 9. Profiling with `perf`

### 9.1 What it captures

`make results` launches `scripts/run_perf_zen2_sweep.sh`, which for each `(variant, m)` cell runs `perf stat` twice to avoid counter multiplexing. The raw `.txt` files are left in `results/<variant>/perf_<variant>_m<M>_{A,B}.txt`; the consolidator `scripts/consolidate_perf_zen2.py` merges them into `results/metrics.csv`.

**Events per group (the ones the script requests from `perf`):**

| Group | Events |
|---|---|
| A (compute side) | `cycles`, `instructions`, `fp_ret_sse_avx_ops.all`, `ls_dispatch.ld_dispatch`, `l2_request_g1.all_no_prefetch` |
| B (memory + TLB) | `cycles`, `instructions`, `l2_cache_req_stat.ls_rd_blk_l_hit_x`, `cache-misses`, `bp_l1_tlb_miss_l2_tlb_miss`, `dTLB-load-misses` |

`cycles` and `instructions` are duplicated in both groups to cross-check between passes. Some events specified by the course rubric are not exposed by the current WSL2 kernel + microcode; the script replaces them with equivalent proxies and documents the details in the header of `scripts/profile_perf_zen2.sh`.

### 9.2 How to launch

```bash
sudo sysctl -w kernel.perf_event_paranoid=1     # once per boot

make profile_zen2                               # full sweep (all variants and MS)
make profile_zen2_one VARIANT=tiled_ikj_avx2 M=4096   # a single cell
make profile_zen2_omp                           # morton_omp with several thread counts
```

### 9.3 Plots

```bash
source ~/venvs/matmul/bin/activate
make plots                                      # generates 4 figures in plots/
```

`make plots` reads `results/metrics.csv` and leaves in `plots/`:

| Figure | Content |
|---|---|
| `gflops_vs_m.png` | Throughput (GFLOPS) vs $m$, one line per variant, log-log axes. Global view. |
| `best_per_family.png` | Curated subset (`naive`, `loop_ikj`, `tiled_ikj` family, `morton` family) with single-core and all-core theoretical ceilings ($\sim 118$ and $\sim 634$ GFLOPS). Summarizes the optimization arc in a single figure. |
| `cache_hierarchy.png` | $3$ panels vs $m$: L1D miss rate, L2 load hit rate, L3 miss rate. Diagnoses at which cache level each variant breaks down. |
| `omp_scaling.png` | Paired bars `*_avx2` vs `*_omp` for morton and tiled_ikj. Labels speedup and parallel efficiency with respect to $6$ cores. |

Color code per family: gray (`naive`), warm yellow/brown tones (`loop_*`), dark green (`tiled_ikj*`), dark blue (`morton*`). Marker per sub-variant: circle (plain), triangle (AVX2), star (OMP).

If you do not have `results/metrics.csv` yet, run `make results` first. The targets are independent:

```bash
make results && make plots
```

---

## 10. Reproducibility and stability

### 10.1 Document the hardware

Save this info alongside your results:

```bash
lscpu
cat /sys/devices/system/cpu/cpu0/cache/index{0,1,2,3}/size
free -h
```

### 10.2 Set the Windows power plan

WSL2 inherits the host governor. From PowerShell:

```powershell
powercfg /setactive SCHEME_MIN   # Maximum performance
```

### 10.3 Close heavy programs

Chrome with many tabs, Slack, Zoom, Docker Desktop, OneDrive syncing, etc. compete for L3 and memory and add noise. Close them before measuring.

### 10.4 Median of several runs

The benches already report the median of $5$ runs by default. For the sweep, adjust with `RUNS=<n>`.

---

## 11. Adapting the project to other hardware or microkernels

The defaults are calibrated for Zen 2. If you are going to run on another PC, these are the places to change:

| What to change | Where | Why |
|---|---|---|
| `-march=znver2` | `CFLAGS_O3_ZEN2` and `CFLAGS_OMP_ZEN2` in the `Makefile` | Selects GCC's ISA tuning for the target chip |
| Tile sizes $M_C$, $k_c$ | `TILED_IKJ_AVX2_MC`, `TILED_IKJ_AVX2_BS_DEFAULT` and their `_OMP` equivalents in `src/algorithms/tiled_ikj/*.h` | They must fit the L1d and L2 of the new chip |
| Microkernel tile $M_R, N_R$ | `kernel_avx2_morton.h` ($4 \times 16$) and `kernel_avx2_tiled.h` ($6 \times 16$) | Depend on the number of available vector registers (16 YMM on Zen 2, 32 ZMM on Zen 5) |
| Default `OMP_NUM_THREADS` | `profile_perf_zen2.sh` (`tiled_ikj_omp` / `morton_omp` case) | Must match the number of physical cores of the new chip |
| Morton thresholds | `g_recursion_threshold*` in `src/algorithms/morton/*.c` | Calibrated to the 4600H's L1d; change with the L1d size |

To replace a microkernel with your own version: copy `src/microkernels/kernel_avx2_*.h` to a new header, adjust the $M_R$, $N_R$ constants and the kernel body, and change the `#include` in the algorithm module that consumes it (`matmul_morton_avx2.c`, `matmul_tiled_ikj_avx2.c`, etc.). The external contract of each `matmul_*` does not change, so the `validate_*` programs keep working untouched.

The parallel branch `main_server` is an example of a port to another microarchitecture: it takes the project to AVX-512 on Zen 5 with the same kernels rescaled to tiles that exploit the 32 ZMM registers. It serves as a reference for how far a "complete" port goes.

---

## 12. Additional documentation

> The documents below are written in Spanish.

| Document | Covers |
|---|---|
| [`docs/API.md`](docs/API.md) | Public contract (signatures, types, layout, constants, binaries) |
| [`docs/0.0) makefile.md`](<docs/0.0) makefile.md>) | Complete reference of `Makefile` targets and knobs |
| [`docs/0.1) matrix_utils.md`](<docs/0.1) matrix_utils.md>) | `core/matrix_utils` helpers |
| [`docs/0.2) morton.md`](<docs/0.2) morton.md>) | Z-order encoding and reorganization |
| [`docs/1.1) matmul_naive.md`](<docs/1.1) matmul_naive.md>) | `ijk` baseline |
| [`docs/1.2) matmul_loops.md`](<docs/1.2) matmul_loops.md>) | Loop reordering ($6$ variants) |
| [`docs/1.3) matmul_tiled_ikj.md`](<docs/1.3) matmul_tiled_ikj.md>) | Explicit tiling targeting L2 |
| [`docs/1.4) matmul_tiled_ikj_avx2.md`](<docs/1.4) matmul_tiled_ikj_avx2.md>) | BLIS-style $6 \times 16$ microkernel with AVX2 + FMA |
| [`docs/1.5) matmul_tiled_ikj_omp.md`](<docs/1.5) matmul_tiled_ikj_omp.md>) | $6 \times 16$ + OpenMP `parallel for` over $i_c$ |
| [`docs/1.6) matmul_morton.md`](<docs/1.6) matmul_morton.md>) | Fine-grained cache-oblivious Morton recursion |
| [`docs/1.7) matmul_morton_avx2.md`](<docs/1.7) matmul_morton_avx2.md>) | Morton-of-blocks + AVX2 $4 \times 16$ microkernel |
| [`docs/1.8) matmul_morton_omp.md`](<docs/1.8) matmul_morton_omp.md>) | Morton-of-blocks + OpenMP tasks |
| [`docs/1.9) tests.md`](<docs/1.9) tests.md>) | Unit-test pyramid (structure and covered cases) |
