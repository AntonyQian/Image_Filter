# Image Filter Project: Extension Ideas

Goal: after the base C++ / OpenMP image filtering tool works, add depth that lines up with NVIDIA's focus areas (parallel computing, computer architecture, GPUs, PyTorch, Linux basics).

Rule of thumb: finish the base project first. Only add an extension once the one before it is working, measured, and written up in the README. One project with real depth beats several half-finished ones.

Application day is Oct 15, 2026. Only put on the resume what is actually done and what I can explain in depth.

---

## Step 0: Base project (do this first)

- C++ image filtering tool (Gaussian blur, plus maybe a couple of other kernels like box blur or Sobel).
- Single-thread version first, then OpenMP parallel version (Visual Studio `/openmp`).
- Benchmark harness: same images, repeated runs, report the median. Put a timing table in the README.
- Make sure the output is correct before trusting any timing (compare single-thread vs multi-thread output pixel by pixel).

Done when: both versions run, outputs match, timings are in the README, repo is on GitHub.

---

## Extension 1: Performance engineering study (best fit, do first)

Turn "I added threads" into "I understand why it is fast or slow."

Layers to add, in this order:
1. Separable Gaussian (two 1D passes instead of one 2D pass). An algorithmic win before any parallelism.
2. Cache-aware tiling. Test different tile sizes and see how memory access patterns change the speed.
3. SIMD, either AVX2 intrinsics or checking what the compiler auto-vectorizes. Compare against the scalar version.
4. Scaling plot across thread counts (1, 2, 4, 8, ...). Explain where it flattens out using Amdahl's law and memory bandwidth.
5. Profile with the Visual Studio profiler and back up each claim with a measurement.

What it shows: computer architecture (caches, vector units, memory bandwidth), parallel computing, measurement discipline.

Stop condition: separable + OpenMP + one of (tiling or SIMD) with a scaling graph is already a strong result.

---

## Extension 2: CUDA port using a free cloud GPU (stretch goal)

No NVIDIA GPU needed. Google Colab and Kaggle notebooks give free NVIDIA GPUs and let me compile and run real CUDA C++.

Plan:
1. Naive CUDA kernel for the blur (one thread per pixel).
2. Shared-memory version (load a tile plus halo into shared memory).
3. Compare CPU single-thread vs OpenMP vs GPU on the same images, including transfer time, not just kernel time.

What it shows: direct CUDA experience, understanding of GPU memory hierarchy, honest benchmarking.

Only start this after Extension 1 has a solid result. Reuses the same algorithm and benchmark harness.

---

## Extension 3: Parallel primitives (reduction, prefix sum, histogram)

Implement each on CPU with OpenMP, and optionally in CUDA on Colab.

- Reduction: sum or max of an array.
- Prefix sum (scan): learn why work-efficient scan matters.
- Histogram: learn how to avoid race conditions (privatized per-thread histograms, then merge).

What it shows: the building blocks of GPU programming, which come up a lot in parallel computing interviews. Small enough to finish in a few days. A histogram of image pixel values ties in naturally with the filter project.

---

## Extension 4: Matrix multiply optimization ladder

Steps: naive triple loop, loop reordering, cache blocking, SIMD, OpenMP. Compare against a library such as Eigen at the end.

What it shows: how hardware actually executes code. Maps directly to what cuBLAS does on GPUs. Good interview story.

---

## Extension 5: PyTorch angle (uses my AI background)

- Write a 2D convolution by hand and verify it against `torch.nn.functional.conv2d`.
- Profile a small model with `torch.profiler` and look at where the time goes.
- Optional: compare my hand-written filter to a PyTorch convolution with the same kernel.

What it shows: PyTorch familiarity, connection between image filtering and CNN layers. Lighter on parallel computing.

---

## Extension 6: Cache simulator in C++

A small simulator that reads a memory access trace and reports hit/miss rates for different cache sizes, associativity, and block sizes.

What it shows: computer architecture understanding, plays to my EE background. Takes about 2 to 3 days. Pairs well with Extension 1, since it can help explain the tiling results.

---

## Small win: Linux basics with WSL2

- Install WSL2 on Windows 11 (under an hour). Keep Visual Studio for main development.
- Build and run the same code with g++ and CMake in WSL2.
- Mention in the README that the project builds on both Windows and Linux.

Covers the "basic knowledge of Linux" line in the posting.

---

## Suggested order

1. Base project (Step 0)
2. Extension 1 (separable, tiling or SIMD, scaling plot)
3. WSL2 build
4. Extension 2 (Colab CUDA port), if time allows
5. Anything else (3, 4, 5, 6) only after the above, and only if there is time before Oct 15

## README checklist

- What the tool does and how to build and run it
- Timing table (single-thread vs multi-thread, and any extensions)
- Scaling graph
- Short "what I learned" section explaining why results look the way they do
- Hardware used (CPU model, core count) so the numbers mean something
