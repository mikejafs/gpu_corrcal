# Why `cpp/` is separate from `src/`

This folder contains CUDA/C++ code that must be **compiled** (e.g., with `nvcc`/CMake) into a shared library (`.so`). It is kept separate from `src/` to maintain a clean separation of concerns:

- `src/` → Python package (importable, no compilation)
- `cpp/` → CUDA/C++ backend (compiled, not directly importable)

## Why not put C++ in `src/`?

- Different build systems: Python uses `pip/setuptools`, C++ uses `nvcc`/CMake
- Avoids packaging issues when installing the Python package
- Keeps a clear mental model: Python interface vs compiled backend
- Makes iteration on kernels and builds easier

## Typical workflow

```text
cpp/ → build → build/libgpu_corrcal.so
                    ↓
src/ → loads via ctypes

## What is `CMakeLists.txt`?

`CMakeLists.txt` defines **how the CUDA/C++ code in this folder is built** into a shared library (e.g. `.so`) that can be used by Python.

Instead of manually writing long `nvcc` compile commands, CMake provides a clean, reproducible way to specify:
- which source files to compile
- which libraries to link (e.g. `cublas`)
- what type of output to produce (shared library vs executable)

## Typical workflow

```text
cpp/ → CMakeLists.txt → build → libgpu_corrcal.so
                                 ↓
                         loaded by Python (ctypes)


                         