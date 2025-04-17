# SPDX - FileCopyrightText : 2025 Advanced Micro Devices, Inc.
#
# SPDX - License - Identifier : Apache - 2.0

import os
from pathlib import Path


def get_include():
    """Return the directory containing the FlashInfer CUDA header files.

    Returns
    -------
    include_dir : str
        Path to flashinfer's CUDA header files.
    """
    root_dir = Path(__file__).parent.resolve()
    include_dir = os.path.join(root_dir, "include")
    return str(include_dir)


def get_cutlass_include():
    """Return the directories containing the vendored CUTLASS headers.

    Returns
    -------
    include_dirs : list
        List of paths to CUTLASS include directories.
    """
    root_dir = Path(__file__).parent.resolve()
    return [
        str(root_dir / "include" / "cutlass"),
        str(root_dir / "include" / "cutlass_tools"),
    ]


def get_tvm_binding_dir():
    """Return the directory containing the TVM binding files.

    Returns
    -------
    tvm_binding_dir : str
        Path to TVM binding files.
    """
    root_dir = Path(__file__).parent.resolve()
    tvm_binding_dir = os.path.join(root_dir, "tvm_binding")
    return str(tvm_binding_dir)


def get_csrc_dir():
    """Return the directory containing the C++/CUDA source files.

    Returns
    -------
    csrc_dir : str
        Path to flashinfer's C++ source files.
    """
    root_dir = Path(__file__).parent.resolve()
    csrc_dir = os.path.join(root_dir, "csrc")
    return str(csrc_dir)
