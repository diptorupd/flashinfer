"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import ctypes
import importlib.util
import os
from typing import Optional, Set

from .. import __config__

# Re-export
from .activation import gen_act_and_mul_module as gen_act_and_mul_module
from .activation import get_act_and_mul_cu_str as get_act_and_mul_cu_str
from .attention import (
    gen_batch_decode_mla_module as gen_batch_decode_mla_module,
)
from .attention import gen_batch_decode_module as gen_batch_decode_module
from .attention import gen_batch_mla_module as gen_batch_mla_module
from .attention import gen_batch_mla_tvm_binding as gen_batch_mla_tvm_binding
from .attention import gen_batch_prefill_module as gen_batch_prefill_module
from .attention import (
    gen_customize_batch_decode_module as gen_customize_batch_decode_module,
)
from .attention import (
    gen_customize_batch_decode_tvm_binding as gen_customize_batch_decode_tvm_binding,
)
from .attention import (
    gen_customize_batch_prefill_module as gen_customize_batch_prefill_module,
)
from .attention import (
    gen_customize_batch_prefill_tvm_binding as gen_customize_batch_prefill_tvm_binding,
)
from .attention import (
    gen_customize_single_decode_module as gen_customize_single_decode_module,
)
from .attention import (
    gen_customize_single_prefill_module as gen_customize_single_prefill_module,
)
from .attention import gen_pod_module as gen_pod_module
from .attention import gen_sampling_tvm_binding as gen_sampling_tvm_binding
from .attention import gen_single_decode_module as gen_single_decode_module
from .attention import gen_single_prefill_module as gen_single_prefill_module
from .attention import get_batch_decode_mla_uri as get_batch_decode_mla_uri
from .attention import get_batch_decode_uri as get_batch_decode_uri
from .attention import get_batch_mla_uri as get_batch_mla_uri
from .attention import get_batch_prefill_uri as get_batch_prefill_uri
from .attention import get_pod_uri as get_pod_uri
from .attention import get_single_decode_uri as get_single_decode_uri
from .attention import get_single_prefill_uri as get_single_prefill_uri
from .core import clear_cache_dir, load_cuda_ops  # noqa: F401
from .env import *
from .utils import parallel_load_modules as parallel_load_modules


def _get_extension_path(name: str) -> Optional[str]:
    """Try to find installed extension module"""
    try:
        spec = importlib.util.find_spec(name)
        if spec and spec.origin:
            return spec.origin
        return None
    except (ImportError, ModuleNotFoundError):
        return None


# Standard extensions
prebuilt_ops_uri: Set[str] = set()

# noqa: F401
has_prebuilt_ops = False
from .core import logger

# Try and Except to break circular dependencies
try:
    from .. import __config__

    if __config__.get_info("aot_torch_exts_cuda"):
        try:
            from .. import flashinfer_kernels

            has_prebuilt_ops = True
            kernels_path = _get_extension_path("flashinfer.flashinfer_kernels")
            if kernels_path:
                prebuilt_ops_uri.add(kernels_path)
        except ImportError:
            logger.warning(
                "CUDA kernels were enabled in build but couldn't be imported"
            )

        # Only try to import SM90 kernels if they were enabled during build
        if 90 in __config__.get_info("aot_torch_exts_cuda_archs"):
            try:
                from .. import flashinfer_kernels_sm90  # noqa: F401

                has_prebuilt_ops = True
                kernels_sm90_path = _get_extension_path(
                    "flashinfer.flashinfer_kernels_sm90"
                )
                if kernels_sm90_path:
                    prebuilt_ops_uri.add(kernels_sm90_path)
            except ImportError:
                logger.warning(
                    "SM90 kernels were enabled in build but couldn't be imported"
                )

    if __config__.get_info("aot_torch_exts_hip"):
        try:
            from .. import _flashinfer_hip_kernels  # noqa: F401

            has_prebuilt_ops = True

            kernels_hip_path = _get_extension_path(
                "flashinfer._flashinfer_hip_kernels.abi3"
            )
            if kernels_hip_path:
                prebuilt_ops_uri.add(kernels_hip_path)
        except ImportError as e:
            print(e)
            logger.warning("HIP kernels were enabled in build but couldn't be imported")

except ImportError:
    pass

if not has_prebuilt_ops:
    logger.info("Prebuilt kernels not found, using JIT backend")
