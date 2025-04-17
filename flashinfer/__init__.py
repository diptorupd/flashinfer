"""
Copyright (c) 2023 by FlashInfer team.

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

"""
FlashInfer: Fast Attention Algorithms for LLM Inference
"""
import os
from pathlib import Path

try:
    from .__config__ import __version__, get_info, show
except ImportError:
    # Fallback for development mode
    version = "0.0.0+dev"

    def get_info(name=None):
        info = {"version": version}
        if name is None:
            return info
        return info.get(name)

    def show():
        print("FlashInfer development configuration")
        print(f"version: {version}")


from .activation import gelu_and_mul as gelu_and_mul
from .activation import gelu_tanh_and_mul as gelu_tanh_and_mul
from .activation import silu_and_mul as silu_and_mul
from .cascade import (
    BatchDecodeWithSharedPrefixPagedKVCacheWrapper as BatchDecodeWithSharedPrefixPagedKVCacheWrapper,
)
from .cascade import (
    BatchPrefillWithSharedPrefixPagedKVCacheWrapper as BatchPrefillWithSharedPrefixPagedKVCacheWrapper,
)
from .cascade import (
    MultiLevelCascadeAttentionWrapper as MultiLevelCascadeAttentionWrapper,
)
from .cascade import merge_state as merge_state
from .cascade import merge_state_in_place as merge_state_in_place
from .cascade import merge_states as merge_states
from .decode import (
    BatchDecodeWithPagedKVCacheWrapper as BatchDecodeWithPagedKVCacheWrapper,
)
from .decode import (
    CUDAGraphBatchDecodeWithPagedKVCacheWrapper as CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
)
from .decode import single_decode_with_kv_cache as single_decode_with_kv_cache
from .gemm import SegmentGEMMWrapper as SegmentGEMMWrapper
from .gemm import bmm_fp8 as bmm_fp8
from .mla import BatchMLAPagedAttentionWrapper as BatchMLAPagedAttentionWrapper
from .norm import fused_add_rmsnorm as fused_add_rmsnorm
from .norm import gemma_fused_add_rmsnorm as gemma_fused_add_rmsnorm
from .norm import gemma_rmsnorm as gemma_rmsnorm
from .norm import rmsnorm as rmsnorm
from .page import append_paged_kv_cache as append_paged_kv_cache
from .page import append_paged_mla_kv_cache as append_paged_mla_kv_cache
from .page import get_batch_indices_positions as get_batch_indices_positions
from .page import get_seq_lens as get_seq_lens
from .pod import PODWithPagedKVCacheWrapper as PODWithPagedKVCacheWrapper
from .prefill import (
    BatchPrefillWithPagedKVCacheWrapper as BatchPrefillWithPagedKVCacheWrapper,
)
from .prefill import (
    BatchPrefillWithRaggedKVCacheWrapper as BatchPrefillWithRaggedKVCacheWrapper,
)
from .prefill import (
    single_prefill_with_kv_cache as single_prefill_with_kv_cache,
)
from .prefill import (
    single_prefill_with_kv_cache_return_lse as single_prefill_with_kv_cache_return_lse,
)
from .quantization import packbits as packbits
from .quantization import segment_packbits as segment_packbits
from .rope import apply_llama31_rope as apply_llama31_rope
from .rope import apply_llama31_rope_inplace as apply_llama31_rope_inplace
from .rope import apply_llama31_rope_pos_ids as apply_llama31_rope_pos_ids
from .rope import (
    apply_llama31_rope_pos_ids_inplace as apply_llama31_rope_pos_ids_inplace,
)
from .rope import apply_rope as apply_rope
from .rope import apply_rope_inplace as apply_rope_inplace
from .rope import apply_rope_pos_ids as apply_rope_pos_ids
from .rope import apply_rope_pos_ids_inplace as apply_rope_pos_ids_inplace
from .rope import apply_rope_with_cos_sin_cache as apply_rope_with_cos_sin_cache
from .rope import (
    apply_rope_with_cos_sin_cache_inplace as apply_rope_with_cos_sin_cache_inplace,
)
from .sampling import chain_speculative_sampling as chain_speculative_sampling
from .sampling import min_p_sampling_from_probs as min_p_sampling_from_probs
from .sampling import sampling_from_probs as sampling_from_probs
from .sampling import top_k_mask_logits as top_k_mask_logits
from .sampling import top_k_renorm_probs as top_k_renorm_probs
from .sampling import top_k_sampling_from_probs as top_k_sampling_from_probs
from .sampling import (
    top_k_top_p_sampling_from_logits as top_k_top_p_sampling_from_logits,
)
from .sampling import (
    top_k_top_p_sampling_from_probs as top_k_top_p_sampling_from_probs,
)
from .sampling import top_p_renorm_probs as top_p_renorm_probs
from .sampling import top_p_sampling_from_probs as top_p_sampling_from_probs
from .sparse import BlockSparseAttentionWrapper as BlockSparseAttentionWrapper


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
