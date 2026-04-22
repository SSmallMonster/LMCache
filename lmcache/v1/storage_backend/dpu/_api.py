# SPDX-License-Identifier: Apache-2.0
"""DPU Cache ctypes API wrapper."""

# Standard
import ctypes
import ctypes.util
import logging
import os
from typing import Optional, Tuple

# Third Party
import torch

logger = logging.getLogger(__name__)

DPU_CACHE_SUCCESS = 0
DPU_CACHE_KEY_NOT_FOUND = -2
CUDA_MEMCPY_DEVICE_TO_HOST = 2
CUDA_MEMCPY_DEVICE_TO_DEVICE = 3

# Torch dtype到整数的映射
TORCH_DTYPE_MAP = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.int32: 3,
    torch.int64: 4,
    torch.uint8: 5,
}

REVERSE_TORCH_DTYPE_MAP = {v: k for k, v in TORCH_DTYPE_MAP.items()}


class DPUError(Exception):
    """DPU基础异常"""

    pass


class DPUConnectionError(DPUError):
    """DPU连接错误"""

    pass


class DPUOperationError(DPUError):
    """DPU操作错误"""

    pass


class DPUConfig:
    """DPU配置参数"""

    def __init__(
        self,
        dpu_ip: str = "192.168.100.2",
        host_pci_addr: str = "0000:ba:00.0",
        gpu_id: int = 0,
        max_concurrent_ops: int = 4,
        fallback_enabled: bool = True,
    ) -> None:
        self.dpu_ip = dpu_ip
        self.host_pci_addr = host_pci_addr
        self.gpu_id = gpu_id
        self.max_concurrent_ops = max_concurrent_ops
        self.fallback_enabled = fallback_enabled

    def __repr__(self) -> str:
        return (
            f"DPUConfig(dpu_ip='{self.dpu_ip}', "
            f"host_pci_addr='{self.host_pci_addr}', "
            f"gpu_id={self.gpu_id}, "
            f"max_concurrent_ops={self.max_concurrent_ops}, "
            f"fallback_enabled={self.fallback_enabled})"
        )


# C结构体定义
class DPUConfigStruct(ctypes.Structure):
    _fields_ = [
        ("dpu_ip", ctypes.c_char * 16),
        ("host_pci_addr", ctypes.c_char * 32),
        ("gpu_id", ctypes.c_int),
        ("max_concurrent_ops", ctypes.c_int),
        ("initialized", ctypes.c_int),
    ]


class DPUAgent:
    """DPU代理 - ctypes包装实现"""

    def __init__(self, config: DPUConfig) -> None:
        self.config = config
        self.lib = None
        self._cudart = None
        self._load_library()
        self._setup_function_signatures()
        self._initialize_dpu(config)

    def _load_library(self) -> None:
        """加载共享库"""
        # 获取当前文件的目录，然后寻找共享库
        current_dir = os.path.dirname(os.path.abspath(__file__))

        possible_paths = [
            # 相对于项目根目录的路径
            os.path.join(current_dir, "build/libdpu_cache.so"),
            os.path.join(current_dir, "build-host/libdpu_cache.so"),
            # 绝对路径
            "/usr/local/lib/libdpu_cache.so",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                try:
                    self.lib = ctypes.CDLL(path)
                    logger.info(f"Loaded DPU library from: {path}")
                    return
                except OSError as e:
                    logger.warning(f"Failed to load {path}: {e}")
                    continue

        raise DPUConnectionError("Could not find or load libdpu_cache.so")

    def _setup_function_signatures(self) -> None:
        """设置ctypes函数签名"""
        # dpu_cache_init
        self.lib.dpu_cache_init.argtypes = [ctypes.POINTER(DPUConfigStruct)]
        self.lib.dpu_cache_init.restype = ctypes.c_int

        # dpu_cache_store
        self.lib.dpu_cache_store.argtypes = [
            ctypes.c_char_p,                    # key_id
            ctypes.c_void_p,                    # k_data
            ctypes.c_size_t,                    # k_size
            ctypes.c_int,                       # k_dtype
            ctypes.POINTER(ctypes.c_int),       # k_shape
            ctypes.c_int,                       # k_ndim
            ctypes.c_void_p,                    # v_data
            ctypes.c_size_t,                    # v_size
            ctypes.c_int,                       # v_dtype
            ctypes.POINTER(ctypes.c_int),       # v_shape
            ctypes.c_int                        # v_ndim
        ]
        self.lib.dpu_cache_store.restype = ctypes.c_int

        # dpu_cache_retrieve
        self.lib.dpu_cache_retrieve.argtypes = [
            ctypes.c_char_p,                    # key_id
            ctypes.POINTER(ctypes.c_void_p),    # k_data
            ctypes.POINTER(ctypes.c_size_t),    # k_size
            ctypes.POINTER(ctypes.c_int),       # k_dtype
            ctypes.POINTER(ctypes.c_int),       # k_shape
            ctypes.POINTER(ctypes.c_int),       # k_ndim
            ctypes.POINTER(ctypes.c_void_p),    # v_data
            ctypes.POINTER(ctypes.c_size_t),    # v_size
            ctypes.POINTER(ctypes.c_int),       # v_dtype
            ctypes.POINTER(ctypes.c_int),       # v_shape
            ctypes.POINTER(ctypes.c_int)        # v_ndim
        ]
        self.lib.dpu_cache_retrieve.restype = ctypes.c_int

        # dpu_cache_free
        self.lib.dpu_cache_free.argtypes = [ctypes.c_void_p]
        self.lib.dpu_cache_free.restype = ctypes.c_int

        # dpu_cache_remove
        self.lib.dpu_cache_remove.argtypes = [ctypes.c_char_p]
        self.lib.dpu_cache_remove.restype = ctypes.c_int

        # dpu_cache_contains
        self.lib.dpu_cache_contains.argtypes = [ctypes.c_char_p]
        self.lib.dpu_cache_contains.restype = ctypes.c_int

        # dpu_cache_cleanup
        self.lib.dpu_cache_cleanup.argtypes = []
        self.lib.dpu_cache_cleanup.restype = ctypes.c_int

    def _initialize_dpu(self, config: DPUConfig) -> None:
        """初始化DPU配置"""
        c_config = DPUConfigStruct()
        c_config.dpu_ip = config.dpu_ip.encode("utf-8")
        c_config.host_pci_addr = config.host_pci_addr.encode("utf-8")
        c_config.gpu_id = config.gpu_id
        c_config.max_concurrent_ops = config.max_concurrent_ops
        c_config.initialized = 0

        result = self.lib.dpu_cache_init(ctypes.byref(c_config))
        if result != 0:
            raise DPUConnectionError(f"Failed to initialize DPU: error code {result}")

        logger.info("DPU Agent initialized successfully")

    def _torch_dtype_to_int(self, dtype: torch.dtype) -> int:
        """转换torch dtype到整数"""
        return TORCH_DTYPE_MAP.get(dtype, 0)

    def _int_to_torch_dtype(self, dtype_int: int) -> torch.dtype:
        """转换整数到torch dtype"""
        return REVERSE_TORCH_DTYPE_MAP.get(dtype_int, torch.float32)

    def store_kv(
        self, key_id: str, k_tensor: torch.Tensor, v_tensor: torch.Tensor
    ) -> bool:
        """存储KV张量到DPU"""
        try:
            # 验证tensor在GPU上
            # if not k_tensor.is_cuda or not v_tensor.is_cuda:
            #     raise ValueError("Tensors must be on CUDA device")

            # # 验证tensor在同一设备
            # if k_tensor.device != v_tensor.device:
            #     raise ValueError("K and V tensors must be on same device")

            # 验证tensor数据类型一致
            if k_tensor.dtype != v_tensor.dtype:
                raise ValueError("K and V tensors must have same dtype")
            if k_tensor.dim() > 4 or v_tensor.dim() > 4:
                raise ValueError(
                    "DPU cache header supports at most 4 tensor dimensions"
                )

            # 验证tensor连续性
            if not k_tensor.is_contiguous():
                k_tensor = k_tensor.contiguous()
            if not v_tensor.is_contiguous():
                v_tensor = v_tensor.contiguous()

            # 获取tensor数据指针
            k_ptr = k_tensor.data_ptr()
            v_ptr = v_tensor.data_ptr()

            # 准备shape数组
            k_shape = list(k_tensor.shape) + [0] * (4 - len(k_tensor.shape))
            v_shape = list(v_tensor.shape) + [0] * (4 - len(v_tensor.shape))

            k_shape_array = (ctypes.c_int * 4)(*k_shape[:4])
            v_shape_array = (ctypes.c_int * 4)(*v_shape[:4])

            # 调用C函数
            result = self.lib.dpu_cache_store(
                key_id.encode("utf-8"),
                k_ptr,
                k_tensor.nbytes,
                self._torch_dtype_to_int(k_tensor.dtype),
                k_shape_array,
                len(k_tensor.shape),
                v_ptr,
                v_tensor.nbytes,
                self._torch_dtype_to_int(v_tensor.dtype),
                v_shape_array,
                len(v_tensor.shape),
            )

            if result == DPU_CACHE_SUCCESS:
                logger.debug(f"Successfully stored KV cache for key: {key_id}")
                return True

            logger.error(f"Failed to store KV cache for key {key_id}, error: {result}")
            return False

        except Exception as e:
            logger.error(f"Exception in store_kv for key {key_id}: {e}")
            raise DPUOperationError(f"Failed to store KV cache: {e}") from e

    def retrieve_kv(
        self, key_id: str, target_device: str = "cuda"
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """从DPU检索KV张量"""
        k_data_ptr = ctypes.c_void_p()
        v_data_ptr = ctypes.c_void_p()
        try:
            # 准备输出参数
            k_size = ctypes.c_size_t()
            v_size = ctypes.c_size_t()
            k_dtype = ctypes.c_int()
            v_dtype = ctypes.c_int()
            k_shape = (ctypes.c_int * 4)()
            v_shape = (ctypes.c_int * 4)()
            k_ndim = ctypes.c_int()
            v_ndim = ctypes.c_int()

            # 调用C函数
            result = self.lib.dpu_cache_retrieve(
                key_id.encode("utf-8"),
                ctypes.byref(k_data_ptr),
                ctypes.byref(k_size),
                ctypes.byref(k_dtype),
                k_shape,
                ctypes.byref(k_ndim),
                ctypes.byref(v_data_ptr),
                ctypes.byref(v_size),
                ctypes.byref(v_dtype),
                v_shape,
                ctypes.byref(v_ndim),
            )

            if result == DPU_CACHE_KEY_NOT_FOUND:
                logger.debug(f"Key not found: {key_id}")
                return None
            if result != DPU_CACHE_SUCCESS:
                logger.error(
                    "Failed to retrieve KV cache for key %s, error: %s",
                    key_id,
                    result,
                )
                return None

            return (
                self._copy_retrieved_tensor(
                    k_data_ptr,
                    k_size.value,
                    k_dtype.value,
                    k_shape,
                    k_ndim.value,
                    target_device,
                ),
                self._copy_retrieved_tensor(
                    v_data_ptr,
                    v_size.value,
                    v_dtype.value,
                    v_shape,
                    v_ndim.value,
                    target_device,
                ),
            )

        except Exception as e:
            logger.error(f"Exception in retrieve_kv for key {key_id}: {e}")
            raise DPUOperationError(f"Failed to retrieve KV cache: {e}") from e
        finally:
            self._free_retrieved_ptr(k_data_ptr)
            self._free_retrieved_ptr(v_data_ptr)

    def remove_kv(self, key_id: str) -> bool:
        """从DPU删除KV缓存"""
        try:
            result = self.lib.dpu_cache_remove(key_id.encode("utf-8"))
            if result == 0:
                logger.debug(f"Successfully removed KV cache for key: {key_id}")
                return True

            logger.warning(
                "Failed to remove KV cache for key %s, error: %s", key_id, result
            )
            return False

        except Exception as e:
            logger.error(f"Exception in remove_kv for key {key_id}: {e}")
            return False

    def contains_key(self, key_id: str) -> bool:
        """检查DPU是否包含指定key"""
        try:
            result = self.lib.dpu_cache_contains(key_id.encode("utf-8"))
            return result == 0

        except Exception as e:
            logger.error(f"Exception in contains_key for key {key_id}: {e}")
            return False

    def __del__(self) -> None:
        """清理资源"""
        if self.lib:
            try:
                self.lib.dpu_cache_cleanup()
            except Exception:
                pass

    def store_kv_cache(
        self, key_id: str, k_tensor: torch.Tensor, v_tensor: torch.Tensor
    ) -> bool:
        """Compatibility alias for older DPU wrapper call sites."""
        return self.store_kv(key_id, k_tensor, v_tensor)

    def retrieve_kv_cache(
        self, key_id: str, target_device: str = "cuda"
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Compatibility alias for older DPU wrapper call sites."""
        return self.retrieve_kv(key_id, target_device)

    def remove_key(self, key_id: str) -> bool:
        """Compatibility alias for older DPU wrapper call sites."""
        return self.remove_kv(key_id)

    def _copy_retrieved_tensor(
        self,
        data_ptr: ctypes.c_void_p,
        size_bytes: int,
        dtype_int: int,
        shape_array: ctypes.Array,
        ndim: int,
        target_device: str,
    ) -> torch.Tensor:
        """Copy a C-owned CUDA buffer into a PyTorch-owned tensor."""
        if ndim < 0 or ndim > 4:
            raise DPUOperationError(f"Invalid tensor rank returned by DPU: {ndim}")
        if size_bytes > 0 and not data_ptr.value:
            raise DPUOperationError("DPU returned a null tensor pointer")

        dtype = self._int_to_torch_dtype(dtype_int)
        shape = tuple(int(shape_array[i]) for i in range(ndim))
        tensor = torch.empty(shape, dtype=dtype, device=torch.device(target_device))
        if tensor.nbytes != size_bytes:
            raise DPUOperationError(
                f"DPU metadata size mismatch: shape={shape}, dtype={dtype}, "
                f"tensor bytes={tensor.nbytes}, returned bytes={size_bytes}"
            )
        if size_bytes == 0:
            return tensor

        copy_kind = (
            CUDA_MEMCPY_DEVICE_TO_DEVICE
            if tensor.device.type == "cuda"
            else CUDA_MEMCPY_DEVICE_TO_HOST
        )
        self._cuda_memcpy(tensor.data_ptr(), data_ptr.value, size_bytes, copy_kind)
        return tensor

    def _cuda_memcpy(
        self, dst_ptr: int, src_ptr: int, size_bytes: int, copy_kind: int
    ) -> None:
        """Copy CUDA memory through libcudart instead of PyTorch private bindings."""
        cudart = self._load_cuda_runtime()
        cuda_error = cudart.cudaMemcpy(
            ctypes.c_void_p(dst_ptr),
            ctypes.c_void_p(src_ptr),
            ctypes.c_size_t(size_bytes),
            ctypes.c_int(copy_kind),
        )
        if cuda_error != 0:
            error_string = cudart.cudaGetErrorString(cuda_error)
            error_message = (
                error_string.decode("utf-8")
                if error_string is not None
                else "unknown CUDA error"
            )
            raise DPUOperationError(
                f"cudaMemcpy failed with error code {cuda_error}: {error_message}"
            )

    def _load_cuda_runtime(self) -> ctypes.CDLL:
        """Load libcudart and configure the CUDA runtime functions used here."""
        if self._cudart is not None:
            return self._cudart

        candidates = [
            ctypes.util.find_library("cudart"),
            "libcudart.so",
            "libcudart.so.12",
            "libcudart.so.11.0",
        ]

        for candidate in candidates:
            if candidate is None:
                continue
            try:
                cudart = ctypes.CDLL(candidate)
                cudart.cudaMemcpy.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.c_int,
                ]
                cudart.cudaMemcpy.restype = ctypes.c_int
                cudart.cudaGetErrorString.argtypes = [ctypes.c_int]
                cudart.cudaGetErrorString.restype = ctypes.c_char_p
                self._cudart = cudart
                return cudart
            except OSError:
                continue

        raise DPUOperationError("Could not load libcudart for cudaMemcpy")

    def _free_retrieved_ptr(self, data_ptr: ctypes.c_void_p) -> None:
        """Release a CUDA buffer allocated by dpu_cache_retrieve."""
        if self.lib is None or not data_ptr.value:
            return

        result = self.lib.dpu_cache_free(data_ptr)
        if result != DPU_CACHE_SUCCESS:
            logger.warning("Failed to free DPU retrieve buffer: error %s", result)


# 兼容agent_wrapper.py的接口
def dpu_agent(config: DPUConfig) -> DPUAgent:
    """创建DPU Agent实例"""
    return DPUAgent(config)


def dpu_agent_config(**kwargs) -> DPUConfig:
    """创建DPU Agent配置"""
    return DPUConfig(**kwargs)
