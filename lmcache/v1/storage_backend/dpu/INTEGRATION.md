# DPU Cache Python包集成说明

## 构建和安装

### 构建 DPU Server

> 这一步是为了构建并运行一个在 DPU 服务器的接收端，用来接收 LMCache 发送的缓存数据

```shell
cd ~/lmcache/v1/storage_backend/dpu && bash scripts/build_dpu.sh
```

如果构建顺利的话，DPU Server 会出现在 `build-dpu/dpu_dma_copy`

启动 DPU Server:

```shell
[root@r6kd-2g2 dpu]# ./build-dpu/dpu_dma_copy -p 0000:03:00.1 -T
Starting DPU DMA server:
  PCI Address: 0000:03:00.1
  Mode: TCP
  TCP Port: 18517

===========================================
  Complete DPU DMA Server                 
===========================================
[DPU] PCI: 0000:03:00.1
[DPU] Stage buffer: 256 MiB
[DPU] Max DMA chunk supported by device: 2097152 bytes
[DPU] Using chunk size: 2097152 bytes, queue depth: 4
[TCP] Listening on 0.0.0.0:18517
[DPU] Waiting for TCP client on port 18517
```

> 可以通过 `doca_caps --list_devs` 来查找能够使用的 DOCA 设备

### 构建 LMCache 插件

> 这步是为了构建 DMA 动态链接库以及 DMA Python 接口库还有 DPUStorageBackend 插件

#### 构建 DMA 动态库 & Python 接口

> 这两个步骤被放在同一个脚本里面实现了，默认都会一起构建

```shell
cd ~/lmcache/v1/storage_backend/dpu && bash build_python_package.sh
```

构建顺利的话，会创建一个 `/usr/local/lib/libdpu_cache.so` 动态链接库以及 `dpu_cache` 库，

可以通过下面命令来测试是否创建成功：

```shell
python3 -c "import dpu_cache; print('module_path:', dpu_cache.__path__)"
```

#### 构建 DPUStorageBackend 插件

> 因为这个插件是新增的，必须先安装 LMCache 开发模式，要不然动态链接不到

```shell
cd ~ && pip install -e . --no-build-isolation --no-deps
```

通过下面命令来测试是否安装成功：

```shell
python3 -c "import lmcache.v1.storage_backend.dpu_storage_backend as dpu_storage_backend; print('module_name:', dpu_storage_backend.__name__)"
```

到这边为止，所有的依赖已经安装成功，下面是如何通过 LMCache 来使用 DPUStorageBackend。

## 使用 DPUStorageBackend

1. 创建 `lmcache_dpu.yaml`

```yaml
chunk_size: 64
local_cpu: False
save_unfull_chunk: True
storage_plugins: lmc_dpu_storage_backend
extra_config:
  storage_plugin.lmc_dpu_storage_backend.module_path: lmcache.v1.storage_backend.dpu_storage_backend
  storage_plugin.lmc_dpu_storage_backend.class_name: DPUStorageBackend
  # DPU configuration parameters
  host_pci_addr: "1a:00.1"
  dpu_ip: "10.75.70.128"
  max_concurrent_ops: 16
  gpu_id: 3
```

2. 启动 vLLM

```shell
CUDA_VISIBLE_DEVICES=3 LMCACHE_CONFIG_FILE=/root/mmzhou/lmcache_dpu.yaml vllm serve /root/models/Qwen3-0.6B --port 7990 --kv-transfer-config '{"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}'
```

读写日志：
```shell
Successfully stored KV cache for key: CacheEngineKey(model_name='/root/models/Qwen3-0.6B', world_size=1, worker_id=0, chunk_hash=7087119673989932035, dtype=torch.bfloat16, request_configs=None, tags=None, _dtype_str='bfloat16')
[DPU_CACHE] Generated safe file path: /tmp/kv_cache_8fa70b5b.bin (from key: CacheEngineKey(model_name='/root/models/Qwen3-0.6B...)
K tensor copy: CPU -> GPU (Host-to-Device)
V tensor copy: CPU -> GPU (Host-to-Device)
[DPU_CACHE] perform_dma_push called: gpu_data=0x7f6827400000, size=7340112, path=/tmp/kv_cache_8fa70b5b.bin
[DPU_CACHE] Performing real DMA push: size=7340112 bytes to /tmp/kv_cache_8fa70b5b.bin
[HOST] Starting real DMA push: 7340112 bytes to /tmp/kv_cache_8fa70b5b.bin
[HOST] Sending DMA request: req_id=1605908235, size=7340112
[HOST] Received response: status=0, error_code=0
[HOST] Push complete: 7340112 bytes to /tmp/kv_cache_8fa70b5b.bin
[HOST] End-to-end: 0.003988 sec, 1.84 GB/s
[HOST] DPU DMA only: 0.000378 sec, 19.42 GB/s
[DPU_CACHE SUCCESS] DMA push completed successfully
```