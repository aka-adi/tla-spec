# 同步接口规格

[返回接口目录](../README.md)

<a name="api-tla-sync-block-set-flag"></a>

## tla.sync.block_set_flag

签名：`tla.sync.block_set_flag(mode, pipe, flagid)`

返回：None。

编译表示：hivm.hir.sync_block_set / hivm.hir.sync_intra_block_set。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `mode` | 选择操作模式 | 编译期 tla.sync.MODE | INTER / SUBBLOCK / BLOCK / INTRA | 前三项为逻辑域，INTRA 只操作单个物理计数器。 | 必填 |
| `pipe` | 同步关联的流水线 | 编译期 tla.sync.PIPE | S / V / M / MTE1 / MTE2 / MTE3 / F | Vector：S / V / MTE2 / MTE3。Cube：S / M / MTE1 / MTE2 / MTE3 / F。 | 必填 |
| `flagid` | 同步标志编号 | int / 标量整数 tl.tensor | 0 至 15。INTRA 在 Cube 上可到 31。 | 运行时整数转换为 int64，由调用者保证范围；bool 不合法。 | 必填 |

**调用示例**

Vector 写入 L1 后发出标志，Cube 等待同一个标志后读取。

```python
@tla.jit
def handoff(src_ub, dst_l1, acc_l0c, rhs_l1):
    with tla.scope.vector():
        if tla.sub_block_id() == 0:
            tla.copy_ub_to_l1(src_ub, dst_l1)
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 0)
    with tla.scope.cube():
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 0)
        tla.cube.mmad(dst_l1, rhs_l1, acc_l0c, clear_accum=True)
```

仓库测试：[tla_ut/dma/nd2nz.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dma/nd2nz.py#L24)。

<a name="api-tla-sync-block-wait-flag"></a>

## tla.sync.block_wait_flag

签名：`tla.sync.block_wait_flag(mode, pipe, flagid)`

返回：None。

编译表示：hivm.hir.sync_block_wait / hivm.hir.sync_intra_block_wait。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `mode` | 选择操作模式 | 编译期 tla.sync.MODE | INTER / SUBBLOCK / BLOCK / INTRA | 前三项为逻辑域，INTRA 只操作单个物理计数器。 | 必填 |
| `pipe` | 同步关联的流水线 | 编译期 tla.sync.PIPE | S / V / M / MTE1 / MTE2 / MTE3 / F | Vector：S / V / MTE2 / MTE3。Cube：S / M / MTE1 / MTE2 / MTE3 / F。 | 必填 |
| `flagid` | 同步标志编号 | int / 标量整数 tl.tensor | 0 至 15。INTRA 在 Cube 上可到 31。 | 运行时整数转换为 int64，由调用者保证范围；bool 不合法。 | 必填 |

**调用示例**

Vector 写入 L1 后发出标志，Cube 等待同一个标志后读取。

```python
@tla.jit
def handoff(src_ub, dst_l1, acc_l0c, rhs_l1):
    with tla.scope.vector():
        if tla.sub_block_id() == 0:
            tla.copy_ub_to_l1(src_ub, dst_l1)
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 0)
    with tla.scope.cube():
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 0)
        tla.cube.mmad(dst_l1, rhs_l1, acc_l0c, clear_accum=True)
```

仓库测试：[tla_ut/dma/nd2nz.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dma/nd2nz.py#L26)。

<a name="api-tla-sync-mutex-lock"></a>

## tla.sync.mutex_lock

签名：`tla.sync.mutex_lock(bufid, pipe, mode=False)`

返回：None。

编译表示：hivm.hir.get_buf。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `bufid` | 缓冲区同步令牌 ID | int / 标量整数 tl.tensor | 0 至 31 | 运行时整数转换为 int64，由调用者保证范围；bool 不合法。 | 必填 |
| `pipe` | 同步关联的流水线 | 编译期 tla.sync.PIPE | S / V / M / MTE1 / MTE2 / MTE3 / F | Vector：S / V / MTE2 / MTE3。Cube：S / M / MTE1 / MTE2 / MTE3 / F。 | 必填 |
| `mode` | 选择操作模式 | 编译期 bool | False / True | False 阻塞，True 非阻塞。 | False |

**调用示例**

```python
@tla.jit
def example():
    with tla.scope.vector():
        tla.sync.mutex_lock(0, tla.sync.PIPE.V)
        tla.sync.mutex_unlock(0, tla.sync.PIPE.V)
```

仓库测试：[tla_ut/sync/buffer_sync.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/sync/buffer_sync.py#L38)。

<a name="api-tla-sync-barrier"></a>

## tla.sync.barrier

签名：`tla.sync.barrier(pipe)`

返回：None。

编译表示：hivm.hir.pipe_barrier。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `pipe` | 同步关联的流水线 | 编译期 tla.sync.PIPE | S / V / M / MTE1 / MTE2 / MTE3 / F / ALL | 按核心的流水线使用域配对。 | 必填 |

**调用示例**

```python
@tla.jit
def example():
    with tla.scope.vector():
        tla.sync.barrier(tla.sync.PIPE.ALL)
```

仓库测试：[tla_ut/dma/set_atomic.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dma/set_atomic.py#L34)。

<a name="api-tla-sync-mutex-unlock"></a>

## tla.sync.mutex_unlock

签名：`tla.sync.mutex_unlock(bufid, pipe, mode=False)`

返回：None。

编译表示：hivm.hir.rls_buf。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `bufid` | 缓冲区同步令牌 ID | int / 标量整数 tl.tensor | 0 至 31 | 运行时整数转换为 int64，由调用者保证范围；bool 不合法。 | 必填 |
| `pipe` | 同步关联的流水线 | 编译期 tla.sync.PIPE | S / V / M / MTE1 / MTE2 / MTE3 / F | Vector：S / V / MTE2 / MTE3。Cube：S / M / MTE1 / MTE2 / MTE3 / F。 | 必填 |
| `mode` | 选择操作模式 | 编译期 bool | False / True | False 阻塞，True 非阻塞。 | False |

**调用示例**

```python
@tla.jit
def example():
    with tla.scope.vector():
        tla.sync.mutex_lock(0, tla.sync.PIPE.V)
        tla.sync.mutex_unlock(0, tla.sync.PIPE.V)
```

仓库测试：[tla_ut/sync/buffer_sync.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/sync/buffer_sync.py#L41)。

<a name="api-tla-sync-set-flag"></a>

## tla.sync.set_flag

签名：`tla.sync.set_flag(pipe1, pipe2, event_id)`

返回：None。

编译表示：hivm.hir.set_flag。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `pipe1` | 生产者流水线 | 编译期 tla.sync.PIPE | S / V / M / MTE1 / MTE2 / MTE3 / F | 按核心的流水线使用域配对。 | 必填 |
| `pipe2` | 消费者流水线 | 编译期 tla.sync.PIPE | S / V / M / MTE1 / MTE2 / MTE3 / F | 按核心的流水线使用域配对。 | 必填 |
| `event_id` | 配对事件编号 | int / 标量整数 tl.tensor | 0 至 7 | 运行时整数转换为 int64，由调用者保证范围；bool 不合法。 | 必填 |

**调用示例**

```python
@tla.jit
def example():
    with tla.scope.vector():
        tla.sync.set_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.V, 0)
        tla.sync.wait_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.V, 0)
```

仓库测试：[tla_ut/sync/buffer_sync.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/sync/buffer_sync.py#L42)。

<a name="api-tla-sync-wait-flag"></a>

## tla.sync.wait_flag

签名：`tla.sync.wait_flag(pipe1, pipe2, event_id)`

返回：None。

编译表示：hivm.hir.wait_flag。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `pipe1` | 生产者流水线 | 编译期 tla.sync.PIPE | S / V / M / MTE1 / MTE2 / MTE3 / F | 按核心的流水线使用域配对。 | 必填 |
| `pipe2` | 消费者流水线 | 编译期 tla.sync.PIPE | S / V / M / MTE1 / MTE2 / MTE3 / F | 按核心的流水线使用域配对。 | 必填 |
| `event_id` | 配对事件编号 | int / 标量整数 tl.tensor | 0 至 7 | 运行时整数转换为 int64，由调用者保证范围；bool 不合法。 | 必填 |

**调用示例**

```python
@tla.jit
def example():
    with tla.scope.vector():
        tla.sync.set_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.V, 0)
        tla.sync.wait_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.V, 0)
```

仓库测试：[tla_ut/sync/buffer_sync.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/sync/buffer_sync.py#L43)。

<a name="api-tla-sync-sync-all"></a>

## tla.sync.sync_all

签名：`tla.sync.sync_all(mode)`

返回：None。

编译表示：hivm.hir.sync_block / hivm.hir.pipe_barrier / hivm.hir.sync_block_set。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `mode` | 选择操作模式 | 编译期 bool | False / True | False 同步 AIV，True 同步 AIC/AIV。必须在 VF 外。 | 必填 |

**调用示例**

同步 AIV 核，在 SIMD/SIMT 区域之外调用。

```python
@tla.jit
def example():
    tla.sync.sync_all(False)
```

仓库测试：[tla_ut/sync/sync_pipe.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/sync/sync_pipe.py#L65)。
