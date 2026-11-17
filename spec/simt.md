# SIMT接口规格

[返回接口目录](../README.md)

<a name="api-tla-scope-simt"></a>

## tla.scope.simt

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`tla.scope.simt(num_warps=32, super_block_factor=1):`

<a name="api-tl-reduce"></a>

## tl.reduce

签名：`tl.reduce(input, axis, combine_fn, keep_dims=False)`

返回：规约结果；多输入时返回对应的结果元组。

沿用 Triton 入口，适用场景仍受 TLA 编译路径和目标约束。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `input` | 规约输入 | tl.tensor 或 tuple[tl.tensor, ...] | 各输入 shape 一致 | — | 必填 |
| `axis` | 规约轴 | 编译期 int / None | 有效轴或 None | None 规约整个输入。 | 必填 |
| `combine_fn` | 合并两个部分结果 | Triton JIT 函数 | 与输入元素对应的签名 | 并行规约要求结合律。 | 必填 |
| `keep_dims` | 保留被规约的轴 | 编译期 bool | False / True | — | False |

**调用示例**

combine_fn 为调用者提供的 Triton JIT 二元规约函数，须满足结合律。

```python
@tla.jit
def example(x, combine_fn: tl.constexpr):
    values = tl.load(x + tl.arange(0, 64))
    result = tl.reduce(values, 0, combine_fn)
```

<a name="api-tl-atomic-add"></a>

## tl.atomic_add

签名：`tl.atomic_add(pointer, val, mask=None, sem=None, scope=None)`

返回：原子更新前的元素值。

沿用 Triton 入口，适用场景仍受 TLA 编译路径和目标约束。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `pointer` | GM 目标指针 | tl.tensor（指针或指针块） | 地址空间：GM | 目标支持的原子签名单独核对。 | 必填 |
| `val` | 加数 | tl.tensor / 标量 | 可转换为指针元素 dtype | 须能广播到地址形状。 | 必填 |
| `mask` | 控制是否访问 | tl.tensor（int1）/ None | 数据类型：int1 | 须能广播到地址形状。 | None |
| `sem` | 内存排序 | str / None | None / acquire / release / acq_rel / relaxed | 按目标原子能力。 | None |
| `scope` | 原子操作可见域 | str / None | None / gpu / cta / sys | 按目标原子能力。 | None |

**调用示例**

ptr 为支持原子加的 GM 指针，value 与其元素类型相容。

```python
@tla.jit
def example(ptr, value):
    old = tl.atomic_add(ptr, value)
```

<a name="api-tl-sum"></a>

## tl.sum

签名：`tl.sum(input, axis=None, keep_dims=False, dtype=None)`

返回：求和结果。

沿用 Triton 入口，适用场景仍受 TLA 编译路径和目标约束。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `input` | 输入元素 | tl.tensor | 数值 tensor | — | 必填 |
| `axis` | 规约轴 | 编译期 int / None | 有效轴或 None | None 规约所有元素。 | None |
| `keep_dims` | 保留被规约的轴 | 编译期 bool | False / True | — | False |
| `dtype` | 指定规约前的元素类型 | tl.dtype / None | None 或数值 dtype | None 按 Triton 默认累加规则。 | None |

**调用示例**

示例使用 Triton tensor。

```python
@tla.jit
def example(x):
    values = tl.load(x + tl.arange(0, 64))
    result = tl.sum(values, axis=0)
```

<a name="api-tl-randint"></a>

## tl.randint

签名：`tl.randint(seed, offset, n_rounds=10)`

返回：每个偏移对应的 32 位整数结果。

沿用 Triton 入口，适用场景仍受 TLA 编译路径和目标约束。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `seed` | 随机种子 | 标量 int / 标量整数 tl.tensor | 数据类型：uint64（转换后） | — | 必填 |
| `offset` | 每个值的计数器偏移 | 整数 tl.tensor | 数据类型：uint32 或 uint64（Philox 计数器） | — | 必填 |
| `n_rounds` | Philox 轮数 | 编译期 int | 正整数 | 默认 Philox 轮数。 | 10 |

**调用示例**

示例使用 Triton tensor。

```python
@tla.jit
def example(seed):
    offsets = tl.arange(0, 64).to(tl.uint32)
    values = tl.randint(seed, offsets)
```

<a name="api-tl-rand"></a>

## tl.rand

签名：`tl.rand(seed, offset, n_rounds=10)`

返回：每个偏移对应的 float32 随机数。

沿用 Triton 入口，适用场景仍受 TLA 编译路径和目标约束。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `seed` | 随机种子 | 标量 int / 标量整数 tl.tensor | 数据类型：uint64（转换后） | — | 必填 |
| `offset` | 每个值的计数器偏移 | 整数 tl.tensor | 数据类型：uint32 或 uint64（Philox 计数器） | — | 必填 |
| `n_rounds` | Philox 轮数 | 编译期 int | 正整数 | 默认 Philox 轮数。 | 10 |

**调用示例**

示例使用 Triton tensor。

```python
@tla.jit
def example(seed):
    offsets = tl.arange(0, 64).to(tl.uint32)
    values = tl.rand(seed, offsets)
```

<a name="api-tla-simt-to-ptr"></a>

## tla.simt.to_ptr

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`tla.simt.to_ptr ( src: tla.buffer ) -> tt.ptr`

## 范围说明

以下条目在当前整理表中没有可定位的 TLA 接口名。

| 原序号 | 范围结论 | 涉及场景 |
| --- | --- | --- |
| 121 | tla暂不开放simt vf内thread级编程 | T.get_thread_binding |
| 128 | tla暂不开放simt vf内thread级编程 | T.warp_reduce_max |
| 129 | tla暂不开放simt vf内thread级编程 | T.warp_reduce_min |
| 130 | tla暂不开放simt vf内thread级编程 | T.warp_reduce_sum |
| 131 | 暂不开放 | T.get_block_binding |
| 132 | 暂不开放 | T.get_block_bindings |
| 133 | 暂不开放 | T.get_block_extent |
| 134 | 暂不开放 | T.get_block_extents |
| 135 | 暂不开放 | T.get_thread_extent |
| 136 | 暂不开放 | T.get_thread_extents |
