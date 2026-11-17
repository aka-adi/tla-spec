# 标量接口规格

[返回接口目录](../README.md)

<a name="api-tla-set-atomic"></a>

## tla.set_atomic

签名：`tla.set_atomic(kind, dtype)`

返回：None。

编译表示：hivm.hir.set_atomic。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `kind` | 原子更新操作 | 编译期 tla.AtomicKind | NONE / ADD / MAX / MIN | NONE 恢复普通存储模式。 | 必填 |
| `dtype` | 元素类型 | 编译期 tl.dtype | 数据类型：<br>B8：int8<br>B16：int16 / float16 / bfloat16<br>B32：int32 / float32<br>B64：不支持 | — | 必填 |

**调用示例**

```python
@tla.jit
def example(src_ub, y):
    tla.set_atomic(tla.AtomicKind.ADD, tl.float32)
    tla.store_ub_to_gm(src_ub, tla.view(y, (64,)))
    tla.set_atomic(tla.AtomicKind.NONE, tl.float32)
```

仓库测试：[tla_ut/dma/set_atomic.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dma/set_atomic.py#L35)。

<a name="api-tla-select"></a>

## tla.select

签名：`tla.select(condition, true_value, false_value)`

返回：计算结果，类型由操作数或显式 dtype 决定。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `condition` | 选择条件 | bool / int / float / 标量 tl.tensor | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 tensor。 | 必填 |
| `true_value` | 条件为真时选取的值 | bool / int / float / 标量 tl.tensor | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 tensor。 | 必填 |
| `false_value` | 条件为假时选取的值 | bool / int / float / 标量 tl.tensor | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 tensor。 | 必填 |

**调用示例**

```python
@tla.jit
def example(x):
    value = tla.select(x > 0, x, 0)
```

仓库测试：[tla_ut/scalar/scalar_frontend_contract.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scalar/scalar_frontend_contract.py#L48)。

<a name="api-tla-abs"></a>

## tla.abs

签名：`tla.abs(x)`

返回：与所选标量或向量路径对应的结果。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `x` | 输入值 | bool / int / float / 标量 tl.tensor | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 tensor。 | 必填 |

**调用示例**

```python
@tla.jit
def example(x):
    value = tla.abs(x)
```

<a name="api-xor"></a>

## ^

签名：`lhs ^ rhs`

返回：计算结果，类型由操作数或显式 dtype 决定。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `lhs` | 左操作数 | int / 标量整数 tl.tensor | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32<br>B64：int64 / uint64 | 对整数位模式做 XOR，类型提升遵循 Triton。 | 必填 |
| `rhs` | 右操作数 | int / 标量整数 tl.tensor | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32<br>B64：int64 / uint64 | 对整数位模式做 XOR，类型提升遵循 Triton。 | 必填 |

**调用示例**

```python
@tla.jit
def example(x, y):
    value = x ^ y
```

<a name="api-tla-cast"></a>

## tla.cast

签名：`tla.cast(input, dtype, fp_downcast_rounding=None, bitcast=False)`

返回：计算结果，类型由操作数或显式 dtype 决定。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `input` | 待转换的标量 | bool / int / float / 标量 tl.tensor | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 tensor。 | 必填 |
| `dtype` | 元素类型 | 编译期 tl.dtype | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 不接受 bfloat16。按源和目标类型对执行 Triton cast。 | 必填 |
| `fp_downcast_rounding` | 浮点窄化舍入模式 | 编译期 str 或 None | None / rtne / rtz | 仅用于 Triton 支持的浮点窄化类型对。 | None |
| `bitcast` | 是否按位重解释 | 编译期 bool | False / True | True 需要等位宽。 | False |

**调用示例**

```python
@tla.jit
def example(x):
    value = tla.cast(x, tl.int32)
```

仓库测试：[tla_ut/scalar/cdiv.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scalar/cdiv.py#L40)。

<a name="api-tla-cdiv"></a>

## tla.cdiv

签名：`tla.cdiv(x, div)`

返回：计算结果，类型由操作数或显式 dtype 决定。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `x` | 输入值 | int / 标量整数 tl.tensor | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32<br>B64：int64 / uint64 | 面向非负被除数、正除数的常规用法，中间计算不能溢出。 | 必填 |
| `div` | 除数 | int / 标量整数 tl.tensor | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32<br>B64：int64 / uint64 | 面向非负被除数、正除数的常规用法，中间计算不能溢出。 | 必填 |

**调用示例**

```python
@tla.jit
def example(x):
    value = tla.cdiv(x, 16)
```

仓库测试：[tla_ut/scalar/cdiv.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scalar/cdiv.py#L42)。

<a name="api-tla-clamp"></a>

## tla.clamp

签名：`tla.clamp(x, min, max, propagate_nan=tl.PropagateNan.NONE)`

返回：计算结果，类型由操作数或显式 dtype 决定。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `x` | 输入值 | bool / int / float / 标量 tl.tensor | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 或 float16 tensor。 | 必填 |
| `min` | 下界 | bool / int / float / 标量 tl.tensor | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 或 float16 tensor。 | 必填 |
| `max` | 上界 | bool / int / float / 标量 tl.tensor | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 或 float16 tensor。 | 必填 |
| `propagate_nan` | 选择 NaN 传播规则 | 编译期 tl.PropagateNan | NONE / ALL | ALL 传播 NaN。NONE 的目标 NaN 行为应按单 NaN 和双 NaN 分别验证。 | tl.PropagateNan.NONE |

**调用示例**

```python
@tla.jit
def example(x):
    value = tla.clamp(x, 0, 255)
```

仓库测试：[tla_ut/scalar/clamp.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scalar/clamp.py#L46)。

<a name="api-tl-constexpr"></a>

## tl.constexpr

签名：`tl.constexpr`

返回：用于表示编译期值的类型，可用于内核参数标注。

沿用 Triton 入口，适用场景仍受 TLA 编译路径和目标约束。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `value` | 编译期值 | Python 对象 | 具体范围由使用该值的表达式决定 | 不作为设备运行时 SSA 参数。 | 必填 |

**调用示例**

```python
@tla.jit
def example(size: tl.constexpr):
    tile = tla.alloc((size,), tl.float32)
```

仓库测试：[tla_ut/scope/gm2l1.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scope/gm2l1.py#L22)。

<a name="api-tla-exp"></a>

## tla.exp

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`tla.exp(x) -> Expr`

<a name="api-tla-infinity"></a>

## tla.infinity

签名：`tla.infinity(dtype)`

返回：计算结果，类型由操作数或显式 dtype 决定。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `dtype` | 元素类型 | 编译期 tl.dtype | 数据类型：<br>B8：不支持<br>B16：float16 / bfloat16<br>B32：float32<br>B64：不支持 | — | 必填 |

**调用示例**

```python
@tla.jit
def example():
    value = tla.infinity(tl.float32)
```

仓库测试：[tla_ut/scalar/infinity.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scalar/infinity.py#L40)。

<a name="api-tla-log1p"></a>

## tla.log1p

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`tla.log1p(x) -> Expr`

<a name="api-tla-maximum"></a>

## tla.maximum

签名：`tla.maximum(a, b, propagate_nan=tl.PropagateNan.NONE)`

返回：计算结果，类型由操作数或显式 dtype 决定。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `a` | 第一个操作数 | bool / int / float / 标量 tl.tensor | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 或 float16 tensor。 | 必填 |
| `b` | 第二个操作数 | bool / int / float / 标量 tl.tensor | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 或 float16 tensor。 | 必填 |
| `propagate_nan` | 选择 NaN 传播规则 | 编译期 tl.PropagateNan | NONE / ALL | ALL 传播 NaN。NONE 的目标 NaN 行为应按单 NaN 和双 NaN 分别验证。 | tl.PropagateNan.NONE |

**调用示例**

```python
@tla.jit
def example(x, y):
    value = tla.maximum(x, y)
```

仓库测试：[tla_ut/scalar/maximum.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scalar/maximum.py#L43)。

<a name="api-tla-max-value"></a>

## tla.max_value

签名：`tla.max_value(dtype)`

返回：计算结果，类型由操作数或显式 dtype 决定。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `dtype` | 元素类型 | 编译期 tl.dtype | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 / float64 | — | 必填 |

**调用示例**

```python
@tla.jit
def example():
    value = tla.max_value(tl.int32)
```

仓库测试：[tla_ut/scalar/max_value.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scalar/max_value.py#L54)。

<a name="api-tla-minimum"></a>

## tla.minimum

签名：`tla.minimum(a, b, propagate_nan=tl.PropagateNan.NONE)`

返回：计算结果，类型由操作数或显式 dtype 决定。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `a` | 第一个操作数 | bool / int / float / 标量 tl.tensor | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 或 float16 tensor。 | 必填 |
| `b` | 第二个操作数 | bool / int / float / 标量 tl.tensor | 数据类型：<br>B1：int1<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 按 Triton 标量转换和类型提升处理。不接受 bfloat16 或 float16 tensor。 | 必填 |
| `propagate_nan` | 选择 NaN 传播规则 | 编译期 tl.PropagateNan | NONE / ALL | ALL 传播 NaN。NONE 的目标 NaN 行为应按单 NaN 和双 NaN 分别验证。 | tl.PropagateNan.NONE |

**调用示例**

```python
@tla.jit
def example(x, y):
    value = tla.minimum(x, y)
```

仓库测试：[tla_ut/scalar/minimum.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scalar/minimum.py#L43)。

<a name="api-tla-sigmoid"></a>

## tla.sigmoid

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`tla.sigmoid(x) -> Expr`
