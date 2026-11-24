# 语言接口规格

[返回接口目录](../README.md)

<a name="api-tla-sqrt"></a>

## tla.sqrt

签名：`tla.sqrt(x)`

返回：输入值的平方根，类型为 float32 标量。

编译表示：math.sqrt。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `x` | 输入值 | 浮点标量 | 数据类型：float32 | Python float 必须被 Triton 推断为 float32。输出保持数据类型。 | 必填 |

**调用示例**

```python
@tla.jit
def example():
    value = tla.sqrt(4.0)
```

仓库测试：[tla_ut/vector/vector_type.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/vector/vector_type.py#L136)。

<a name="api-tla-symbolic"></a>

## tla.symbolic

范围结论：不需要。保留原提案供追溯。

原始提案：`tla.symbolic(name: str, dtype: str = "int32") -> Expr`

<a name="api-tla-truncmod"></a>

## tla.truncmod

签名：`tla.truncmod(a, b)`

返回：计算结果，类型由操作数或显式 dtype 决定。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `a` | 第一个操作数 | 整数标量 | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32<br>B64：int64 / uint64 | 不接受 bool。禁止混合 signed 和 unsigned 整数标量。除数非零，signed 最小值 ÷ −1 未定义。 | 必填 |
| `b` | 第二个操作数 | 整数标量 | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32<br>B64：int64 / uint64 | 不接受 bool。禁止混合 signed 和 unsigned 整数标量。除数非零，signed 最小值 ÷ −1 未定义。 | 必填 |

**调用示例**

```python
@tla.jit
def example(x):
    value = tla.truncmod(x, 16)
```

仓库测试：[tla_ut/vector/vector_type.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/vector/vector_type.py#L104)。

<a name="api-tl-assume"></a>

## tl.assume

签名：`tl.assume(cond)`

返回：None。

沿用 Triton 入口，适用场景仍受 TLA 编译路径和目标约束。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `cond` | 必须成立的条件 | bool 标量 | 数据类型：int1 | 条件为假时行为未定义。 | 必填 |

**调用示例**

```python
@tla.jit
def example(x):
    tl.assume(x >= 0)
```

<a name="api-tla-where"></a>

## tla.where

范围结论：不需要。与 [tla.select](scalar.md#api-tla-select) 功能重叠，保留原提案供追溯。

签名：`tla.where(condition, x, y)`

返回：与所选标量或向量路径对应的结果。

编译表示：hivm.hir.vsel。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `condition` | 选择条件 | 编译期 bool / 标量 int1 tl.tensor / tla.vector（int1 谓词） | 标量：int1<br>SIMD：同 lanes 的 int1 谓词 | 运行时标量条件只选择标量分支。 | 必填 |
| `x` | 输入值 | bool / int / float / 标量 tl.tensor / tla.vector | 标量：bool / 数值标量<br>SIMD：同 dtype、同 lanes 的一对向量 | 两分支均提前求值。 | 必填 |
| `y` | 条件为假时选择的值 | bool / int / float / 标量 tl.tensor / tla.vector | 标量：bool / 数值标量<br>SIMD：同 dtype、同 lanes 的一对向量 | 两分支均提前求值。 | 必填 |

**调用示例**

```python
@tla.jit
def example(x):
    value = tla.where(x > 0, x, 0)
```

仓库测试：[tla_ut/simd/add_cmp.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/add_cmp.py#L114)。

<a name="api-tla-range"></a>

## tla.range

签名：`tla.range(arg1, arg2=None, step=None)`

返回：用于 `for` 语句的运行期循环迭代器，每轮迭代给出一个整数标量归纳变量。

与 `tl.range` 为同一对象。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `arg1` | 起始值，单参数时为结束值 | int / tla.constexpr / 标量整数 tl.tensor | 整数标量 | 只传一个参数时等价于 `tla.range(0, arg1)`。 | 必填 |
| `arg2` | 结束值（不含） | 同 `arg1` | 整数标量 | — | None |
| `step` | 步长 | int / tla.constexpr / 标量整数 tl.tensor | 非零整数标量 | 运行时步长须为正。 | 1 |

约束：

- 上下界和步长可为编译期整数或运行时标量，归纳变量的类型取三者的整型提升结果。
- 循环保留到目标 IR，边界为编译期常量时也不展开。
- 负步长只接受编译期整数。
- 不提供 num_stages、loop_unroll_factor 等调度提示参数。
- 可在 kernel 主体、Vector 执行域和 SIMD 区域内使用。

**调用示例**

```python
@tla.jit
def example(n):
    for i in tla.range(1, n, 2):
        tla.device_print("runtime", i)
```

仓库测试：[tla_ut/test_range.py](https://github.com/triton-lang/triton-ascend/blob/51b4e01ca5f6ec35304765856e6c9ce006e095cf/third_party/ascend/unittest/tla_ut/test_range.py#L13)。

<a name="api-tla-static-range"></a>

## tla.static_range

签名：`tla.static_range(arg1, arg2=None, step=None)`

返回：用于 `for` 语句的编译期迭代器，每轮迭代的归纳变量为编译期整数。

与 `tl.static_range` 为同一对象。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `arg1` | 起始值，单参数时为结束值 | int / tla.constexpr | 编译期整数 | 只传一个参数时等价于 `tla.static_range(0, arg1)`。 | 必填 |
| `arg2` | 结束值（不含） | int / tla.constexpr | 编译期整数 | — | None |
| `step` | 步长 | int / tla.constexpr | 非零编译期整数，可为负 | — | 1 |

约束：

- 三个参数都必须是编译期整数，传入运行时值会在前端断言失败。
- 循环在前端完全展开，IR 中不保留循环，迭代次数不宜过大。
- 可在 kernel 主体、Vector 执行域和 SIMD 区域内使用。

**调用示例**

```python
@tla.jit
def example():
    for i in tla.static_range(4, 0, -2):
        tla.device_print("unrolled", i)
```

仓库测试：[tla_ut/test_range.py](https://github.com/triton-lang/triton-ascend/blob/51b4e01ca5f6ec35304765856e6c9ce006e095cf/third_party/ascend/unittest/tla_ut/test_range.py#L15)。

## 范围说明

以下条目在原整理表中没有单独的 TLA 接口入口，结论如下。

| 原序号 | 范围结论 | 涉及场景 |
| --- | --- | --- |
| 216 | 规范不支持（triton里不能写break） | T.loop_break |
| 217 | 由 [tla.range](#api-tla-range) 提供运行期循环 | T.serial for i in range(n) |
| 218 | 由 [tla.static_range](#api-tla-static-range) 提供编译期展开 | T.unroll for i in tl.static_range(n) |
| 219 | 暂不支持 | T.vectorized |
