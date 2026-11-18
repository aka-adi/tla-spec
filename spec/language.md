# 语言接口规格

[返回接口目录](../README.md)

<a name="api-tla-sqrt"></a>

## tla.sqrt

签名：`tla.sqrt(x)`

返回：与所选标量或向量路径对应的结果。

编译表示：hivm.hir.vsqrt。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `x` | 输入值 | float / 标量 tl.tensor / tla.vector | 标量：float32<br>SIMD 向量：float16 / float32 | Python float 必须被 Triton 推断为 float32。输出保持数据类型，向量保持 lane 数。 | 必填 |

**调用示例**

```python
@tla.jit
def example():
    value = tla.sqrt(4.0)
```

仓库测试：[tla_ut/vector/vector_type.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/vector/vector_type.py#L124)。

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
| `a` | 第一个操作数 | int / 标量整数 tl.tensor | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32<br>B64：int64 / uint64 | 不接受 bool。禁止混合 signed 和 unsigned tensor。除数非零，signed 最小值 ÷ −1 未定义。 | 必填 |
| `b` | 第二个操作数 | int / 标量整数 tl.tensor | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32<br>B64：int64 / uint64 | 不接受 bool。禁止混合 signed 和 unsigned tensor。除数非零，signed 最小值 ÷ −1 未定义。 | 必填 |

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
| `cond` | 必须成立的条件 | 标量 int1 tl.tensor / bool | 数据类型：int1 | 条件为假时行为未定义。 | 必填 |

**调用示例**

```python
@tla.jit
def example(x):
    tl.assume(x >= 0)
```

<a name="api-tla-where"></a>

## tla.where

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

## 范围说明

以下条目在当前整理表中没有可定位的 TLA 接口名。

| 原序号 | 范围结论 | 涉及场景 |
| --- | --- | --- |
| 216 | 规范不支持（triton里不能写break） | T.loop_break |
| 217 | 暂不支持 | T.serial for i in range(n) |
| 218 | 暂不支持 | T.unroll for i in tl.static_range(n) |
| 219 | 暂不支持 | T.vectorized |
