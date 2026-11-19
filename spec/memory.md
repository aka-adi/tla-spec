# 内存接口规格

[返回接口目录](../README.md)

以下 DMA、L0 搬运和填充约束面向 Ascend950（RegBase）。

<a name="api-tla-alloc"></a>

## tla.alloc

计划签名：`tla.alloc(shape, dtype, space=None)`

返回：tla.buffer。

分配由 `space` 指定地址空间。多缓冲版本数通过 [tla.annotate_multi_buffer](schedule.md#api-tla-annotate-multi-buffer) 单独标注。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `shape` | 逻辑形状 | tuple[int, ...] / list[int] | 非空正整数组 | 按元素计。 | 必填 |
| `dtype` | 元素类型 | 编译期 tl.dtype | 可物化为 memref 元素的整数或浮点 dtype | 分配类型须同时满足该空间的内存规划和后续操作的 dtype 约束。 | 必填 |
| `space` | 指定地址空间 | 编译期 tla.AddressSpace / str / None | 地址空间：UB / L1 / L0C（已确认分配路径） | 省略或传 None 时为 UB。GM 禁止 alloc，L0A_SF/L0B_SF 无映射。其他有空间映射的名字不自动获得分配支持。 | None |

**调用示例**

```python
@tla.jit
def example():
    with tla.scope.vector():
        tile = tla.alloc((16, 16), tl.float16, space='ub')
```

仓库测试：[tla_ut/scope/pointer_abi.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scope/pointer_abi.py#L15)。

<a name="api-tla-reshape"></a>

## tla.reshape

签名：`tla.reshape(src, shape)`

返回：tla.buffer。

仅支持静态形状：`src` 的 shape 与 strides 须在编译期已知，`shape` 各维须为编译期正整数；不支持运行时计算的形状。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | tla.buffer | 地址空间及数据类型继承源缓冲区 | 源的静态 strides 必须等于其形状对应的 row-major strides。允许非零或动态 offset，结果继承 offset。 | 必填 |
| `shape` | 逻辑形状 | tuple[int, ...] / list[int] | 非空正整数组 | 各维须为编译期正整数，reshape 后元素总数相同。 | 必填 |

**调用示例**

x_ub 为连续的 16×16 UB 缓冲区。

```python
@tla.jit
def example(x_ub, y):
    flat = tla.reshape(x_ub, (256,))
    tla.store_ub_to_gm(flat, tla.view(y, (256,)))
```

仓库测试：[tla_ut/vector/subview.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/vector/subview.py#L64)。

<a name="api-tla-view"></a>

## tla.view

签名：`tla.view(src, shape, strides=None, offsets=None)`

返回：tla.buffer。

视图固定为 GM，不提供地址空间参数。`src` 须为 GM kernel 指针，偏移按有符号 64 位元素索引计算，保留指针来源，并允许运行时标量参与指针地址计算。指针源的访问范围由调用者保证。不接受 tla.buffer 源，已有缓冲区的切片使用 [tla.subview](#api-tla-subview)，改秩使用 [tla.reshape](#api-tla-reshape)。

仅支持静态形状与静态步长：`shape` 和 `strides` 须为编译期正整数；`offsets` 可为编译期整数或运行时整数标量。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | 标量 Triton 指针 | GM kernel 指针；元素须为整字节宽的整数或浮点 | 指针源继承指针元素类型。 | 必填 |
| `shape` | 逻辑形状 | tuple[int, ...] / list[int] | 非空正整数组 | 各维须为编译期正整数，按元素计。 | 必填 |
| `strides` | 各轴步长 | tuple[int, ...] / list[int] / None | 非空正整数组 / None | 各轴须为编译期正整数。与结果秩相同。省略时使用 row-major。 | None |
| `offsets` | 基于源地址的元素偏移 | 编译期整数或整数标量 tl.tensor 组成的 tuple / None | 按元素计的整数偏移 / None | 维数须匹配结果秩，相对于指针。 | None |

**调用示例**

```python
@tla.jit
def example(x, dst_ub):
    tile = tla.view(x, (16, 16), offsets=(0, 0))
    tla.load_gm_to_ub(tile, dst_ub)
```

仓库测试：[tla_ut/scope/pointer_abi.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scope/pointer_abi.py#L16)。

<a name="api-tla-subview"></a>

## tla.subview

签名：`tla.subview(src, offsets, sizes, strides=None)`

返回：tla.buffer。

编译表示：memref.subview。

支持 GM 与片上空间：`src` 须为 tla.buffer，结果继承源空间，不改变地址空间。标量指针不能直接传入；GM 缓冲区须先经 [tla.view](#api-tla-view) 从 kernel 指针建立。

仅支持静态形状与静态步长：`sizes` 和 `strides` 须为编译期正整数；`offsets` 可为编译期整数或运行时整数标量。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | tla.buffer | 地址空间：GM / UB / L1 / L0C；数据类型继承源缓冲区 | 同秩切片，结果与源共享存储，并保留源地址空间。 | 必填 |
| `offsets` | 相对于源视图的各轴逻辑元素偏移 | 编译期整数或整数标量 tl.tensor 组成的 tuple | 按元素计的整数偏移 | 维数须匹配。静态子视图越界拒绝，动态偏移由调用者保证。 | 必填 |
| `sizes` | 各轴要截取的元素个数 | tuple[int, ...] / list[int] | 非空正整数组 | 各维须为编译期正整数，按元素计。 | 必填 |
| `strides` | 各轴步长 | tuple[int, ...] / list[int] / None | 非空正整数组 / None | 各轴步长须为编译期正整数。与对应视图秩相同。省略时各轴相对步长为 1。结果 stride 为源 stride 与相对步长的乘积。 | None |

**调用示例**

x_ub 为至少 16×16 个元素的二维缓冲区，目的视图与子视图类型一致。

```python
@tla.jit
def example(x_ub, y):
    tile = tla.subview(x_ub, (0, 0), (16, 16))
    tla.store_ub_to_gm(tile, tla.view(y, (16, 16)))
```

仓库测试：[tla_ut/simd/predicate_views.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/predicate_views.py#L39)。

<a name="api-tla-fill"></a>

## tla.fill

签名：`tla.fill(dst, value)`

返回：None。将标量广播到整个目标缓冲区，编译表示为标量 `hivm.hir.vbrc`。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `dst` | 待填充的缓冲区 | tla.buffer | UB；float32 / bfloat16 / int32；1D / 2D | 连续 row-major、offset=0 的直接分配或等价 reshape；元素数为正且不超过 UB 容量；允许非对齐尾部。 | 必填 |
| `value` | 广播标量 | Python int / Python float | 仅限下表组合 | 不接受 Python bool 或 tensor（含标量 tl.tensor）。 | 必填 |

| 目的元素类型 | Python 常量 |
| --- | --- |
| float32 | int32 范围内的整数；float |
| bfloat16 | int32 范围内的整数；暂不支持 Python float |
| int32 | int32 范围内的整数；不接受 float |

整数常量范围为 `[-2**31, 2**31-1]`，填充浮点目的时按目的类型舍入。当前仅提供 Python 常量输入，不提供运行时标量 tensor 输入或其隐式类型转换。

目的仅提供直接 `tla.alloc` 或保持连续、零偏移的 `tla.reshape` 结果。暂不支持任何目的 `subview`（包括静态全零 offset）、非零或动态偏移、带行间距的非连续视图及其他地址空间。

浮点常量按数值语义填充，不承诺负零符号或 NaN 载荷的位级保持。调用方须确保生产、消费及复用缓冲区的依赖有正确同步。
