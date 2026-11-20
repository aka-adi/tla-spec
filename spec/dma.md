# DMA接口规格

[返回接口目录](../README.md)

以下 DMA、L0 搬运和填充约束面向 Ascend950（RegBase）。

<a name="api-tla-load-gm-to-l1"></a>

## tla.load_gm_to_l1

签名：`tla.load_gm_to_l1(src, dst, params=None)`

返回：None。编译表示：`hivm.hir.nd2nz`。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源 | tla.buffer | GM；float16 / bfloat16 | 2D ND，末维连续；行间距、偏移和范围须匹配底层分配。 | 必填 |
| `dst` | 目的 | tla.buffer | L1；与 src 相同 | 4D NZ，与源逻辑矩阵一致。 | 必填 |
| `params` | 搬运配置 | tla.LoadGM2L1Params / None | 下述配置 | None 等价于默认配置。 | None |
| `params.enable_nd2nz` | 启用 ND→NZ 转换 | 编译期 bool | True | False 暂不支持并抛出 NotImplementedError；非 bool 抛出 TypeError。 | True |

逻辑 M、N 为 16 的正整数倍，源和目的须满足 NZ 分块对齐且不越界；当前范围不包括尾块、3D→5D 批量和其他 dtype。

约束：暂不支持手动设置 `l2_cache_ctrl` 和 `enable_dn2nz`。

```python
@tla.jit
def example(src, dst):
    with tla.scope.cube():
        tla.load_gm_to_l1(src, dst)
```

仓库测试：[tla_ut/dma/nd2nz.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dma/nd2nz.py#L22)。

<a name="api-tla-load-gm-to-ub"></a>

## tla.load_gm_to_ub

签名：`tla.load_gm_to_ub(src, dst, params=None)`

返回：None。

编译表示：hivm.hir.load。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 待搬运的源缓冲区 | tla.buffer | 地址空间：GM<br>数据类型：与 dst 相同（不含 `float64`） | 源地址需要按照对应数据类型所占字节数对齐，与 dst 的 rank、逻辑 shape 相同。 | 必填 |
| `dst` | 接收数据的目标缓冲区 | tla.buffer | 地址空间：UB<br>数据类型：与 src 相同 | 目的地址需要32字节对齐。 | 必填 |
| `params` | L2 缓存配置 | tla.LoadGM2UBParams 或 None | None / tla.LoadGM2UBParams 实例 | None 等价于默认配置。 | None |
| `params.l2_cache_ctrl` | GM→UB 的 L2 缓存策略 | tla.L2CachePolicy | `NORMAL_FV` 表示普通模式优先淘汰<br>`NORMAL_LV` 表示普通模式、优先保留 | 无 | `None`（使用 `NORMAL_FV`） |

本接口在`tla.scope.vector()`内、SIMD/SIMT VF 外区域调用。

硬件可能在 UB 中写入补齐字节。

<a name="gm-ub-capacity"></a>

GM↔UB 搬运须按 UB 端视图的布局预留空间。

- 一维数据连续或二维各行紧密相接时，从 UB 视图起始地址算起，可用空间至少为有效数据总字节数按 32 字节向上对齐后的大小。补齐空间位于整块末尾。
- 二维每行连续且行间留有空隙时，补齐空间位于各行末尾。UB 行距须为 32 字节的整数倍，且能容纳一行有效数据。每行均按该行距分配空间，包括最后一行。

将仅覆盖有效数据的视图作为 `dst`。逐行预留时，用 `tla.subview` 取各行的有效区域。

**调用示例**

前提：`x` 指向连续的 60 个 float32 元素。UB 每行分配 32 个元素，前 30 个为有效数据，末尾 8 字节留给补齐。

```python
@tla.jit
def example(x):
    with tla.scope.vector():
        tile = tla.alloc((2, 32), tl.float32)
        dst = tla.subview(tile, offsets=(0, 0), sizes=(2, 30))
        p: tl.constexpr = tla.LoadGM2UBParams(l2_cache_ctrl=tla.L2CachePolicy.NORMAL_LV)
        tla.load_gm_to_ub(tla.view(x, (2, 30)), dst, p)
```

仓库测试：`tla_ut/dma/load_store.py`。

<a name="api-tla-store-ub-to-gm"></a>

## tla.store_ub_to_gm

签名：`tla.store_ub_to_gm(src, dst)`

返回：None。

编译表示：hivm.hir.store。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 待搬运的源缓冲区 | tla.buffer | 地址空间：UB<br>数据类型：与 dst 相同（不含 `float64`） | 源地址需要32字节对齐。 | 必填 |
| `dst` | 接收数据的目的缓冲区 | tla.buffer | 地址空间：GM<br>数据类型：与 src 相同 | 目的地址需要按照对应数据类型所占字节数对齐, 与 src 的 rank、逻辑 shape 相同。 | 必填 |

原子写回使用独立的 [tla.set_atomic](scalar.md#api-tla-set-atomic)，受控写回完成后用 NONE 显式复位。
本接口在`tla.scope.vector()`内、SIMD/SIMT VF 外区域调用。

以 `src` 的起始地址和布局为准，按 [UB 容量和行距规则](#gm-ub-capacity) 预留空间。

**调用示例**

前提：`x`、`y` 均指向连续的 64 个 float32 元素，使用 `(2, 32)` 布局。

```python
@tla.jit
def example(x, y):
    with tla.scope.vector():
        tile = tla.alloc((2, 32), tl.float32)
        tla.load_gm_to_ub(tla.view(x, (2, 32)), tile)
        tla.store_ub_to_gm(tile, tla.view(y, (2, 32)))
```

仓库测试：`tla_ut/dma/load_store.py`。

<a name="api-tla-copy-l1-to-l0"></a>

## tla.copy_l1_to_l0

签名：`tla.copy_l1_to_l0(src, dst, params=None)`

返回：None。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 待搬运的源缓冲区 | tla.buffer | L1；float16 / bfloat16 | 源视图描述所需矩阵分块，物理布局须与转置模式匹配。 | 必填 |
| `dst` | 接收数据的目标缓冲区 | tla.buffer | L0A / L0B；与 src 相同 | 目标视图决定每块搬运范围；L0A 为 zN，L0B 为 nZ，地址及 offset 按 512B/fractal 对齐。 | 必填 |
| `params` | 搬运配置 | tla.CopyL1ToL0Params 或 None | 仅包含 is_transpose | None 等价于使用默认配置。 | None |
| `params.is_transpose` | 搬运时是否转置 | 编译期 bool | False / True | A、B 各自独立设置；转置选择须与源、目的布局匹配。 | False |

搬运范围、步长及起始位置由源、目的 buffer 视图的形状、布局、stride 和 offset 描述；分块由调用方准备相应视图，无需额外传递分区字段或矩阵维度。

约束：每次调用仅处理单 batch，不提供批量搬运配置。源、目的使用相容的 4D 物理布局，FP16/BF16 的内层为 (16,16)；地址范围不得越界，且不得超过 L0A/L0B 各 64KiB 的容量。调用方负责 MTE2→MTE1→M 的依赖及缓冲区复用生命周期。

约束：stride 以元素为单位。源、目的均须满足 `strides[1:] = (256, 16, 1)`；`stride[0]` 为 256 的倍数，且不小于 `shape[1] × 256`。仅最外层允许额外间距，内层 fractal 必须连续；转置模式不放宽这些物理布局要求。

约束：暂不支持其他数据类型、布局及自动补齐尾块。

仓库 IR 构造用例：[tla_ut/dma/copy.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dma/copy.py#L42)。

<a name="api-tla-make-partition-view"></a>

## tla.make_partition_view

**本轮不需要。** 缺少同名可执行入口和完整 lowering。保留规格供追溯。

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`make_partition_view( base: tla.buffer (ptr), shape: list[int], strides: list[int], tile_shape: list[int], dim_map: list[int], padding_value: enum<zero,nan,inf,-inf> )-> tla.tensor_view`

<a name="api-tla-make-strided-view"></a>

## tla.make_strided_view

**本轮不需要。** 缺少同名可执行入口和完整 lowering。保留规格供追溯。

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`make_strided_view( base: tla.buffer (ptr), shape: list[int], strides: list[int], tile_shape: list[int], dim_map: list[int], padding_value: enum<zero,nan,inf,-inf>, traversal_strides: list[int] )-> tla.tensor_view`

<a name="api-tla-make-gather-scatter-view"></a>

## tla.make_gather_scatter_view

**本轮不需要。** 缺少同名可执行入口和完整 lowering。保留规格供追溯。

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`make_gather_scatter_view( base: tla.buffer (ptr), shape: list[int], strides: list[int], tile_shape: list[int], sparse_dim: list[int], padding_value: enum<zero,nan,inf,-inf> )-> tla.tensor_view`

<a name="api-tla-load-view"></a>

## tla.load_view

**本轮不需要。** 缺少同名可执行入口和完整 lowering。保留规格供追溯。

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`load_view( src: tla.tensor_view, dst: tla.buffer, tile: list[list[int]] ) 只用于vector kernel`

<a name="api-tla-store-view"></a>

## tla.store_view

**本轮不需要。** 缺少同名可执行入口和完整 lowering。保留规格供追溯。

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`store_view( src: tla.buffer, dst: tla.tensor_view, tile: list[list[int]] ) 只用于vector kernel`

<a name="api-tla-set-pad-value"></a>

## tla.set_pad_value

**本轮不需要。** 缺少同名可执行入口和完整 lowering。保留规格供追溯。

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`set_pad_value()`

<a name="api-tla-set-post-copy"></a>

## tla.set_post_copy

**本轮不需要。** 缺少同名可执行入口和完整 lowering。保留规格供追溯。

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`set_post_copy()`

<a name="api-tla-stride-store"></a>

## tla.stride_store

范围结论：不需要。保留原提案供追溯。

原始提案：`【主推｜结构化带步长访存，不暴露指针】 stride_store(src: tla.buffer, dst: tla.buffer, offset: tla.expr, strides: tuple[tla.expr, ...], numels: tuple[tla.expr, ...]) -> None stride_load(src: tla.buffer, dst: tla.buffer, offset: tla.expr, other: tla.scalar, strides: tuple[tla.expr, ...], numels: tuple[tla.expr, ...]) -> None # rank<=3；numels 是有效元素数（越界预测），非重复次数；仅 AIV 【SIMD 级｜VF 内逐寄存器写】 simd.store_with_stride(dst: tla.buffer, indices, stride: int, mask: tla.mask, val: tla.vreg) -> None # stride 必须编译期常量且为 32B 块整数倍；无 post-update 【流式推进｜用循环携带地址，不提供可变指针对象】 for addr in tla.carry(init_addr): # 下沉 scf.for iter_args buf = tla.make_tensor(addr, ...) addr = addr + step`

<a name="api-tla-stride-load"></a>

## tla.stride_load

范围结论：不需要。保留原提案供追溯。

原始提案：`【主推｜结构化带步长访存，不暴露指针】 stride_store(src: tla.buffer, dst: tla.buffer, offset: tla.expr, strides: tuple[tla.expr, ...], numels: tuple[tla.expr, ...]) -> None stride_load(src: tla.buffer, dst: tla.buffer, offset: tla.expr, other: tla.scalar, strides: tuple[tla.expr, ...], numels: tuple[tla.expr, ...]) -> None # rank<=3；numels 是有效元素数（越界预测），非重复次数；仅 AIV 【SIMD 级｜VF 内逐寄存器写】 simd.store_with_stride(dst: tla.buffer, indices, stride: int, mask: tla.mask, val: tla.vreg) -> None # stride 必须编译期常量且为 32B 块整数倍；无 post-update 【流式推进｜用循环携带地址，不提供可变指针对象】 for addr in tla.carry(init_addr): # 下沉 scf.for iter_args buf = tla.make_tensor(addr, ...) addr = addr + step`
