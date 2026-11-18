# SIMD接口规格

[返回接口目录](../README.md)

<a name="api-tla-scope-vector"></a>

## tla.scope.vector

签名：`tla.scope.vector()`

返回：无独立返回值。通过 with 声明内核区域。

将区域标记为 Vector 核模式，可在其内部声明 SIMD 区域。

**调用示例**

```python
@tla.jit
def example(x, dst_ub):
    with tla.scope.vector():
        tla.load_gm_to_ub(tla.view(x, (64,)), dst_ub)
```

仓库测试：[tla_ut/scope/pointer_abi.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scope/pointer_abi.py#L14)。

<a name="api-tla-scope-simd"></a>

## tla.scope.simd

签名：`tla.scope.simd()`

返回：无独立返回值。通过 with 声明内核区域。

将区域标记为 AIV SIMD 向量函数，供后端提取为独立的向量函数。

**调用示例**

两个 UB 缓冲区均为连续 64 个 float32 元素。

```python
@tla.jit
def example(x_ub, y_ub):
    with tla.scope.simd():
        tile = tla.simd.vlds(x_ub[0])
        tla.simd.vsts(y_ub[0], tile, dist='NORM_B32')
```

仓库测试：[tla_ut/scope/simd_only.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scope/simd_only.py#L25)。

<a name="api-tla-simd-membar"></a>

## tla.simd.membar

签名：`tla.simd.membar(scope)`

返回：None。

编译表示：ave.hir.membar。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `scope` | 指定执行域或地址空间 | 编译期 str | VV_ALL / VST_VLD / VLD_VST / VST_VST / VS_ALL / VST_LD / VLD_ST / VST_ST / SV_ALL / ST_VLD / LD_VST / ST_VST / SS_ALL / ST_LD / LD_ST / ST_ST | SIMD VF 内的标量或向量 load/store 屏障，按访问顺序选择。 | 必填 |

**调用示例**

两个 UB 缓冲区均为 64 个 float32 元素。

```python
@tla.jit
def example(src_ub, dst_ub):
    with tla.scope.simd():
        values = tla.simd.vlds(src_ub[0])
        tla.simd.vsts(dst_ub[0], values, dist='NORM_B32')
        tla.simd.membar('VST_VLD')
        next_values = tla.simd.vlds(dst_ub[0])
```

仓库测试：[tla_ut/vector/norm_forward.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/vector/norm_forward.py#L49)。

<a name="api-tla-simd-pand"></a>

## tla.simd.pand

签名：`tla.simd.pand(src0, src1, mask=None)`

返回：谓词向量。

编译表示：ave.hir.preg。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src0` | 第一个源操作数 | tla.vector（int1 谓词） | 数据类型：int1 | 各谓词 lane 数相同。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector（int1 谓词） | 数据类型：int1 | 各谓词 lane 数相同。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |

**调用示例**

```python
@tla.jit
def example():
    with tla.scope.simd():
        a = tla.simd.pge(32, 64)
        b = tla.simd.pge(16, 64)
        mask = tla.simd.pand(a, b)
```

仓库测试：[tla_ut/simd/pand.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/pand.py#L43)。

<a name="api-tla-simd-pnot"></a>

## tla.simd.pnot

签名：`tla.simd.pnot(src, mask=None)`

返回：谓词向量。

编译表示：ave.hir.preg。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | tla.vector（int1 谓词） | 数据类型：int1 | 各谓词 lane 数相同。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |

**调用示例**

```python
@tla.jit
def example():
    with tla.scope.simd():
        a = tla.simd.pge(32, 64)
        mask = tla.simd.pnot(a)
```

仓库测试：[tla_ut/simd/pnot.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/pnot.py#L34)。

<a name="api-tla-simd-pset"></a>

## tla.simd.pset

签名：`tla.simd.pset(elem_width, dist=tla.simd.PAT.ALL)`

返回：谓词向量。

编译表示：ave.hir.pge。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `elem_width` | 数据元素位宽 | 编译期 int | 8 / 16 / 32 / 64 | 输出 int1 谓词，lane 数为 2048 ÷ elem_width。 | 必填 |
| `dist` | 指令的分布模式 | 编译期 str | ALL / VL1 / VL2 / VL3 / VL4 / VL8 / VL16 / VL32 / VL64 / VL128 / M3 / M4 / H / Q / ALLF | 硬件图样需与谓词宽度配合。 | PAT.ALL |

**调用示例**

```python
@tla.jit
def example():
    with tla.scope.simd():
        mask = tla.simd.pset(32)
```

仓库测试：[tla_ut/simd/pat_mask.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/pat_mask.py#L46)。

<a name="api-tla-simd-pxor"></a>

## tla.simd.pxor

签名：`tla.simd.pxor(src0, src1, mask=None)`

返回：谓词向量。

编译表示：ave.hir.preg。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src0` | 第一个源操作数 | tla.vector（int1 谓词） | 数据类型：int1 | 各谓词 lane 数相同。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector（int1 谓词） | 数据类型：int1 | 各谓词 lane 数相同。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |

**调用示例**

```python
@tla.jit
def example():
    with tla.scope.simd():
        a = tla.simd.pge(32, 64)
        b = tla.simd.pge(16, 64)
        mask = tla.simd.pxor(a, b)
```

仓库测试：[tla_ut/simd/pxor.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/pxor.py#L43)。

<a name="api-tla-simd-vabs"></a>

## tla.simd.vabs

签名：`tla.simd.vabs(src, mask=None, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.vabs。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：int8<br>B16：int16 / float16<br>B32：int32 / float32<br>B64：int64 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

**调用示例**

输入为各含 64 个 float32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        result = tla.simd.vabs(a)
```

仓库测试：[tla_ut/simd/vabs.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/vabs.py#L46)。

<a name="api-tla-simd-vadd"></a>

## tla.simd.vadd

签名：`tla.simd.vadd(src0, src1, mask=None, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.vadd。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src0` | 第一个源操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 参与计算的向量需同 dtype、同 lane 数。 B64 走 RegBase 库函数，使用 2×VL 表示。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 参与计算的向量需同 dtype、同 lane 数。 B64 走 RegBase 库函数，使用 2×VL 表示。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

**调用示例**

输入为各含 64 个 float32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub, b_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        b = tla.simd.vlds(b_ub[0])
        result = tla.simd.vadd(a, b)
```

仓库测试：[tla_ut/scope/simd_only.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scope/simd_only.py#L27)。

<a name="api-tla-simd-vaddc"></a>

## tla.simd.vaddc

签名：`tla.simd.vaddc(src0, src1, mask=None)`

返回：结果向量与进位谓词。

编译表示：ave.hir.vadd / ave.hir.vcmp。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src0` | 第一个源操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：不支持<br>B32：int32 / uint32<br>B64：不支持 | 返回模 2³² 的和与进位谓词。比较按无符号解释。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：不支持<br>B32：int32 / uint32<br>B64：不支持 | 返回模 2³² 的和与进位谓词。比较按无符号解释。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |

**调用示例**

输入为各含 64 个 int32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub, b_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        b = tla.simd.vlds(b_ub[0])
        result, carry = tla.simd.vaddc(a, b)
```

仓库测试：[tla_ut/simd/add_cmp.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/add_cmp.py#L117)。

<a name="api-tla-simd-vadds"></a>

## tla.simd.vadds

签名：`tla.simd.vadds(src, scalar, mask=None, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.vadds。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 参与计算的向量需同 dtype、同 lane 数。 B64 走 RegBase 库函数，使用 2×VL 表示。 | 必填 |
| `scalar` | 广播给各 lane 的标量操作数 | int / float / 标量 tl.tensor | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 标量 tensor 与源 dtype 一致。整数常量使用源 dtype 的数值范围。前端也接收单 lane 向量，但 AVE scalar 操作数要求标量。 B64 大整数常量经由 double 构造，需保持精确值时传入同 dtype 的标量 tensor。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

**调用示例**

输入为各含 64 个 float32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        result = tla.simd.vadds(a, 2.0)
```

仓库测试：[tla_ut/simd/add_cmp.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/add_cmp.py#L102)。

<a name="api-tla-simd-vand"></a>

## tla.simd.vand

签名：`tla.simd.vand(src0, src1, mask=None, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.vand / ave.hir.vsel。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src0` | 第一个源操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 参与计算的向量需同 dtype、同 lane 数。 B64 走 RegBase 库函数，使用 2×VL 表示。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 参与计算的向量需同 dtype、同 lane 数。 B64 走 RegBase 库函数，使用 2×VL 表示。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | MERGING 保留第一个向量操作数的 inactive lane，ZEROING 将其置零。 | MODE.ZEROING |

**调用示例**

输入为各含 64 个 int32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub, b_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        b = tla.simd.vlds(b_ub[0])
        result = tla.simd.vand(a, b)
```

仓库测试：[tla_ut/simd/pat_mask.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/pat_mask.py#L47)。

<a name="api-tla-simd-vaxpy"></a>

## tla.simd.vaxpy

签名：`tla.simd.vaxpy(acc, src, scalar, mask=None, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.vmuls / ave.hir.vadd / ave.hir.vsel。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `acc` | 累加器 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `scalar` | 广播给各 lane 的标量操作数 | int / float / 标量 tl.tensor | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 标量 tensor 与源 dtype 一致。整数常量使用源 dtype 的数值范围。前端也接收单 lane 向量，但 AVE scalar 操作数要求标量。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | MERGING 保留第一个向量操作数的 inactive lane，ZEROING 将其置零。 | MODE.ZEROING |

**调用示例**

输入为各含 64 个 float32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub, b_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        b = tla.simd.vlds(b_ub[0])
        result = tla.simd.vaxpy(a, b, 2.0)
```

仓库测试：[tla_ut/simd/vaxpy.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/vaxpy.py#L38)。

<a name="api-tla-simd-vcadd"></a>

## tla.simd.vcadd

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`def vcadd( src: tla.vector, mask: tla.vector = None, mode: tl.constexpr = tla.simd.MODE.ZEROING, ) -> tla.vector`

<a name="api-tla-simd-vcgadd"></a>

## tla.simd.vcgadd

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`def vcgadd( src: tla.vector, mask: tla.vector = None, mode: tl.constexpr = tla.simd.MODE.ZEROING, ) -> tla.vector`

<a name="api-tla-simd-vcgmax"></a>

## tla.simd.vcgmax

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`def vcgmax( src: tla.vector, mask: tla.vector = None, mode: tl.constexpr = tla.simd.MODE.ZEROING, ) -> tla.vector`

<a name="api-tla-simd-vcgmin"></a>

## tla.simd.vcgmin

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`def vcgmin( src: tla.vector, mask: tla.vector = None, mode: tl.constexpr = tla.simd.MODE.ZEROING, ) -> tla.vector`

<a name="api-tla-simd-vci"></a>

## tla.simd.vci

签名：`tla.simd.vci(base)`

返回：tla.vector。

编译表示：ave.hir.vci。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `base` | 序列的起始值 | 编译期 int / 标量 int32 tl.tensor | −2³¹ 至 2³¹−1 | 结果固定为 64 lane int32 递增序列。 | 必填 |

**调用示例**

```python
@tla.jit
def example():
    with tla.scope.simd():
        indices = tla.simd.vci(0)
```

仓库测试：[tla_ut/simd/pand.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/pand.py#L38)。

<a name="api-tla-simd-vcmax"></a>

## tla.simd.vcmax

签名：`tla.simd.vcmax(src, mask=None, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.reduction。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 规约结果放在 lane 0，其余 lane 不应当作规约结果使用。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING | MERGING 会报错。 | MODE.ZEROING |

**调用示例**

输入为各含 64 个 float32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        result = tla.simd.vcmax(a)
```

仓库测试：[tla_ut/simd/add_cmp.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/add_cmp.py#L105)。

<a name="api-tla-simd-vcmin"></a>

## tla.simd.vcmin

签名：`tla.simd.vcmin(src, mask=None, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.reduction。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 规约结果放在 lane 0，其余 lane 不应当作规约结果使用。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING | MERGING 会报错。 | MODE.ZEROING |

**调用示例**

输入为各含 64 个 float32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        result = tla.simd.vcmin(a)
```

仓库测试：[tla_ut/simd/add_cmp.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/add_cmp.py#L106)。

<a name="api-tla-simd-vcmp"></a>

## tla.simd.vcmp

签名：`tla.simd.vcmp(src0, src1, mask=None, op=tla.simd.CMP.EQ)`

返回：谓词向量。

编译表示：ave.hir.vcmp。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src0` | 第一个源操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 参与计算的向量需同 dtype、同 lane 数。 B64 走 RegBase 库函数，使用 2×VL 表示。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 参与计算的向量需同 dtype、同 lane 数。 B64 走 RegBase 库函数，使用 2×VL 表示。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `op` | 比较关系 | 编译期 tla.simd.CMP 或 str | EQ / NE / GT / GE / LT / LE / UGT / UGE / ULT / ULE | 有符号和无符号的有序比较按输入 dtype 区分。浮点使用浮点比较式。 | CMP.EQ |

**调用示例**

输入为各含 64 个 float32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub, b_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        b = tla.simd.vlds(b_ub[0])
        result = tla.simd.vcmp(a, b)
```

仓库测试：[tla_ut/simd/pnot.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/pnot.py#L30)。

<a name="api-tla-simd-vcmps"></a>

## tla.simd.vcmps

签名：`tla.simd.vcmps(src, scalar, mask=None, op=tla.simd.CMP.LT)`

返回：谓词向量。

编译表示：ave.hir.broadcast / ave.hir.vcmp。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `scalar` | 广播给各 lane 的标量操作数 | int / float / 标量 tl.tensor | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 标量常量按向量 dtype 构造，运行时标量需符合当前转换路径。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `op` | 比较关系 | 编译期 tla.simd.CMP 或 str | EQ / NE / GT / GE / LT / LE / UGT / UGE / ULT / ULE | 有符号和无符号的有序比较按输入 dtype 区分。浮点使用浮点比较式。 | CMP.LT |

**调用示例**

输入为各含 64 个 float32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        result = tla.simd.vcmps(a, 2.0)
```

仓库测试：[tla_ut/simd/pst.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/pst.py#L41)。

<a name="api-tla-simd-vcvt"></a>

## tla.simd.vcvt

签名：`tla.simd.vcvt(src, target_dtype, mask=None, round=tla.simd.ROUND.R, sat=True, part=tla.simd.PART.EVEN, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.vtruncf / ave.hir.vextf / ave.hir.vsitofp / ave.hir.vuitofp / ave.hir.vfptosi / ave.hir.vfptoui。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | tla.vector | 浮点：float16 / bfloat16 / float8e4m3fn / float8e5m2 → float32，float32 → float16 / bfloat16 / float8e4m3fn / float8e5m2<br>有符号整数→浮点：int8 → float16，int16 → float16 / float32，int32 / int64 → float32<br>无符号整数→浮点：uint8 → float16，uint64 → float32<br>浮点→有符号整数：float16 → int8 / int16 / int32，bfloat16 → int32，float32 → int16 / int32 / int64<br>浮点→无符号整数：float16 → uint8 | 类型对取自 ODS、转换 verifier 和直接 intrinsic 分支。要求总位数可整除目标元素位宽。舍入、饱和、part 是否参与因类型对而异。 | 必填 |
| `target_dtype` | 目标元素类型 | 编译期 tl.dtype | 浮点：float16 / bfloat16 / float8e4m3fn / float8e5m2 → float32，float32 → float16 / bfloat16 / float8e4m3fn / float8e5m2<br>有符号整数→浮点：int8 → float16，int16 → float16 / float32，int32 / int64 → float32<br>无符号整数→浮点：uint8 → float16，uint64 → float32<br>浮点→有符号整数：float16 → int8 / int16 / int32，bfloat16 → int32，float32 → int16 / int32 / int64<br>浮点→无符号整数：float16 → uint8 | 拒绝 int→int。须同时满足对应转换 op 的源/目标类型、lane 数和属性约束。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。按源向量 lane 匹配。 | None |
| `round` | 数值转换的舍入方式 | 编译期 tla.simd.ROUND 常量或 str | rint / round / floor / ceil / trunc / odd | 实际可用舍入依源→目标类型对。 | ROUND.R |
| `sat` | 是否请求饱和 | 编译期 bool | False / True | 后端是否消费由实际转换 op 决定。 | True |
| `part` | 选择偶数或奇数槽位 | 编译期 tla.simd.PART 常量或 str | part_even / part_odd | 按转换或 packing 的槽位规则解释。 | PART.EVEN |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING | MERGING 会报错。 | MODE.ZEROING |

**调用示例**

输入为各含 64 个 int32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        result = tla.simd.vcvt(a, tl.float32)
```

仓库测试：[tla_ut/simd/vcvt.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/vcvt.py#L36)。

<a name="api-tla-simd-vdintlv"></a>

## tla.simd.vdintlv

签名：`tla.simd.vdintlv(src1, src2)`

返回：两个解交错后的向量。

编译表示：ave.hir.vdintlv。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src1` | 第二个源操作数 | tla.vector | 数据类型：<br>B8：float8e4m3fn / float8e5m2 / int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 两个输入同 dtype、同 lanes。构造 SPARSE 去交错，返回两个同型值。 | 必填 |
| `src2` | 第二个输入向量 | tla.vector | 数据类型：<br>B8：float8e4m3fn / float8e5m2 / int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 两个输入同 dtype、同 lanes。构造 SPARSE 去交错，返回两个同型值。 | 必填 |

**调用示例**

输入为各含 64 个 float32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub, b_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        b = tla.simd.vlds(b_ub[0])
        even, odd = tla.simd.vdintlv(a, b)
```

仓库测试：[tla_ut/vector/vector_type.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/vector/vector_type.py#L90)。

<a name="api-tla-simd-vdiv"></a>

## tla.simd.vdiv

签名：`tla.simd.vdiv(src0, src1, mask=None, mode=tla.simd.MODE.ZEROING, precision=None)`

返回：tla.vector。

编译表示：ave.hir.vdiv / ave.hir.vdivfhp / ave.hir.vsel。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src0` | 第一个源操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |
| `precision` | 精度和 FTZ 方式 | 编译期 str 或 None | None / ftz_true / exact / vdiv_0ulp_ftz_true | exact 路径仅支持 float32。vexp 的 ftz_false 路径会拒绝。 | None |

**调用示例**

输入为各含 64 个 float32 元素的连续 UB 缓冲区。 示例使用非零的除数。

```python
@tla.jit
def example(a_ub, b_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        b = tla.simd.vlds(b_ub[0])
        result = tla.simd.vdiv(a, b)
```

仓库测试：[tla_ut/simd/elementwise.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/elementwise.py#L57)。

<a name="api-tla-simd-vdups"></a>

## tla.simd.vdups

签名：`tla.simd.vdups(scalar, dtype, mask=None, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.broadcast。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `scalar` | 广播给各 lane 的标量操作数 | int / float / 标量 tl.tensor | 数据类型：<br>B8：float8e4m3fn / int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 标量常量按向量 dtype 构造，运行时标量需符合当前转换路径。 | 必填 |
| `dtype` | 元素类型 | 编译期 tl.dtype 或 dtype 名称 str | 数据类型：<br>B8：float8e4m3fn / int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 用对应元素类型将标量扩成满寄存器。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

**调用示例**

```python
@tla.jit
def example():
    with tla.scope.simd():
        tile = tla.simd.vdups(1.0, tl.float32)
```

仓库测试：[tla_ut/simd/simd_tail.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/simd_tail.py#L105)。

<a name="api-tla-simd-vdupv"></a>

## tla.simd.vdupv

签名：`tla.simd.vdupv(src, mask=None, pos=tla.simd.POS.LOWEST, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.vector_broadcast。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：float8e4m3fn / int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `pos` | 选择广播的源 lane | 编译期 str | POS_LOWEST / LOWEST / POS_HIGHEST / HIGHEST | 选择原向量的第一个或最后一个 lane。 | POS.LOWEST |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

**调用示例**

输入为各含 64 个 float32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        result = tla.simd.vdupv(a)
```

仓库测试：[tla_ut/simd/simd_tail.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/simd_tail.py#L106)。

<a name="api-tla-simd-vexp"></a>

## tla.simd.vexp

签名：`tla.simd.vexp(src, mask=None, mode=tla.simd.MODE.ZEROING, precision=None)`

返回：tla.vector。

编译表示：ave.hir.vexp。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：不支持<br>B16：float16<br>B32：float32<br>B64：不支持 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |
| `precision` | 精度和 FTZ 方式 | 编译期 str 或 None | None / ftz_true | exact 路径仅支持 float32。vexp 的 ftz_false 路径会拒绝。 | None |

**调用示例**

输入为各含 64 个 float32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        result = tla.simd.vexp(a)
```

仓库测试：[tla_ut/simd/elementwise.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/elementwise.py#L61)。

<a name="api-tla-simd-vexpdif"></a>

## tla.simd.vexpdif

签名：`tla.simd.vexpdif(lhs, rhs, mask=None, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.vexpdif。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `lhs` | 左操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：不支持<br>B32：float32<br>B64：不支持 | 发出融合 exp(lhs−rhs)，不能假定分步实现有相同舍入。 | 必填 |
| `rhs` | 右操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：不支持<br>B32：float32<br>B64：不支持 | 发出融合 exp(lhs−rhs)，不能假定分步实现有相同舍入。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

**调用示例**

输入为各含 64 个 float32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub, b_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        b = tla.simd.vlds(b_ub[0])
        result = tla.simd.vexpdif(a, b)
```

仓库测试：[tla_ut/simd/vexpdif.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/vexpdif.py#L46)。

<a name="api-tla-simd-vgather"></a>

## tla.simd.vgather

签名：`tla.simd.vgather(ptr, index, mask=None, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.vgather。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `ptr` | 缓冲区或索引得到的地址 | tla.buffer / buffer_access | 地址空间：UB<br>数据类型：<br>B8：不支持<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 一维连续。活跃索引须在有效范围内，mask 抑制无效 lane 的读取。 | 必填 |
| `index` | 按元素计的索引向量 | tla.vector | B16 payload：uint16，128 lanes<br>B32 payload：uint32，64 lanes | 单位为元素，活跃索引不得越界。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

**调用示例**

x_ub 为 float32，indices_ub 为 64 个 uint32 元素索引，均在 x_ub 有效范围内。

```python
@tla.jit
def example(x_ub, indices_ub):
    with tla.scope.simd():
        indices = tla.simd.vlds(indices_ub[0])
        values = tla.simd.vgather(x_ub[0], indices)
```

仓库测试：[tla_ut/simd/vld.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/vld.py#L49)。

<a name="api-tla-simd-vintlv"></a>

## tla.simd.vintlv

签名：`tla.simd.vintlv(src1, src2)`

返回：两个交错后的向量。

编译表示：ave.hir.vintlv。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src1` | 第二个源操作数 | tla.vector | 数据类型：<br>B8：float8e4m3fn / float8e5m2 / int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 两个输入同 dtype、同 lanes。构造 SPARSE 交错，返回两个同型值。 | 必填 |
| `src2` | 第二个输入向量 | tla.vector | 数据类型：<br>B8：float8e4m3fn / float8e5m2 / int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 两个输入同 dtype、同 lanes。构造 SPARSE 交错，返回两个同型值。 | 必填 |

**调用示例**

输入为各含 64 个 float32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub, b_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        b = tla.simd.vlds(b_ub[0])
        lo, hi = tla.simd.vintlv(a, b)
```

仓库测试：[tla_ut/vector/vector_type.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/vector/vector_type.py#L89)。

<a name="api-tla-simd-vld"></a>

## tla.simd.vld

签名：`tla.simd.vld(ptr, index, mask=None, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.vgather。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `ptr` | 缓冲区或索引得到的地址 | tla.buffer / buffer_access | 地址空间：UB<br>数据类型：<br>B8：不支持<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 一维连续，按活跃索引读取。 | 必填 |
| `index` | 按元素计的索引向量 | tla.vector | B16 payload：uint16，128 lanes<br>B32 payload：uint32，64 lanes | 单位为元素，活跃索引不得越界。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

**调用示例**

x_ub 为 float32，indices_ub 为 64 个 uint32 元素索引，均在 x_ub 有效范围内。

```python
@tla.jit
def example(x_ub, indices_ub):
    with tla.scope.simd():
        indices = tla.simd.vlds(indices_ub[0])
        values = tla.simd.vld(x_ub[0], indices)
```

仓库测试：[tla_ut/simd/vld.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/vld.py#L50)。

<a name="api-tla-simd-vld2"></a>

## tla.simd.vld2

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`def vld2( src: tla.buffer, dist: tl.constexpr = tla.simd.LOAD_DIST.DINTLV_B16, off=None, # 附加偏移操作数（标量/表达式） ) -> tuple[tla.vector, tla.vector]: # (even, odd)`

<a name="api-tla-simd-vln"></a>

## tla.simd.vln

签名：`tla.simd.vln(src, mask=None, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.vln / ave.hir.vsel。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：不支持<br>B16：float16<br>B32：float32<br>B64：不支持 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | MERGING 保留第一个向量操作数的 inactive lane，ZEROING 将其置零。 | MODE.ZEROING |

**调用示例**

输入为各含 64 个 float32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        result = tla.simd.vln(a)
```

仓库测试：[tla_ut/simd/vln.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/vln.py#L32)。

<a name="api-tla-simd-vmadd"></a>

## tla.simd.vmadd

签名：`tla.simd.vmadd(acc, src0, src1, mask=None, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.vmula / ave.hir.vsel。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `acc` | 累加器 | tla.vector | 数据类型：<br>B8：不支持<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `src0` | 第一个源操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | MERGING 保留第一个向量操作数的 inactive lane，ZEROING 将其置零。 | MODE.ZEROING |

**调用示例**

输入为各含 64 个 float32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub, b_ub, c_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        b = tla.simd.vlds(b_ub[0])
        c = tla.simd.vlds(c_ub[0])
        result = tla.simd.vmadd(a, b, c)
```

仓库测试：[tla_ut/simd/vmula_vmadd.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/vmula_vmadd.py#L57)。

<a name="api-tla-simd-vmax"></a>

## tla.simd.vmax

签名：`tla.simd.vmax(src0, src1, mask=None, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.vmax / ave.hir.vumax。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src0` | 第一个源操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

**调用示例**

输入为各含 64 个 float32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub, b_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        b = tla.simd.vlds(b_ub[0])
        result = tla.simd.vmax(a, b)
```

仓库测试：[tla_ut/simd/elementwise.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/elementwise.py#L58)。

<a name="api-tla-simd-vmaxs"></a>

## tla.simd.vmaxs

签名：`tla.simd.vmaxs(src0, scalar, mask=None, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.vmaxs / ave.hir.vumaxs / ave.hir.vsel。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src0` | 第一个源操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `scalar` | 广播给各 lane 的标量操作数 | int / float / 标量 tl.tensor | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 标量常量按向量 dtype 构造，运行时标量需符合当前转换路径。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | MERGING 保留第一个向量操作数的 inactive lane，ZEROING 将其置零。 | MODE.ZEROING |

**调用示例**

输入为各含 64 个 float32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        result = tla.simd.vmaxs(a, 2.0)
```

仓库测试：[tla_ut/simd/elementwise.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/elementwise.py#L60)。

<a name="api-tla-simd-vmin"></a>

## tla.simd.vmin

签名：`tla.simd.vmin(src0, src1, mask=None, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.vmin / ave.hir.vumin。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src0` | 第一个源操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

**调用示例**

输入为各含 64 个 float32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub, b_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        b = tla.simd.vlds(b_ub[0])
        result = tla.simd.vmin(a, b)
```

仓库测试：[tla_ut/simd/elementwise.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/elementwise.py#L59)。

<a name="api-tla-simd-vmins"></a>

## tla.simd.vmins

签名：`tla.simd.vmins(src, scalar, mask=None, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.vmins / ave.hir.vumins。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `scalar` | 广播给各 lane 的标量操作数 | int / float / 标量 tl.tensor | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 标量常量按向量 dtype 构造，运行时标量需符合当前转换路径。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

**调用示例**

输入为各含 64 个 float32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        result = tla.simd.vmins(a, 2.0)
```

仓库测试：[tla_ut/simd/vmins.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/vmins.py#L38)。

<a name="api-tla-simd-vmul"></a>

## tla.simd.vmul

签名：`tla.simd.vmul(lhs, rhs, mask=None, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.vmul。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `lhs` | 左操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 | 参与计算的向量需同 dtype、同 lane 数。 B64 走 RegBase 库函数，使用 2×VL 表示，仅支持 int64。 | 必填 |
| `rhs` | 右操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 | 参与计算的向量需同 dtype、同 lane 数。 B64 走 RegBase 库函数，使用 2×VL 表示，仅支持 int64。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

**调用示例**

输入为各含 64 个 float32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub, b_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        b = tla.simd.vlds(b_ub[0])
        result = tla.simd.vmul(a, b)
```

仓库测试：[tla_ut/scope/multi_vf.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scope/multi_vf.py#L38)。

<a name="api-tla-simd-vmula"></a>

## tla.simd.vmula

签名：`tla.simd.vmula(acc, src0, src1, mask=None, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.vmula / ave.hir.vsel。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `acc` | 累加器 | tla.vector | 数据类型：<br>B8：不支持<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `src0` | 第一个源操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | MERGING 保留第一个向量操作数的 inactive lane，ZEROING 将其置零。 | MODE.ZEROING |

**调用示例**

输入为各含 64 个 float32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub, b_ub, c_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        b = tla.simd.vlds(b_ub[0])
        c = tla.simd.vlds(c_ub[0])
        result = tla.simd.vmula(a, b, c)
```

仓库测试：[tla_ut/simd/pat_mask.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/pat_mask.py#L52)。

<a name="api-tla-simd-vmull"></a>

## tla.simd.vmull

签名：`tla.simd.vmull(src0, src1, mask=None)`

返回：乘积的低位向量与高位向量。

编译表示：ave.hir.mull。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src0` | 第一个源操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：不支持<br>B32：int32 / uint32<br>B64：不支持 | 两个输入同型，返回同型低半部和高半部。uint32 的最高位场景应验证高半部符号解释。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：不支持<br>B32：int32 / uint32<br>B64：不支持 | 两个输入同型，返回同型低半部和高半部。uint32 的最高位场景应验证高半部符号解释。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |

**调用示例**

输入为各含 64 个 int32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub, b_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        b = tla.simd.vlds(b_ub[0])
        lo, hi = tla.simd.vmull(a, b)
```

仓库测试：[tla_ut/simd/vmull.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/vmull.py#L35)。

<a name="api-tla-simd-vmuls"></a>

## tla.simd.vmuls

签名：`tla.simd.vmuls(src, scalar, mask=None, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.vmuls / ave.hir.vsel。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 参与计算的向量需同 dtype、同 lane 数。 B64 走 RegBase 库函数，使用 2×VL 表示。 | 必填 |
| `scalar` | 广播给各 lane 的标量操作数 | int / float / 标量 tl.tensor | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 标量 tensor 与源 dtype 一致。整数常量使用源 dtype 的数值范围。前端也接收单 lane 向量，但 AVE scalar 操作数要求标量。 B64 大整数常量经由 double 构造，需保持精确值时传入同 dtype 的标量 tensor。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | MERGING 保留第一个向量操作数的 inactive lane，ZEROING 将其置零。 | MODE.ZEROING |

**调用示例**

输入为各含 64 个 float32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        result = tla.simd.vmuls(a, 2.0)
```

仓库测试：[tla_ut/simd/vcvt.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/vcvt.py#L54)。

<a name="api-tla-simd-vneg"></a>

## tla.simd.vneg

签名：`tla.simd.vneg(src, mask=None, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.vneg。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

**调用示例**

输入为各含 64 个 float32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        result = tla.simd.vneg(a)
```

仓库测试：[tla_ut/simd/vneg.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/vneg.py#L46)。

<a name="api-tla-simd-vor"></a>

## tla.simd.vor

签名：`tla.simd.vor(lhs, rhs, mask=None, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.vor。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `lhs` | 左操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 | 参与计算的向量需同 dtype、同 lane 数。 B64 走 RegBase 库函数，使用 2×VL 表示，仅支持 int64。 | 必填 |
| `rhs` | 右操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 | 参与计算的向量需同 dtype、同 lane 数。 B64 走 RegBase 库函数，使用 2×VL 表示，仅支持 int64。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

**调用示例**

输入为各含 64 个 int32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub, b_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        b = tla.simd.vlds(b_ub[0])
        result = tla.simd.vor(a, b)
```

仓库测试：[tla_ut/simd/vor.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/vor.py#L38)。

<a name="api-tla-simd-vpack"></a>

## tla.simd.vpack

签名：`tla.simd.vpack(src, part='part_even')`

返回：tla.vector。

编译表示：ave.hir.vpack。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：不支持<br>B16：int16 / uint16<br>B32：int32 / uint32<br>B64：不支持 | int16/uint16 → int8/uint8，int32/uint32 → int16/uint16。结果 2×输入 lanes，非选中槽位未指定。 | 必填 |
| `part` | 选择偶数或奇数槽位 | 编译期 tla.simd.PART 常量或 str | part_even / part_odd | 按转换或 packing 的槽位规则解释。 | 'part_even' |

**调用示例**

输入为各含 128 个 int16 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        result = tla.simd.vpack(a)
```

仓库测试：[tla_ut/simd/vpack.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/vpack.py#L24)。

<a name="api-tla-simd-vrelu"></a>

## tla.simd.vrelu

签名：`tla.simd.vrelu(src, mask=None, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.vrelu。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：不支持<br>B16：float16<br>B32：int32 / float32<br>B64：不支持 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

**调用示例**

输入为各含 64 个 float32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        result = tla.simd.vrelu(a)
```

仓库测试：[tla_ut/simd/vrelu.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/vrelu.py#L50)。

<a name="api-tla-simd-vscatter"></a>

## tla.simd.vscatter

签名：`tla.simd.vscatter(dst, indices, src, mask)`

返回：None。

编译表示：ave.hir.vscatter。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `dst` | 目标缓冲区 | tla.buffer | 地址空间：UB<br>数据类型：<br>B8：不支持<br>B16：不支持<br>B32：uint32<br>B64：不支持 | dst 必须为一维连续、零 offset 的 UB buffer。src、indices 固定 64 lanes。活跃索引不得重复或越界。 | 必填 |
| `indices` | 索引向量 | tla.vector | 数据类型：uint32<br>lane 数：64 | 按元素计，活跃索引不得重复或越界。 | 必填 |
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：不支持<br>B16：不支持<br>B32：uint32<br>B64：不支持 | dst 必须为一维连续、零 offset 的 UB buffer。src、indices 固定 64 lanes。活跃索引不得重复或越界。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） | 数据类型：int1<br>lane 数：64 | — | 必填 |

**调用示例**

三者均为 uint32 UB 缓冲区，活跃索引互异且位于 dst_ub 范围内。

```python
@tla.jit
def example(src_ub, indices_ub, dst_ub):
    with tla.scope.simd():
        values = tla.simd.vlds(src_ub[0])
        indices = tla.simd.vlds(indices_ub[0])
        tla.simd.vscatter(dst_ub, indices, values, tla.simd.pge(32, 64))
```

<a name="api-tla-simd-vsel"></a>

## tla.simd.vsel

签名：`tla.simd.vsel(mask, true_val, false_val)`

返回：tla.vector。

编译表示：ave.hir.vsel。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） | 数据类型：int1<br>lane 数：与相应向量一致 | — | 必填 |
| `true_val` | 参数 | tla.vector | 数据类型：<br>B1：int1<br>B8：float8e4m3fn / float8e5m2 / int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 参与计算的向量需同 dtype、同 lane 数。 B64 走 RegBase 库函数，使用 2×VL 表示。 | 必填 |
| `false_val` | 参数 | tla.vector | 数据类型：<br>B1：int1<br>B8：float8e4m3fn / float8e5m2 / int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 参与计算的向量需同 dtype、同 lane 数。 B64 走 RegBase 库函数，使用 2×VL 表示。 | 必填 |

**调用示例**

输入为各含 64 个 float32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub, b_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        b = tla.simd.vlds(b_ub[0])
        result = tla.simd.vsel(tla.simd.pge(32, 64), a, b)
```

仓库测试：[tla_ut/simd/tail_mask.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/tail_mask.py#L29)。

<a name="api-tla-simd-vselr"></a>

## tla.simd.vselr

签名：`tla.simd.vselr(src, indices)`

返回：tla.vector。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：不支持<br>B16：uint16<br>B32：不支持<br>B64：不支持 | 两个输入均为 128 lanes。索引表示源 lane。 | 必填 |
| `indices` | 索引向量 | tla.vector | 数据类型：uint16<br>lane 数：128<br>正常索引：0 至 127 | 按源向量 lane 编号寻址。 | 必填 |

**调用示例**

输入及索引均为 128 个 uint16 元素，索引位于 0–127。

```python
@tla.jit
def example(src_ub, indices_ub):
    with tla.scope.simd():
        values = tla.simd.vlds(src_ub[0])
        indices = tla.simd.vlds(indices_ub[0])
        result = tla.simd.vselr(values, indices)
```

<a name="api-tla-simd-vshl"></a>

## tla.simd.vshl

签名：`tla.simd.vshl(src0, src1, mask=None, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.vshl。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src0` | 第一个源操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32<br>B64：不支持 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32<br>B64：不支持 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

**调用示例**

输入为各含 64 个 int32 元素的连续 UB 缓冲区。 位移量位于 0–31。

```python
@tla.jit
def example(a_ub, b_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        b = tla.simd.vlds(b_ub[0])
        result = tla.simd.vshl(a, b)
```

仓库测试：[tla_ut/simd/vshift.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/vshift.py#L31)。

<a name="api-tla-simd-vshls"></a>

## tla.simd.vshls

签名：`tla.simd.vshls(src, scalar, mask=None, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.vshls。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32<br>B64：不支持 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `scalar` | 广播给各 lane 的标量操作数 | int / 标量整数 tl.tensor | 编译期值：−32768 至 32767 | 最终物化为 src 的元素类型。按元素位宽测试 0 至 B−1 的正常位移。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

**调用示例**

输入为各含 64 个 int32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        result = tla.simd.vshls(a, 1)
```

仓库测试：[tla_ut/simd/vshift.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/vshift.py#L85)。

<a name="api-tla-simd-vshr"></a>

## tla.simd.vshr

签名：`tla.simd.vshr(src0, src1, mask=None, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.vshr。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src0` | 第一个源操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32<br>B64：不支持 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32<br>B64：不支持 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

**调用示例**

输入为各含 64 个 int32 元素的连续 UB 缓冲区。 位移量位于 0–31。

```python
@tla.jit
def example(a_ub, b_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        b = tla.simd.vlds(b_ub[0])
        result = tla.simd.vshr(a, b)
```

仓库测试：[tla_ut/simd/vshift.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/vshift.py#L50)。

<a name="api-tla-simd-vsqrt"></a>

## tla.simd.vsqrt

签名：`tla.simd.vsqrt(src, mask=None, mode=tla.simd.MODE.ZEROING, precision=None)`

返回：tla.vector。

编译表示：ave.hir.vsqrt。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：不支持<br>B16：float16<br>B32：float32<br>B64：不支持 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |
| `precision` | 精度和 FTZ 方式 | 编译期 str 或 None | None / ftz_true | 使用目标 SFU。 | None |

**调用示例**

输入为各含 64 个 float32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        result = tla.simd.vsqrt(a)
```

仓库测试：[tla_ut/simd/vsqrt.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/vsqrt.py#L47)。

<a name="api-tla-simd-vsstb"></a>

## tla.simd.vsstb

签名：`tla.simd.vsstb(src, dst, stride, mask=None, update=False)`

返回：None。

编译表示：ave.hir.store_with_stride。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：float8e4m3fn / float8e5m2 / int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | dst 为二维 UB 地址。dtype 与 src 相同，步长以元素计，32 字节为块。 | 必填 |
| `dst` | 目标缓冲区 | tla.buffer / buffer_access | 地址空间：UB<br>数据类型：<br>B8：float8e4m3fn / float8e5m2 / int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | dst 为二维 UB 地址。dtype 与 src 相同，步长以元素计，32 字节为块。 | 必填 |
| `stride` | 相邻传输块的步长 | 编译期 int | 正整数，换算成 32 字节块后在 1 至 65535 内 | 原值按元素计，必须整除对应的每块元素数。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `update` | 请求将存储地址写回 | 编译期 bool | False | True 抛 NotImplementedError。 | False |

**调用示例**

源为 FP16 向量；目的缓冲区为二维 UB，stride=16 表示以 32 字节为一块。

```python
@tla.jit
def example(x_ub, y_ub):
    with tla.scope.simd():
        values = tla.simd.vlds(x_ub[0])
        tla.simd.vsstb(values, y_ub[0, 0], 16)
```

仓库测试：[tla_ut/simd/vsstb.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/vsstb.py#L30)。

<a name="api-tla-simd-vsts"></a>

## tla.simd.vsts

签名：`tla.simd.vsts(ptr, offsets_or_val, val=None, *, dist=None, mask=None)`

返回：None。

编译表示：ave.hir.masked_store。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `ptr` | 缓冲区或索引得到的地址 | tla.buffer / buffer_access | 地址空间：UB<br>数据类型：<br>B1：int1<br>B8：float8e4m3fn / float8e5m2 / int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 普通 NORM/ONEPT 形式使用与写值一致的元素类型。目标范围须容纳分布产生的写入。 | 必填 |
| `offsets_or_val` | 省略 val 时是待写入值，提供 val 时是显式偏移 | 整数序列 / tla.vector | 偏移按元素计<br>数据类型：<br>B1：int1<br>B8：float8e4m3fn / float8e5m2 / int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 偏移元数匹配地址秩。省略 val 时第二个参数是写值。 | 必填 |
| `val` | 待存入的向量或谓词 | tla.vector 或 None | 数据类型：<br>B1：int1<br>B8：float8e4m3fn / float8e5m2 / int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 偏移元数匹配地址秩。省略 val 时第二个参数是写值。 | None |
| `dist` | 指令的分布模式 | 编译期 str | None（NORM_B32） / NORM_B8 / NORM_B16 / NORM_B32 / NORM_B64 / ONEPT_B8 / ONEPT_B16 / ONEPT_B32 / ONEPT_B64 | 分布后缀须与指令所需位宽对应。ONEPT 只存一个元素。packing 和通道合并需要单独验证源、目的与目标组合。 | None |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |

**调用示例**

两个 UB 缓冲区均为 64 个 float32 元素。

```python
@tla.jit
def example(x_ub, y_ub):
    with tla.scope.simd():
        values = tla.simd.vlds(x_ub[0])
        tla.simd.vsts(y_ub[0], values, dist='NORM_B32')
```

仓库测试：[tla_ut/scope/simd_only.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scope/simd_only.py#L27)。

<a name="api-tla-simd-vsub"></a>

## tla.simd.vsub

签名：`tla.simd.vsub(lhs, rhs, mask=None, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.vsub。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `lhs` | 左操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 | 参与计算的向量需同 dtype、同 lane 数。 B64 走 RegBase 库函数，使用 2×VL 表示，仅支持 int64。 | 必填 |
| `rhs` | 右操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 | 参与计算的向量需同 dtype、同 lane 数。 B64 走 RegBase 库函数，使用 2×VL 表示，仅支持 int64。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

**调用示例**

输入为各含 64 个 float32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub, b_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        b = tla.simd.vlds(b_ub[0])
        result = tla.simd.vsub(a, b)
```

仓库测试：[tla_ut/simd/vsub.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/vsub.py#L38)。

<a name="api-tla-simd-vxor"></a>

## tla.simd.vxor

签名：`tla.simd.vxor(lhs, rhs, mask=None, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.vxor。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `lhs` | 左操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 | 参与计算的向量需同 dtype、同 lane 数。 B64 走 RegBase 库函数，使用 2×VL 表示，仅支持 int64。 | 必填 |
| `rhs` | 右操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 | 参与计算的向量需同 dtype、同 lane 数。 B64 走 RegBase 库函数，使用 2×VL 表示，仅支持 int64。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

**调用示例**

输入为各含 64 个 int32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub, b_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        b = tla.simd.vlds(b_ub[0])
        result = tla.simd.vxor(a, b)
```

仓库测试：[tla_ut/simd/vxor.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/vxor.py#L38)。

<a name="api-tla-simd-pge"></a>

## tla.simd.pge

签名：`tla.simd.pge(n, lanes)`

返回：谓词向量。

编译表示：ave.hir.pge。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `n` | 前缀中为真的 lane 数 | 编译期 int | 1 / 2 / 3 / 4 / 8 / 16 / 32 / 64 / 128 | 前 n 个 lane 置 true，结果 lanes 至少要容纳 n。 | 必填 |
| `lanes` | 谓词 lane 数 | 编译期 int | 正整数，至多 256 | 结果为 int1。有效组合需匹配指定 VL 图样和消费指令。 | 必填 |

**调用示例**

```python
@tla.jit
def example():
    with tla.scope.simd():
        mask = tla.simd.pge(32, 64)
```

仓库测试：[tla_ut/simd/vsstb.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/vsstb.py#L29)。

<a name="api-tla-simd-tail-mask"></a>

## tla.simd.tail_mask

签名：`tla.simd.tail_mask(value, width=32)`

返回：谓词向量。

编译表示：ave.hir.vci / ave.hir.broadcast / ave.hir.vcmp。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `value` | 输入值 | 编译期 int / 标量 int32 tl.tensor | 0 至 64（尾块计数） | 超出正常域时当前走无符号比较，负值会表现为很大的无符号数。 | 必填 |
| `width` | 数据 lane 的逻辑位宽 | 编译期 int | 32 | 按逻辑元素位宽选择谓词密度。 | 32 |

**调用示例**

```python
@tla.jit
def example(n):
    with tla.scope.simd():
        mask = tla.simd.tail_mask(n)
```

仓库测试：[tla_ut/simd/tail_mask.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/tail_mask.py#L29)。

<a name="api-tla-simd-por"></a>

## tla.simd.por

签名：`tla.simd.por(src0, src1, mask=None)`

返回：谓词向量。

编译表示：ave.hir.preg。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src0` | 第一个源操作数 | tla.vector（int1 谓词） | 数据类型：int1 | 各谓词 lane 数相同。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector（int1 谓词） | 数据类型：int1 | 各谓词 lane 数相同。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |

**调用示例**

```python
@tla.jit
def example():
    with tla.scope.simd():
        a = tla.simd.pge(32, 64)
        b = tla.simd.pge(16, 64)
        mask = tla.simd.por(a, b)
```

仓库测试：[tla_ut/simd/por.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/por.py#L43)。

<a name="api-tla-simd-psel"></a>

## tla.simd.psel

签名：`tla.simd.psel(src0, src1, mask)`

返回：谓词向量。

编译表示：ave.hir.vsel。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src0` | 第一个源操作数 | tla.vector（int1 谓词） | 数据类型：int1 | 各谓词 lane 数相同。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector（int1 谓词） | 数据类型：int1 | 各谓词 lane 数相同。 | 必填 |
| `mask` | 为真选 src0，为假选 src1 | tla.vector（int1 谓词） | 数据类型：int1<br>lane 数：与相应向量一致 | — | 必填 |

**调用示例**

```python
@tla.jit
def example():
    with tla.scope.simd():
        a = tla.simd.pge(32, 64)
        b = tla.simd.pge(16, 64)
        mask = tla.simd.psel(a, b, a)
```

仓库测试：[tla_ut/simd/psel.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/psel.py#L41)。

<a name="api-tla-simd-ppack"></a>

## tla.simd.ppack

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`def ppack( src: tla.vector, part=tla.simd.PART.EVEN, ) -> tla.vector`

<a name="api-tla-simd-punpack"></a>

## tla.simd.punpack

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`def punpack( src: tla.vector, part=tla.simd.PART.EVEN, ) -> tla.vector`

<a name="api-tla-simd-pintlv"></a>

## tla.simd.pintlv

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`def pintlv( src0: tla.vector, src1: tla.vector, width=32, )-> tla.vector`

<a name="api-tla-simd-pdintlv"></a>

## tla.simd.pdintlv

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`def pdintlv( src0: tla.vector, src1: tla.vector, width=32, )-> tla.vector`

<a name="api-tla-simd-pld"></a>

## tla.simd.pld

签名：`tla.simd.pld(ptr, offsets=None, *, width=8, dist='NORM')`

返回：谓词向量。

编译表示：ave.hir.vload。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `ptr` | 缓冲区或索引得到的地址 | tla.buffer / buffer_access | 地址空间：UB<br>数据类型：<br>B8：int8 / uint8<br>B16：不支持<br>B32：不支持<br>B64：不支持 | 一维连续字节缓冲区，实际地址需 32 字节对齐。NORM/US 读 32 字节，DS 读 64 字节。 | 必填 |
| `offsets` | 基于源地址或视图的元素偏移 | 整数序列或 None | 偏移按元素计<br>整数偏移或 None | 地址需 32 字节对齐。None 使用零偏移。 | None |
| `width` | 数据 lane 的逻辑位宽 | 编译期 int | 8 / 16 / 32 | 按逻辑元素位宽选择谓词密度。 | 8 |
| `dist` | 指令的分布模式 | 编译期 str | NORM / US / DS | NORM 读取连续位。US 读第 floor(i/2) 位，DS 读第 2i 位。 | 'NORM' |

**调用示例**

mask_ub 为 32 字节对齐的 UB 字节缓冲区。

```python
@tla.jit
def example(mask_ub):
    with tla.scope.simd():
        mask = tla.simd.pld(mask_ub, width=32)
```

仓库测试：[tla_ut/simd/pld.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/pld.py#L40)。

<a name="api-tla-simd-pst"></a>

## tla.simd.pst

签名：`tla.simd.pst(ptr, offsets_or_val, val=None)`

返回：None。

编译表示：ave.hir.masked_store。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `ptr` | 缓冲区或索引得到的地址 | tla.buffer / buffer_access | 地址空间：UB<br>数据类型：<br>B8：int8 / uint8<br>B16：不支持<br>B32：不支持<br>B64：不支持 | 一维连续字节缓冲区，实际地址需 32 字节对齐。按 256/128/64 lanes 写 32/16/8 字节，最低位在前。 | 必填 |
| `offsets_or_val` | 省略 val 时是待写入值，提供 val 时是显式偏移 | 整数序列 / tla.vector（int1 谓词） | 偏移按元素计<br>数据类型：int1<br>lane 数：256 / 128 / 64 | 偏移元数匹配地址秩。省略 val 时第二个参数是写值。 | 必填 |
| `val` | 待存入的向量或谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：256 / 128 / 64 | 偏移元数匹配地址秩。省略 val 时第二个参数是写值。 | None |

**调用示例**

mask_ub 为 32 字节对齐的 UB 字节缓冲区。

```python
@tla.jit
def example(mask_ub):
    with tla.scope.simd():
        mask = tla.simd.pge(32, 64)
        tla.simd.pst(mask_ub, mask)
```

仓库测试：[tla_ut/simd/predicate_views.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/predicate_views.py#L51)。

<a name="api-tla-simd-vsubc"></a>

## tla.simd.vsubc

签名：`tla.simd.vsubc(src0, src1, mask=None)`

返回：结果向量与借位谓词。

编译表示：ave.hir.vsub / ave.hir.vcmp。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src0` | 第一个源操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：不支持<br>B32：int32 / uint32<br>B64：不支持 | 两个输入均为 64 lanes。返回模 2³² 的差与借位谓词，inactive 得到 0 和 false。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：不支持<br>B32：int32 / uint32<br>B64：不支持 | 两个输入均为 64 lanes。返回模 2³² 的差与借位谓词，inactive 得到 0 和 false。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：64 | None 生成全真 mask。 | None |

**调用示例**

输入为各含 64 个 int32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub, b_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        b = tla.simd.vlds(b_ub[0])
        result, borrow = tla.simd.vsubc(a, b)
```

仓库测试：[tla_ut/simd/vsubc.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/vsubc.py#L44)。

<a name="api-tla-simd-vaddcs"></a>

## tla.simd.vaddcs

签名：`tla.simd.vaddcs(src0, src1, carry, mask=None)`

返回：`(sum, carry_out)`。`sum` 为 tla.vector，`carry_out` 为 int1 谓词。

编译表示：ave.hir.vaddcs。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src0` | 左操作数 | tla.vector | 数据类型：<br>B32：int32 / uint32 | 必须为满 64 lanes，与 `src1` 同 dtype。 | 必填 |
| `src1` | 右操作数 | tla.vector | 数据类型：<br>B32：int32 / uint32 | 同上。 | 必填 |
| `carry` | 输入进位谓词 | tla.vector（int1 谓词） | 数据类型：int1<br>lane 数：64 | 必填；lane 为真表示该 lane 加 1。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词）或 None | 数据类型：int1<br>lane 数：64 | None 生成全真 mask；inactive lane 的 sum 为 0、carry_out 为假。 | None |

**语义**：`sum = src0 + src1 + carry`（按 2³² 取模），`carry_out` 为无符号进位。

**调用示例**

输入为各含 64 个 int32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub, b_ub, out_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        b = tla.simd.vlds(b_ub[0])
        total, carry_out = tla.simd.vaddcs(a, b, tla.simd.pge(0, 64))
```

仓库测试：[tla_ut/simd/carry.py](https://github.com/triton-lang/triton-ascend/blob/aa927de3356e671160f2b3cf29fe23b6072f2b16/third_party/ascend/unittest/tla_ut/simd/carry.py#L15)。

<a name="api-tla-simd-vsubcs"></a>

## tla.simd.vsubcs

签名：`tla.simd.vsubcs(src0, src1, borrow, mask=None)`

返回：`(diff, borrow_out)`。`diff` 为 tla.vector，`borrow_out` 为 int1 谓词。

编译表示：ave.hir.vsubcs。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src0` | 左操作数（被减数） | tla.vector | 数据类型：<br>B32：int32 / uint32 | 必须为满 64 lanes，与 `src1` 同 dtype。 | 必填 |
| `src1` | 右操作数（减数） | tla.vector | 数据类型：<br>B32：int32 / uint32 | 同上。 | 必填 |
| `borrow` | 输入借位谓词 | tla.vector（int1 谓词） | 数据类型：int1<br>lane 数：64 | 必填；lane 为真表示该 lane 减 1。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词）或 None | 数据类型：int1<br>lane 数：64 | None 生成全真 mask；inactive lane 的 diff 为 0、borrow_out 为假。 | None |

**语义**：`diff = src0 - src1 - borrow`（按 2³² 取模），`borrow_out` 为无符号借位。

**极性**：AVE 侧的 `carry_in` 是 no-borrow 语义（真表示不借位），与本接口的 borrow 相反；前端已按此取反，跨层直接复用 `ave.hir.vsubcs` 时不要照搬。

**调用示例**

输入为各含 64 个 int32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub, b_ub, out_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        b = tla.simd.vlds(b_ub[0])
        diff, borrow_out = tla.simd.vsubcs(a, b, tla.simd.pge(0, 64))
```

仓库测试：[tla_ut/simd/carry.py](https://github.com/triton-lang/triton-ascend/blob/aa927de3356e671160f2b3cf29fe23b6072f2b16/third_party/ascend/unittest/tla_ut/simd/carry.py#L38)。

<a name="api-tla-simd-vlrelu"></a>

## tla.simd.vlrelu

签名：`tla.simd.vlrelu(src, alpha, mask=None)`

返回：tla.vector。

编译表示：ave.hir.vlrelus。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：不支持<br>B16：float16<br>B32：float32<br>B64：不支持 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `alpha` | 负值分支的斜率 | int / float / 标量 tl.tensor | 数据类型：<br>B8：不支持<br>B16：float16<br>B32：float32<br>B64：不支持 | 斜率转换为 src 的 float16 或 float32。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |

**调用示例**

输入为各含 64 个 float32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        result = tla.simd.vlrelu(a, 2.0)
```

仓库测试：[tla_ut/simd/vrelu_ext.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/vrelu_ext.py#L39)。

<a name="api-tla-simd-vprelu"></a>

## tla.simd.vprelu

签名：`tla.simd.vprelu(src0, src1, mask=None)`

返回：tla.vector。

编译表示：ave.hir.vprelu。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src0` | 第一个源操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：float16<br>B32：float32<br>B64：不支持 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：float16<br>B32：float32<br>B64：不支持 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |

**调用示例**

输入为各含 64 个 float32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub, b_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        b = tla.simd.vlds(b_ub[0])
        result = tla.simd.vprelu(a, b)
```

仓库测试：[tla_ut/simd/vrelu_ext.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/vrelu_ext.py#L63)。

<a name="api-tla-simd-vcpadd"></a>

## tla.simd.vcpadd

签名：`tla.simd.vcpadd(src, mask=None, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector，lane 数与 `src` 相同。

编译表示：ave.hir.vcpadd。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | tla.vector | 数据类型：<br>B16：float16<br>B32：float32 | 必须为满寄存器：float16 取 128 lanes，float32 取 64 lanes。 | 必填 |
| `mask` | 控制各源 lane 是否参与相加 | tla.vector（int1 谓词）或 None | 数据类型：int1<br>lane 数：与 `src` 一致 | None 生成全真 mask；未选中的输入 lane 按 0 参与。 | None |
| `mode` | 未选中 lane 的输出约定 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | ZEROING 时 inactive lane 与结果上半部均为 0；MERGING 时 inactive lane 回填 `src`。 | MODE.ZEROING |

**语义**：结果 lane `j`（`j < N/2`）为 `src[2j] + src[2j+1]`，其余 lane 为 0。

**调用示例**

输入为各含 64 个 float32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub, out_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        sums = tla.simd.vcpadd(a)
        tla.simd.vsts(out_ub[0], sums, dist='NORM_B32')
```

仓库测试：[tla_ut/simd/vcpadd.py](https://github.com/triton-lang/triton-ascend/blob/aa927de3356e671160f2b3cf29fe23b6072f2b16/third_party/ascend/unittest/tla_ut/simd/vcpadd.py#L13)。

<a name="api-tla-simd-vsqz"></a>

## tla.simd.vsqz

签名：`tla.simd.vsqz(src, mask, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector，lane 数与 `src` 相同。

编译表示：ave.hir.vsqz（`squeeze_store` 固定为 `STORED`）。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 待压缩的源数据 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32 | 必须为满寄存器：8 位 256 lanes、16 位 128 lanes、32 位 64 lanes。 | 必填 |
| `mask` | 选择活跃 lane 的谓词 | tla.vector（int1 谓词） | 数据类型：int1<br>lane 数：与 `src` 相同 | 必填，不接受 None。 | 必填 |
| `mode` | 未选中 lane 的输出约定 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | ZEROING 时压缩后剩余的尾部 lane 为 0；MERGING 时 inactive lane 回填 `src`。 | MODE.ZEROING |

**语义**：活跃 lane 按原顺序压到低位 lane，尾部置零。

**调用示例**

输入为 64 个 int32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        result = tla.simd.vsqz(a, tla.simd.pge(32, 64))
```

仓库测试：[tla_ut/simd/vsqz.py](https://github.com/triton-lang/triton-ascend/blob/aa927de3356e671160f2b3cf29fe23b6072f2b16/third_party/ascend/unittest/tla_ut/simd/vsqz.py#L13)。

<a name="api-tla-simd-vusqz"></a>

## tla.simd.vusqz

签名：`tla.simd.vusqz(mask, dtype='u32')`

返回：tla.vector，lane 数由 `dtype` 决定。

编译表示：ave.hir.vusqz。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `mask` | 活跃 lane 谓词 | tla.vector（int1 谓词） | 数据类型：int1<br>lane 数：等于结果 lane 数 | 必填。 | 必填 |
| `dtype` | 结果元素类型 | 编译期 tla.dtype 或字符串 | int8 / uint8 / int16 / uint16 / int32 / uint32 | 决定结果位宽与 lane 数：8 位 256 lanes、16 位 128 lanes、32 位 64 lanes。 | 'u32'（uint32） |

**语义**：排他前缀计数。结果 lane `i` 为索引小于 `i` 的活跃 lane 个数。

**调用示例**

输出为 64 个 uint32 元素。

```python
@tla.jit
def example(out_ub):
    with tla.scope.simd():
        counts = tla.simd.vusqz(tla.simd.pge(32, 64))
        tla.simd.vsts(out_ub[0], counts, dist='NORM_B32')
```

仓库测试：[tla_ut/simd/vusqz.py](https://github.com/triton-lang/triton-ascend/blob/aa927de3356e671160f2b3cf29fe23b6072f2b16/third_party/ascend/unittest/tla_ut/simd/vusqz.py#L13)。

<a name="api-tla-simd-vunpack"></a>

## tla.simd.vunpack

签名：`tla.simd.vunpack(src, target_dtype, mask=None, part=tla.simd.PART.EVEN, pp='pp0', signed=False, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.pge / ave.hir.vextsi / ave.hir.vextui / ave.hir.vsel。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：不支持<br>B64：不支持 | 满寄存器 8→16、8→32 或 16→32，mask 对应输出 lanes。2 倍扩展用 part，4 倍扩展用 pp。 | 必填 |
| `target_dtype` | 目标元素类型 | 编译期 tl.dtype | 数据类型：<br>B8：不支持<br>B16：int16 / uint16<br>B32：int32 / uint32<br>B64：不支持 | 满寄存器 8→16、8→32 或 16→32，mask 对应输出 lanes。2 倍扩展用 part，4 倍扩展用 pp。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。按输出 lane 匹配。 | None |
| `part` | 选择偶数或奇数槽位 | 编译期 tla.simd.PART 常量或 str | part_even / part_odd | 按转换或 packing 的槽位规则解释。 | PART.EVEN |
| `pp` | 选择四路布局中的槽位 | 编译期 str | pp0 / pp1 / pp2 / pp3 | 用于 4 倍扩展。 | 'pp0' |
| `signed` | 是否按有符号值扩展 | 编译期 bool | False / True | True 符号扩展，False 零扩展。 | False |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

**调用示例**

输入为各含 128 个 int16 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        result = tla.simd.vunpack(a, tl.int32)
```

仓库测试：[tla_ut/simd/vunpack.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/vunpack.py#L49)。

<a name="api-tla-simd-vgatherb"></a>

## tla.simd.vgatherb

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`def vgatherb( base: tla.buffer, index: tla.vector, mask=None, ) -> tla.vector`

<a name="api-tla-simd-vabsdif"></a>

## tla.simd.vabsdif

签名：`tla.simd.vabsdif(lhs, rhs, mask=None, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.vabs_diff。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `lhs` | 左操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：float16<br>B32：float32<br>B64：不支持 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `rhs` | 右操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：float16<br>B32：float32<br>B64：不支持 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 或 str | ZEROING / MODE_ZEROING / MERGING / MODE_MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

**调用示例**

输入为各含 64 个 float32 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(a_ub, b_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        b = tla.simd.vlds(b_ub[0])
        result = tla.simd.vabsdif(a, b)
```

仓库测试：[tla_ut/simd/vabsdif.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/simd/vabsdif.py#L50)。

<a name="api-tla-simd-dhistv2"></a>

## tla.simd.dhistv2

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`def dhistv2( dst: tla.vector, src: tla.vector, mask=None, bin=0 ) -> tla.vector`

<a name="api-tla-simd-chistv2"></a>

## tla.simd.chistv2

签名：`tla.simd.chistv2(acc, src, mask, bin)`

返回：tla.vector。

编译表示：ave.hir.chistv2。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `acc` | 累加器 | tla.vector | 数据类型：uint16<br>lane 数：128 | acc 固定 128 lanes，src 与 mask 固定 256 lanes。计数累加保持在 uint16 范围内。 | 必填 |
| `src` | 源数据 | tla.vector | 数据类型：uint8<br>lane 数：256 | acc 固定 128 lanes，src 与 mask 固定 256 lanes。计数累加保持在 uint16 范围内。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） | 数据类型：int1<br>lane 数：256 | — | 必填 |
| `bin` | 选择直方图阈值区间 | 编译期 int | 0 / 1 | 0 对应阈值 0–127，1 对应 128–255。 | 必填 |

**语义**：累积直方图。结果 lane `i` 为 `acc[i]` 加上活跃源字节中数值 ≤ `i + 128 × bin` 的个数。

**调用示例**

源为 256 个 uint8 元素的连续 UB 缓冲区。

```python
@tla.jit
def example(src_ub):
    with tla.scope.simd():
        src = tla.simd.vlds(src_ub[0])
        acc = tla.simd.vdups(0, tl.uint16)
        counts = tla.simd.chistv2(acc, src, tla.simd.pset(8), 0)
```
