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
| `scope` | 指定执行域或地址空间 | 编译期 tla.simd.SCOPE | SCOPE.VV_ALL / SCOPE.VST_VLD / SCOPE.VLD_VST / SCOPE.VST_VST / SCOPE.VS_ALL / SCOPE.VST_LD / SCOPE.VLD_ST / SCOPE.VST_ST / SCOPE.SV_ALL / SCOPE.ST_VLD / SCOPE.LD_VST / SCOPE.ST_VST / SCOPE.SS_ALL / SCOPE.ST_LD / SCOPE.LD_ST / SCOPE.ST_ST | SIMD VF 内的标量或向量 load/store 屏障，按访问顺序选择。不在上表的取值直接报错。 | 必填 |

**调用示例**

两个 UB 缓冲区均为 64 个 float32 元素。

```python
@tla.jit
def example(src_ub, dst_ub):
    with tla.scope.simd():
        values = tla.simd.vlds(src_ub[0])
        tla.simd.vsts(dst_ub[0], values, dist='NORM_B32')
        tla.simd.membar(tla.simd.SCOPE.VST_VLD)
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
| `src0` | 第一个源操作数 | tla.vector（int1 谓词） | 数据类型：<br>B1：int1 | 各谓词 lane 数相同。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector（int1 谓词） | 数据类型：<br>B1：int1 | 各谓词 lane 数相同。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |

输入 dtype 或 lane 数不一致时抛 ValueError；同型但非 int1 的输入抛 TypeError。mask 必须为同 lane 数的 int1 谓词，不进行类型转换或广播。

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
| `src` | 源数据 | tla.vector（int1 谓词） | 数据类型：<br>B1：int1 | 各谓词 lane 数相同。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |

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
| `dist` | 指令的分布模式 | 编译期 str（使用 PAT 常量） | PAT.ALL / VL1 / VL2 / VL3 / VL4 / VL8 / VL16 / VL32 / VL64 / VL128 / M3 / M4 / H / Q / ALLF（均为 PAT 成员） | 硬件图样需与谓词宽度配合。 | PAT.ALL |

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
| `src0` | 第一个源操作数 | tla.vector（int1 谓词） | 数据类型：<br>B1：int1 | 各谓词 lane 数相同。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector（int1 谓词） | 数据类型：<br>B1：int1 | 各谓词 lane 数相同。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |

输入 dtype 或 lane 数不一致时抛 ValueError；同型但非 int1 的输入抛 TypeError。mask 必须为同 lane 数的 int1 谓词，不进行类型转换或广播。

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
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：int8<br>B16：int16 / float16<br>B32：int32 / float32<br>B64：int64 | 无符号整数输入被拒绝。lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE | MODE.ZEROING / MODE.MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

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
| `src0` | 第一个源操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 两操作数必须同 dtype、同 lane 数，不一致直接报错，不做隐式转换或广播；lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 B64 走 RegBase 库函数，使用 2×VL 表示。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 两操作数必须同 dtype、同 lane 数，不一致直接报错，不做隐式转换或广播；lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 B64 走 RegBase 库函数，使用 2×VL 表示。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE | MODE.ZEROING / MODE.MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

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
| `src0` | 第一个源操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：不支持<br>B32：int32 / uint32<br>B64：不支持 | 返回模 2³² 的和与进位谓词。比较按无符号解释。两操作数必须同 dtype、同 lane 数，不一致直接报错，不做隐式转换或广播；lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：不支持<br>B32：int32 / uint32<br>B64：不支持 | 返回模 2³² 的和与进位谓词。比较按无符号解释。两操作数必须同 dtype、同 lane 数，不一致直接报错，不做隐式转换或广播；lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |

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
| `scalar` | 广播给各 lane 的标量操作数 | int / float / 标量 tl.tensor | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 标量 tensor 与源 dtype 一致。整数常量使用源 dtype 的数值范围。前端也接收单 lane 向量，但 AVE scalar 操作数要求标量。 B64 大整数常量经由 double 构造，需保持精确值时传入同 dtype 的标量 tensor。整数必须落在 src 元素类型的取值范围内，类型不可转换或超出范围直接报错。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE | MODE.ZEROING / MODE.MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

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
| `src0` | 第一个源操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 参与计算的向量需同 dtype（含符号性）、同 lane 数；不一致抛 ValueError，不隐式转换或广播。 B64 走 RegBase 库函数，使用 2×VL 表示。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 参与计算的向量需同 dtype（含符号性）、同 lane 数；不一致抛 ValueError，不隐式转换或广播。 B64 走 RegBase 库函数，使用 2×VL 表示。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 枚举 | MODE.ZEROING / MODE.MERGING | MERGING 保留第一个向量操作数的 inactive lane，ZEROING 将其置零。 | MODE.ZEROING |

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
| `acc` | 累加器 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 参与计算的向量需同 dtype（含符号性）、同 lane 数；不一致抛 ValueError，不隐式转换或广播。 | 必填 |
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 参与计算的向量需同 dtype（含符号性）、同 lane 数；不一致抛 ValueError，不隐式转换或广播。 | 必填 |
| `scalar` | 广播给各 lane 的标量操作数 | int / float / 标量 tl.tensor | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：不支持 | tensor 须为秩 0 且与源 dtype（含符号性）一致；dtype 不同抛 TypeError，非秩 0 抛 ValueError。整数源只接受非 bool 的整数常量，越出源 dtype 范围抛 ValueError。单 lane vector 不满足 AVE 标量类型要求，不能作为有效输入；当前前端尚未提前拒绝。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 枚举 | MODE.ZEROING / MODE.MERGING | MERGING 保留第一个向量操作数的 inactive lane，ZEROING 将其置零。 | MODE.ZEROING |

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

签名：`tla.simd.vcadd(src, mask=None, mode=tla.simd.MODE.ZEROING)`

实现状态：AVE 开发分支已实现；提交和验证边界见[实现基线](../tracking/simd.md#simd-baseline)。

返回：与 src 同 dtype、同 lane 数的向量；只有 lane 0 是求和结果。

编译表示：ave.hir.reduction。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 待求和的数据 | tla.vector | 本节验证范围：<br>数据类型：<br>B32：int32 / float32<br>lane 数：64 | 前端未逐类型限制；其他类型组合不据此视为已验证。 | 必填 |
| `mask` | 选择参与计算的元素 | tla.vector（int1）或 None | 数据类型：<br>B1：int1<br>lane 数：64 | None 表示全部启用。 | None |
| `mode` | 结果模式 | 编译期 tla.simd.MODE 枚举 | MODE.ZEROING | MERGING 报错。 | MODE.ZEROING |

对 mask 为真的所有源元素求和，结果放在第 0 个 lane。其余 lane 未定义为有效结果，不能依赖其为零。返回类型不扩宽；浮点结果遵循目标归约顺序，不保证与逐元素顺序累加逐位一致。

专用用例覆盖 int32、float32 的全真、全假、前缀及混合 mask；尚未完成的类型矩阵见 [SIMD-014](../tracking/simd.md#simd-014)。

接口描述整向量求和，不绑定硬件指令名称。AVE 已有 Ascend960 的 `vcgaddv2` 分组累加再接 `vcaddv2` 汇总路径，仍应满足本节的输入 mask、结果 dtype 和 lane 0 契约；这是源码适配证据，尚非 A6 设备验收。A6 的具体 target/toolchain 对应关系及验证项见 [SIMD-014](../tracking/simd.md#simd-014)。

**调用示例**

x_ub 为连续 64 个 float32 元素；只使用 result 的第 0 个 lane。

```python
@tla.jit
def example(x_ub):
    with tla.scope.simd():
        src = tla.simd.vlds(x_ub[0])
        result = tla.simd.vcadd(src)
```

仓库测试：[memory_completion.py](https://github.com/triton-lang/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/memory_completion.py#L48)。

<a name="api-tla-simd-vcgadd"></a>

## tla.simd.vcgadd

签名：`tla.simd.vcgadd(src, mask=None, mode=tla.simd.MODE.ZEROING)`

实现状态：AVE 开发分支已实现；提交和验证边界见[实现基线](../tracking/simd.md#simd-baseline)。

返回：满寄存器向量，前 8 个 lane 为各组结果，其余 lane 为零。

编译表示：ave.hir.group_reduction。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 待分组归约的数据 | tla.vector | 数据类型：<br>B8：不支持<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：不支持<br>lane 数：16 位为 128，32 位为 64 | 必须是完整 256 字节寄存器；不支持 int8 / uint8 / bfloat16 / int64 / uint64。 | 必填 |
| `mask` | 选择参与计算的元素 | tla.vector（int1）或 None | 数据类型：<br>B1：int1<br>lane 数：与 src 相同，16 位输入为 128，32 位为 64 | None 表示全部启用。 | None |
| `mode` | 结果的掩码处理方式 | 编译期 tla.simd.MODE 枚举 | MODE.ZEROING / MODE.MERGING | 两种模式结果相同；没有旧目标值可保留，结果尾部始终为零。 | MODE.ZEROING |

仅支持 Ascend950。把源寄存器分成 8 个连续的 32 字节组，每组独立求和：16 位数据每组 16 个元素，32 位数据每组 8 个元素。mask 控制源元素是否参与，不改变分组或结果位置。

int16 / uint16 输入扩宽为 int32 / uint32，返回 64 lanes；int32 / uint32 加法按 32 位回绕。float16 / float32 保持输入类型，按硬件顺序求和；空组结果为零。

当前 lowering 仅放行 Ascend950，拒绝 Ascend960。新后端必须保持这里的八组边界、结果位置、扩宽和 mask 规则；不能因底层也有 group 指令就直接替换。若要暴露 A6 不同的原生分组语义，须另行设计显式参数或接口，见 [SIMD-015](../tracking/simd.md#simd-015)。

**调用示例**

Ascend950；x_ub 为连续 64 个 float32 元素，result[0:8] 分别对应原数据的八组。

```python
@tla.jit
def example(x_ub):
    with tla.scope.simd():
        src = tla.simd.vlds(x_ub[0])
        result = tla.simd.vcgadd(src)
```

仓库测试：[group_reduce.py](https://github.com/triton-lang/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/group_reduce.py#L33)。

<a name="api-tla-simd-vcgmax"></a>

## tla.simd.vcgmax

签名：`tla.simd.vcgmax(src, mask=None, mode=tla.simd.MODE.ZEROING)`

实现状态：AVE 开发分支已实现；提交和验证边界见[实现基线](../tracking/simd.md#simd-baseline)。

返回：满寄存器向量，前 8 个 lane 为各组结果，其余 lane 为零。

编译表示：ave.hir.group_reduction。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 待分组归约的数据 | tla.vector | 数据类型：<br>B8：不支持<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：不支持<br>lane 数：16 位为 128，32 位为 64 | 必须是完整 256 字节寄存器；不支持 int8 / uint8 / bfloat16 / int64 / uint64。 | 必填 |
| `mask` | 选择参与计算的元素 | tla.vector（int1）或 None | 数据类型：<br>B1：int1<br>lane 数：与 src 相同，16 位输入为 128，32 位为 64 | None 表示全部启用。 | None |
| `mode` | 结果的掩码处理方式 | 编译期 tla.simd.MODE 枚举 | MODE.ZEROING / MODE.MERGING | 两种模式结果相同；没有旧目标值可保留，结果尾部始终为零。 | MODE.ZEROING |

仅支持 Ascend950。把源寄存器分成 8 个连续的 32 字节组，每组独立取最大值：16 位数据每组 16 个元素，32 位数据每组 8 个元素。mask 控制源元素是否参与，不改变分组或结果位置。

结果保持输入 dtype 和 lane 数；整数比较按 dtype 的符号解释。空组使用硬件对应归约的初始值；浮点 NaN、正负零和无穷值遵循目标指令规则，不承诺与 Python 顺序比较一致。

当前 lowering 仅放行 Ascend950，拒绝 Ascend960。新后端必须保持这里的八组边界、结果位置、扩宽和 mask 规则；不能因底层也有 group 指令就直接替换。若要暴露 A6 不同的原生分组语义，须另行设计显式参数或接口，见 [SIMD-016](../tracking/simd.md#simd-016)。

**调用示例**

Ascend950；x_ub 为连续 64 个 float32 元素，result[0:8] 分别对应原数据的八组。

```python
@tla.jit
def example(x_ub):
    with tla.scope.simd():
        src = tla.simd.vlds(x_ub[0])
        result = tla.simd.vcgmax(src)
```

仓库测试：[group_reduce.py](https://github.com/triton-lang/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/group_reduce.py#L33)。

<a name="api-tla-simd-vcgmin"></a>

## tla.simd.vcgmin

签名：`tla.simd.vcgmin(src, mask=None, mode=tla.simd.MODE.ZEROING)`

实现状态：AVE 开发分支已实现；提交和验证边界见[实现基线](../tracking/simd.md#simd-baseline)。

返回：满寄存器向量，前 8 个 lane 为各组结果，其余 lane 为零。

编译表示：ave.hir.group_reduction。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 待分组归约的数据 | tla.vector | 数据类型：<br>B8：不支持<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：不支持<br>lane 数：16 位为 128，32 位为 64 | 必须是完整 256 字节寄存器；不支持 int8 / uint8 / bfloat16 / int64 / uint64。 | 必填 |
| `mask` | 选择参与计算的元素 | tla.vector（int1）或 None | 数据类型：<br>B1：int1<br>lane 数：与 src 相同，16 位输入为 128，32 位为 64 | None 表示全部启用。 | None |
| `mode` | 结果的掩码处理方式 | 编译期 tla.simd.MODE 枚举 | MODE.ZEROING / MODE.MERGING | 两种模式结果相同；没有旧目标值可保留，结果尾部始终为零。 | MODE.ZEROING |

仅支持 Ascend950。把源寄存器分成 8 个连续的 32 字节组，每组独立取最小值：16 位数据每组 16 个元素，32 位数据每组 8 个元素。mask 控制源元素是否参与，不改变分组或结果位置。

结果保持输入 dtype 和 lane 数；整数比较按 dtype 的符号解释。空组使用硬件对应归约的初始值；浮点 NaN、正负零和无穷值遵循目标指令规则，不承诺与 Python 顺序比较一致。

当前 lowering 仅放行 Ascend950，拒绝 Ascend960。新后端必须保持这里的八组边界、结果位置、扩宽和 mask 规则；不能因底层也有 group 指令就直接替换。若要暴露 A6 不同的原生分组语义，须另行设计显式参数或接口，见 [SIMD-017](../tracking/simd.md#simd-017)。

**调用示例**

Ascend950；x_ub 为连续 64 个 float32 元素，result[0:8] 分别对应原数据的八组。

```python
@tla.jit
def example(x_ub):
    with tla.scope.simd():
        src = tla.simd.vlds(x_ub[0])
        result = tla.simd.vcgmin(src)
```

仓库测试：[group_reduce.py](https://github.com/triton-lang/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/group_reduce.py#L33)。

<a name="api-tla-simd-vci"></a>

## tla.simd.vci

签名：`tla.simd.vci(base)`

返回：tla.vector。

编译表示：ave.hir.vci。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `base` | 序列的起始值 | 编译期 int / 标量 int32 tl.tensor | −2³¹ 至 2³¹−1 | 结果固定为 64 lane int32 递增序列。 | 必填 |

`base` 的编译期值不是 int 或不在 int32 范围时抛 ValueError；运行时值不是秩 0 int32 时抛 TypeError。若需无溢出的递增索引，调用方还需保证 `base ≤ 2147483584`，使最后一项 `base + 63` 不超出 int32；前端未额外检查这一上界。

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
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 规约结果放在 lane 0，其余 lane 不应当作规约结果使用。lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE | MODE.ZEROING | MERGING 会报错。 | MODE.ZEROING |

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
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 规约结果放在 lane 0，其余 lane 不应当作规约结果使用。lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE | MODE.ZEROING | MERGING 会报错。 | MODE.ZEROING |

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
| `src0` | 第一个源操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 两操作数必须同 dtype、同 lane 数，不一致直接报错，不做隐式转换或广播；lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 B64 走 RegBase 库函数，使用 2×VL 表示。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 两操作数必须同 dtype、同 lane 数，不一致直接报错，不做隐式转换或广播；lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 B64 走 RegBase 库函数，使用 2×VL 表示。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `op` | 比较关系 | 编译期 tla.simd.CMP | CMP.EQ / CMP.NE / CMP.GT / CMP.GE / CMP.LT / CMP.LE / CMP.UGT / CMP.UGE / CMP.ULT / CMP.ULE | 有符号和无符号的有序比较按输入 dtype 区分。浮点使用浮点比较式。 | CMP.EQ |

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
| `scalar` | 广播给各 lane 的标量操作数 | int / float / 标量 tl.tensor | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 标量 tensor 要求秩 0、与 src 同 dtype（含符号性）；常量按 src dtype 构造，整数越出源 dtype 范围抛 ValueError。不同整数 dtype 的当前实现例外见下文。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `op` | 比较关系 | 编译期 tla.simd.CMP 枚举 | CMP.EQ / NE / GT / GE / LT / LE / UGT / UGE / ULT / ULE（均为 CMP 成员） | 有符号和无符号的有序比较按输入 dtype 区分。浮点使用浮点比较式。 | CMP.LT |

运行时标量须与 src 同 dtype，否则应抛 TypeError。当前不同整数 dtype 仍会自动转换，待修复，见 [SIMD-022](../tracking/simd.md#simd-022)；非标量 tensor 及 vector 抛 TypeError。

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

返回：tla.vector；8 / 16 / 32 位路径保持寄存器总位数，输出 lane 数为 `src.lanes × src位宽 ÷ dst位宽`。64 位路径暂不作此承诺，见下表。

编译表示：ave.hir.vtruncf / ave.hir.vextf / ave.hir.vsitofp / ave.hir.vuitofp / ave.hir.vfptosi / ave.hir.vfptoui。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | tla.vector | 源→目标类型对见下表；使用对应 dtype 的满寄存器 | 不支持的组合不得作为有效调用；前端尚未统一提前拒绝。 | 必填 |
| `target_dtype` | 目标元素类型 | 编译期 tl.dtype | 仅限下表列出的目标 dtype | 不接受字符串 dtype；int→int 不由本接口处理。 | 必填 |
| `mask` | 选择参与转换的源元素 | tla.vector（int1）或 None | 数据类型：<br>B1：int1<br>lane 数与 src 相同 | None 为全真；非 int1 或 lane 数不匹配抛 ValueError。 | None |
| `round` | 舍入方式 | 编译期 str（使用 ROUND 常量） | ROUND.R / A / F / C / Z / O | R：最近偶数；A：最近、半值远离零；F：向下；C：向上；Z：向零；O：round-to-odd。生效条件见类型对表；并非所有目标都支持六种模式。 | ROUND.R |
| `sat` | 请求转换饱和 | 编译期 bool | False / True | 仅在下表标注生效的路径上传入；不能据此承诺所有转换都钳位。 | True |
| `part` | 选择偶数或奇数槽位 | 编译期 str（使用 PART 常量） | PART.EVEN / PART.ODD | 只在宽度变化且该路径使用 part 时生效；不是连续 LOWER / HIGHER 半区。 | PART.EVEN |
| `mode` | 结果模式 | 编译期 tla.simd.MODE 枚举 | MODE.ZEROING | MODE.MERGING 抛 ValueError。 | MODE.ZEROING |

**源→目标类型对**

以下「已接通」表示已核对 TLA builder、AVE 类型约束及目标转换分支，**不等于该类型对的全部参数组合已经通过设备测试**。依据为 main-dev `0227cde` 配套 AscendNPU-IR `bd228fa`；其他后端需独立验收。「不使用」表示当前 builder 不向该指令传递参数，调用时保持默认值。

| src dtype | dst dtype | 当前边界 | 生效参数 |
| --- | --- | --- | --- |
| float16 / bfloat16 | float32 | 已接通 | part；round / sat 不使用 |
| float32 | float16 / bfloat16 | 已接通 | round、sat、part |
| int8 / uint8 | float16 | 已接通；无符号路径的 lane 映射待设备验收 | part；round / sat 不使用 |
| int16 | float16 | 已接通 | round；part / sat 不使用 |
| int16 | float32 | 已接通 | part；round / sat 不使用 |
| int32 | float32 | 已接通 | round；part / sat 不使用 |
| float16 | int8 / uint8 | 已接通；无符号路径的 lane 映射待设备验收 | round、sat、part |
| float16 | int16 | 已接通 | round、sat；part 不使用 |
| float16 | int32 | 已接通 | round、part；sat 不使用 |
| bfloat16 | int32 | 已接通 | round、sat、part |
| float32 | int16 | 已接通 | round、sat、part |
| float32 | int32 | 已接通 | round、sat；part 不使用 |
| float8e4m3fn / float8e5m2 | float32 | 有转换分支，暂不列设备正例 | 8↔32 的四路槽位与当前 EVEN / ODD 接口需确认 |
| float32 | float8e4m3fn / float8e5m2 | 有转换分支，暂不列设备正例 | round、sat、part 的目标约束及四路槽位需确认 |
| int64 / uint64 | float32 | AVE 有库路径，暂不列设备正例 | TLA 的 lane 计算与库的 2×VL 表示需对齐；mask、part、舍入需回归 |
| float32 | int64 / uint64 | AVE 有库路径，暂不列设备正例 | 同上；不能由存在库函数推定 sat / part / mask 和精度全部正确 |
| 任意整数 | 任意整数 | 不支持；TLA 抛 ValueError | 包括同型和仅改变符号性；bitcast 仅重解释位，不是数值转换 |
| 同型浮点；float16 ↔ bfloat16；FP8 ↔ float16 / bfloat16；两种 FP8 互转 | 对应目标 | 不支持此直接调用 | 部分会通过前端，但无匹配的当前 lowering；不能用 bitcast 代替数值转换 |
| 其余未列出的组合 | — | 当前规格不支持 | 例如 uint16 / uint32→浮点、float32→int8 / uint8 / uint16 / uint32、float16→uint16 / uint32 / int64 / uint64，以及 int1 / float64 相关组合 |

FP8 的公开 dtype 写作 `tl.float8e4nv` / `tl.float8e5`。参数名是 `round`；未知 round / part 值抛 ValueError。`sat` 规范为 bool，但当前前端使用 bool 转换，尚未严格拒绝其他类型。表中列为不支持或待确认的路径不用于设备正例；报错位置及未补齐的前置检查见 [SIMD-023](../tracking/simd.md#simd-023)。

**数值与 lane 规则**

浮点→整数先舍入；支持饱和的类型对在 `sat=True` 时将有限越界结果钳到目标范围（N 位有符号整数为 −2^(N−1)～2^(N−1)−1，无符号为 0～2^N−1）。`sat=False` 不承诺模回绕；浮点特殊值和非默认舍入须按目标确认。整数→浮点及浮点缩窄可能丢失精度。

float16→float32：输出 `i` 读取源 `2i`（EVEN）或 `2i+1`（ODD），由该源位置的 mask 控制。float32→float16：源 `i` 写入输出 `2i` 或 `2i+1`，由源 mask 的第 `i` 位控制，未选位置为零。该槽位规则有 950PR 专用用例，不能外推为所有类型对的实测结论。

**调用示例**

a_ub 为连续 64 个 int32 元素；本例使用默认 ROUND.R。

```python
@tla.jit
def example(a_ub):
    with tla.scope.simd():
        a = tla.simd.vlds(a_ub[0])
        result = tla.simd.vcvt(a, tl.float32, round=tla.simd.ROUND.R)
```

仓库测试：[vcvt.py](https://github.com/triton-lang/triton-ascend/blob/0227cdebc7940f38cbe67007110a44fe2434ec94/third_party/ascend/unittest/tla_ut/simd/vcvt.py#L29) 覆盖 int32↔float32、R / Z、饱和及 mask；[vcvt_mask.py](https://github.com/triton-lang/triton-ascend/blob/0227cdebc7940f38cbe67007110a44fe2434ec94/third_party/ascend/unittest/tla_ut/simd/vcvt_mask.py#L38) 覆盖 float16↔float32 的 EVEN / ODD 与源 mask。本次只审阅用例，未重跑编译或设备测试。

<a name="api-tla-simd-vdintlv"></a>

## tla.simd.vdintlv

签名：`tla.simd.vdintlv(src1, src2)`

返回：两个解交错后的向量。

编译表示：ave.hir.vdintlv。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src1` | 第二个源操作数 | tla.vector | 数据类型：<br>B8：float8e4m3fn / float8e5m2 / int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 两个输入同 dtype（含符号性）、同 lane 数；不一致抛 ValueError，不隐式转换。构造 SPARSE 去交错，返回两个同型值。 | 必填 |
| `src2` | 第二个输入向量 | tla.vector | 数据类型：<br>B8：float8e4m3fn / float8e5m2 / int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 两个输入同 dtype（含符号性）、同 lane 数；不一致抛 ValueError，不隐式转换。构造 SPARSE 去交错，返回两个同型值。 | 必填 |

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
| `src0` | 第一个源操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 两操作数必须同 dtype、同 lane 数，不一致直接报错，不做隐式转换或广播；lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 两操作数必须同 dtype、同 lane 数，不一致直接报错，不做隐式转换或广播；lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE | MODE.ZEROING / MODE.MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |
| `precision` | 精度和 FTZ 方式 | 编译期 str（闭集） | None / ftz_true / exact / vdiv_0ulp_ftz_true | exact 路径仅支持 float32。vexp 的 ftz_false 路径会拒绝。只接受上表取值，其它精度档直接报错。 | None |

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
| `scalar` | 广播给各 lane 的标量操作数 | int / float / 标量 tl.tensor | 数据类型：<br>B8：float8e4m3fn / int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 标量常量按向量 dtype 构造，运行时标量需符合当前转换路径。必须为 Python 数值或 rank-0 tl.tensor；按 dtype 构造，整数必须落在该 dtype 的取值范围内，类型不可转换或超出范围直接报错。 | 必填 |
| `dtype` | 元素类型 | 编译期 tl.dtype | 数据类型：<br>B8：float8e4m3fn / int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 用对应元素类型将标量扩成满寄存器。其它 dtype 直接报错。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE | MODE.ZEROING / MODE.MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

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
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：float8e4m3fn / int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `pos` | 选择广播的源 lane | 编译期 tla.simd.POS | POS.LOWEST / POS.HIGHEST | 选择原向量的第一个或最后一个 lane。 | POS.LOWEST |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE | MODE.ZEROING / MODE.MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

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
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：不支持<br>B16：float16<br>B32：float32<br>B64：不支持 | lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE | MODE.ZEROING / MODE.MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |
| `precision` | 精度和 FTZ 方式 | 编译期 str（闭集） | None / ftz_true | exact 路径仅支持 float32。vexp 的 ftz_false 路径会拒绝。只接受上表取值，其它精度档直接报错。 | None |

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
| `lhs` | 左操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：不支持<br>B32：float32<br>B64：不支持 | 发出融合 exp(lhs−rhs)，不能假定分步实现有相同舍入。两操作数必须同 dtype、同 lane 数，不一致直接报错，不做隐式转换或广播；lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 | 必填 |
| `rhs` | 右操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：不支持<br>B32：float32<br>B64：不支持 | 发出融合 exp(lhs−rhs)，不能假定分步实现有相同舍入。两操作数必须同 dtype、同 lane 数，不一致直接报错，不做隐式转换或广播；lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE | MODE.ZEROING / MODE.MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

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
| `index` | 按元素计的索引向量 | tla.vector | 数据类型：<br>B16：uint16<br>B32：uint32<br>lane 数：16 位为 128，32 位为 64 | 索引位宽须与读取的数据位宽一致。单位为元素，活跃索引不得越界。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 枚举 | MODE.ZEROING / MODE.MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

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
| `src1` | 第二个源操作数 | tla.vector | 数据类型：<br>B8：float8e4m3fn / float8e5m2 / int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 两个输入同 dtype（含符号性）、同 lane 数；不一致抛 ValueError，不隐式转换。构造 SPARSE 交错，返回两个同型值。 | 必填 |
| `src2` | 第二个输入向量 | tla.vector | 数据类型：<br>B8：float8e4m3fn / float8e5m2 / int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 两个输入同 dtype（含符号性）、同 lane 数；不一致抛 ValueError，不隐式转换。构造 SPARSE 交错，返回两个同型值。 | 必填 |

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
| `index` | 按元素计的索引向量 | tla.vector | 数据类型：<br>B16：uint16<br>B32：uint32<br>lane 数：16 位为 128，32 位为 64 | 索引位宽须与读取的数据位宽一致。单位为元素，活跃索引不得越界。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 枚举 | MODE.ZEROING / MODE.MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

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

签名：`tla.simd.vld2(src, dist='DINTLV_B16', off=None)`

实现状态：AVE 开发分支已实现；提交和验证边界见[实现基线](../tracking/simd.md#simd-baseline)。

返回：两个与源缓冲区同 dtype 的满寄存器向量 (even, odd)。

编译表示：两次 ave.hir.vload + ave.hir.vdintlv。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 连续读取的起始地址 | tla.buffer / buffer_access | 地址空间：UB<br>数据类型：<br>B8：不支持<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 一维连续；buf 等价于 buf[0]；起始地址后必须能完整读取两个寄存器。 | 必填 |
| `dist` | 加载后去交错的元素宽度 | 编译期 str | DINTLV_B16 / DINTLV_B32 | 必须与 src 元素位宽一致；32 位数据需显式传 DINTLV_B32。 | 'DINTLV_B16' |
| `off` | 在 src 基础上再增加的偏移 | 整数常量 / 整数标量 / None | 单位为元素 | 最终范围由 src 下标与 off 共同决定；动态偏移也必须满足边界。 | None（不增加偏移） |

设每个寄存器有 N 个元素，从最终起点连续读 2N 个元素：even[i] 取第 2i 个，odd[i] 取第 2i+1 个。16 位数据 N=128，32 位数据 N=64。无 mask，也不更新地址。

专用用例包含 int16、float16、int32、float32；uint16、uint32、bfloat16 属于入口接受的组合，尚不能据此声明相同的真机覆盖。

**调用示例**

x_ub 为连续 128 个 float32 元素，满足完整读取范围。

```python
@tla.jit
def example(x_ub):
    with tla.scope.simd():
        even, odd = tla.simd.vld2(x_ub[0], dist='DINTLV_B32')
```

仓库测试：[memory_completion.py](https://github.com/triton-lang/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/memory_completion.py#L51)。

<a name="api-tla-simd-vln"></a>

## tla.simd.vln

签名：`tla.simd.vln(src, mask=None, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.vln / ave.hir.vsel。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：不支持<br>B16：float16<br>B32：float32<br>B64：不支持 | 参与计算的向量需同 dtype、同 lane 数。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 枚举 | MODE.ZEROING / MODE.MERGING | MERGING 保留第一个向量操作数的 inactive lane，ZEROING 将其置零。 | MODE.ZEROING |

有限实数正例使用 `0 < x ≤ 65504`（float16）或 `0 < x ≤ (2−2⁻²³)×2¹²⁷`（float32）的可表示值。零、负数、Inf、NaN 和非规格化数需按目标单独验收；当前前端只检查 dtype，不逐 lane 检查数值。

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
| `acc` | 累加器 | tla.vector | 数据类型：<br>B8：不支持<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 参与计算的向量需同 dtype（含符号性）、同 lane 数；不一致抛 ValueError，不隐式转换或广播。 | 必填 |
| `src0` | 第一个源操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 参与计算的向量需同 dtype（含符号性）、同 lane 数；不一致抛 ValueError，不隐式转换或广播。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 参与计算的向量需同 dtype（含符号性）、同 lane 数；不一致抛 ValueError，不隐式转换或广播。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 枚举 | MODE.ZEROING / MODE.MERGING | MERGING 保留第一个向量操作数的 inactive lane，ZEROING 将其置零。 | MODE.ZEROING |

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
| `src0` | 第一个源操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 两操作数必须同 dtype、同 lane 数，不一致直接报错，不做隐式转换或广播；lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 两操作数必须同 dtype、同 lane 数，不一致直接报错，不做隐式转换或广播；lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE | MODE.ZEROING / MODE.MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

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
| `scalar` | 广播给各 lane 的标量操作数 | int / float / 标量 tl.tensor | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 标量常量按向量 dtype 构造，运行时标量需符合当前转换路径。必须为 Python 数值或 rank-0 tl.tensor；按 src 的元素类型构造，整数必须落在该 dtype 的取值范围内（如 int8 −128–127、uint8 0–255），类型不可转换或超出范围直接报错。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE | MODE.ZEROING / MODE.MERGING | MERGING 保留第一个向量操作数的 inactive lane，ZEROING 将其置零。 | MODE.ZEROING |

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
| `src0` | 第一个源操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 两操作数必须同 dtype、同 lane 数，不一致直接报错，不做隐式转换或广播；lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 两操作数必须同 dtype、同 lane 数，不一致直接报错，不做隐式转换或广播；lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE | MODE.ZEROING / MODE.MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

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
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：不支持 | lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 | 必填 |
| `scalar` | 广播给各 lane 的标量操作数 | int / float / 标量 tl.tensor | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 必须为 Python 数值或 rank-0 tl.tensor；按 src 的元素类型构造：整数必须落在该 dtype 的取值范围内（如 int8 −128–127、uint8 0–255）；类型不可转换或超出范围直接报错。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE | MODE.ZEROING / MODE.MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

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
| `lhs` | 左操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 | 两操作数必须同 dtype、同 lane 数，不一致直接报错，不做隐式转换或广播；lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 B64 走 RegBase 库函数，使用 2×VL 表示，仅支持 int64。 | 必填 |
| `rhs` | 右操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 | 两操作数必须同 dtype、同 lane 数，不一致直接报错，不做隐式转换或广播；lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 B64 走 RegBase 库函数，使用 2×VL 表示，仅支持 int64。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE | MODE.ZEROING / MODE.MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

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
| `acc` | 累加器 | tla.vector | 数据类型：<br>B8：不支持<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 参与计算的向量需同 dtype（含符号性）、同 lane 数；不一致抛 ValueError，不隐式转换或广播。 | 必填 |
| `src0` | 第一个源操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 参与计算的向量需同 dtype（含符号性）、同 lane 数；不一致抛 ValueError，不隐式转换或广播。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：不支持 | 参与计算的向量需同 dtype（含符号性）、同 lane 数；不一致抛 ValueError，不隐式转换或广播。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 枚举 | MODE.ZEROING / MODE.MERGING | MERGING 保留第一个向量操作数的 inactive lane，ZEROING 将其置零。 | MODE.ZEROING |

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
| `src0` | 第一个源操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：不支持<br>B32：int32 / uint32<br>B64：不支持 | 两个输入同 dtype（含符号性）、同 lane 数；不一致抛 ValueError。返回同型低半部和高半部。前端接受 uint32，但当前高半部有已知错误，见下方限制。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：不支持<br>B32：int32 / uint32<br>B64：不支持 | 两个输入同 dtype（含符号性）、同 lane 数；不一致抛 ValueError。返回同型低半部和高半部。前端接受 uint32，但当前高半部有已知错误，见下方限制。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |

当前限制：问题单报告 950PR 上 `uint32(0xFFFFFFFF) × uint32(2)` 的高半部得到 `0xFFFFFFFF`，正确值应为 `0x00000001`；低半部均为 `0xFFFFFFFE`。当前 builder 未传入无符号 cast。`uint32` 暂不纳入设备正向验收，待修复并回归；`int32` 用例可独立保留。证据与后续验证见 [SIMD-042](../tracking/simd.md#simd-042)。

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
| `scalar` | 广播给各 lane 的标量操作数 | int / float / 标量 tl.tensor | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | tensor 须为秩 0 且与源 dtype（含符号性）一致；dtype 不同抛 TypeError，非秩 0 抛 ValueError。整数源只接受非 bool 的整数常量，越出源 dtype 范围抛 ValueError。单 lane vector 不满足 AVE 标量类型要求，不能作为有效输入；当前前端尚未提前拒绝。 B64 整数常量经 double 构造，2**53+1 会先变为 2**53；精确值须通过同 dtype 的秩 0 tensor 传入。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 枚举 | MODE.ZEROING / MODE.MERGING | MERGING 保留第一个向量操作数的 inactive lane，ZEROING 将其置零。 | MODE.ZEROING |

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
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16<br>B32：int32 / uint32 / float32<br>B64：int64 | lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE | MODE.ZEROING / MODE.MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

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
| `lhs` | 左操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 | 两操作数必须同 dtype、同 lane 数，不一致直接报错，不做隐式转换或广播；lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 B64 走 RegBase 库函数，使用 2×VL 表示，仅支持 int64。 | 必填 |
| `rhs` | 右操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 | 两操作数必须同 dtype、同 lane 数，不一致直接报错，不做隐式转换或广播；lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 B64 走 RegBase 库函数，使用 2×VL 表示，仅支持 int64。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE | MODE.ZEROING / MODE.MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

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
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：不支持<br>B16：int16 / uint16<br>B32：int32 / uint32<br>B64：不支持 | int16/uint16 → int8/uint8，int32/uint32 → int16/uint16。结果 2×输入 lanes，非选中槽位未指定。float16 / bfloat16 / fp8 / 64 位整数不支持。 | 必填 |
| `part` | 选择偶数或奇数槽位 | 编译期 tla.simd.PART | PART.EVEN / PART.ODD | 按转换或 packing 的槽位规则解释；PART.LOWER / PART.HIGHER 不属于本接口，传入直接报错。 | 'part_even' |

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
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：不支持<br>B16：float16<br>B32：int32 / float32<br>B64：不支持 | lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE | MODE.ZEROING / MODE.MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

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
| `indices` | 索引向量 | tla.vector | 数据类型：<br>B32：uint32<br>lane 数：64 | 按元素计，活跃索引不得重复或越界。 | 必填 |
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：不支持<br>B16：不支持<br>B32：uint32<br>B64：不支持 | dst 必须为一维连续、零 offset 的 UB buffer。src、indices 固定 64 lanes。活跃索引不得重复或越界。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） | 数据类型：<br>B1：int1<br>lane 数：64 | — | 必填 |

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
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | — | 必填 |
| `true_val` | 参数 | tla.vector | 数据类型：<br>B1：int1<br>B8：float8e4m3fn / float8e5m2 / int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 参与计算的向量需同 dtype（含符号性）、同 lane 数；不一致抛 ValueError，不隐式转换或广播。 B64 走 RegBase 库函数，使用 2×VL 表示。 | 必填 |
| `false_val` | 参数 | tla.vector | 数据类型：<br>B1：int1<br>B8：float8e4m3fn / float8e5m2 / int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 / uint64 | 参与计算的向量需同 dtype（含符号性）、同 lane 数；不一致抛 ValueError，不隐式转换或广播。 B64 走 RegBase 库函数，使用 2×VL 表示。 | 必填 |

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
| `indices` | 索引向量 | tla.vector | 数据类型：<br>B16：uint16<br>lane 数：128<br>正常索引：0 至 127 | 按源向量 lane 编号寻址。 | 必填 |

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
| `src0` | 第一个源操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32<br>B64：不支持 | 两操作数必须同 dtype、同 lane 数，不一致直接报错，不做隐式转换或广播；lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32<br>B64：不支持 | 两操作数必须同 dtype、同 lane 数，不一致直接报错，不做隐式转换或广播；lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE | MODE.ZEROING / MODE.MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

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
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32<br>B64：不支持 | lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 | 必填 |
| `scalar` | 广播给各 lane 的标量操作数 | int / 标量整数 tl.tensor | 编译期值：−32768 至 32767 | 必须为 Python 数值或 rank-0 tl.tensor；最终物化为 src 的元素类型，非整数标量或超出 int16 范围直接报错。按元素位宽测试 0 至 B−1 的正常位移。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE | MODE.ZEROING / MODE.MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

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
| `src0` | 第一个源操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32<br>B64：不支持 | 两操作数必须同 dtype、同 lane 数，不一致直接报错，不做隐式转换或广播；lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32<br>B64：不支持 | 两操作数必须同 dtype、同 lane 数，不一致直接报错，不做隐式转换或广播；lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE | MODE.ZEROING / MODE.MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

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
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：不支持<br>B16：float16<br>B32：float32<br>B64：不支持 | lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE | MODE.ZEROING / MODE.MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |
| `precision` | 精度和 FTZ 方式 | 编译期 str（闭集） | None / ftz_true | 使用目标 SFU；只接受上表取值，其它精度档直接报错。 | None |

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
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
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
| `dist` | 指令的分布模式 | 编译期 str 或 None | None（NORM_B32） / NORM_B8 / NORM_B16 / NORM_B32 / NORM_B64 / ONEPT_B8 / ONEPT_B16 / ONEPT_B32 / ONEPT_B64 | 分布后缀须与指令所需位宽对应。ONEPT 只存一个元素。packing 和通道合并需要单独验证源、目的与目标组合。 | None |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |

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
| `lhs` | 左操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 | 两操作数必须同 dtype、同 lane 数，不一致直接报错，不做隐式转换或广播；lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 B64 走 RegBase 库函数，使用 2×VL 表示，仅支持 int64。 | 必填 |
| `rhs` | 右操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 | 两操作数必须同 dtype、同 lane 数，不一致直接报错，不做隐式转换或广播；lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 B64 走 RegBase 库函数，使用 2×VL 表示，仅支持 int64。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE | MODE.ZEROING / MODE.MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

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
| `lhs` | 左操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 | 两操作数必须同 dtype、同 lane 数，不一致直接报错，不做隐式转换或广播；lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 B64 走 RegBase 库函数，使用 2×VL 表示，仅支持 int64。 | 必填 |
| `rhs` | 右操作数 | tla.vector | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16 / float16 / bfloat16<br>B32：int32 / uint32 / float32<br>B64：int64 | 两操作数必须同 dtype、同 lane 数，不一致直接报错，不做隐式转换或广播；lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 B64 走 RegBase 库函数，使用 2×VL 表示，仅支持 int64。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE | MODE.ZEROING / MODE.MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

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
| `src0` | 第一个源操作数 | tla.vector（int1 谓词） | 数据类型：<br>B1：int1 | 各谓词 lane 数相同。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector（int1 谓词） | 数据类型：<br>B1：int1 | 各谓词 lane 数相同。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |

输入 dtype 或 lane 数不一致时抛 ValueError；同型但非 int1 的输入抛 TypeError。mask 必须为同 lane 数的 int1 谓词，不进行类型转换或广播。

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
| `src0` | 第一个源操作数 | tla.vector（int1 谓词） | 数据类型：<br>B1：int1 | 各谓词 lane 数相同。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector（int1 谓词） | 数据类型：<br>B1：int1 | 各谓词 lane 数相同。 | 必填 |
| `mask` | 为真选 src0，为假选 src1 | tla.vector（int1 谓词） | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | — | 必填 |

输入 dtype 或 lane 数不一致时抛 ValueError；同型但非 int1 的输入抛 TypeError。mask 必须为同 lane 数的 int1 谓词，不进行类型转换或广播。

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

签名：`tla.simd.ppack(src, part=tla.simd.PART.LOWER)`

实现状态：AVE 开发分支已实现；提交和验证边界见[实现基线](../tracking/simd.md#simd-baseline)。

返回：int1 谓词向量；64→128 或 128→256 lanes。

编译表示：ave.hir.preg.pack。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源逻辑谓词 | tla.vector（int1） | 数据类型：<br>B1：int1<br>lane 数：64 / 128 | lane 表示逻辑布尔元素；物理寄存器 padding 不参与接口语义。 | 必填 |
| `part` | 选择连续的下半区或上半区 | 编译期 str（使用 PART 常量） | PART.LOWER / PART.HIGHER | 不接受 EVEN / ODD。 | PART.LOWER |

设源有 N 个元素，输出有 2N 个。LOWER 将 src 放在前 N 个位置，HIGHER 放在后 N 个位置，另一半全部为 false。例如 128 个 true 做 LOWER pack 后得到前 128 个 true、后 128 个 false。

此接口改变逻辑谓词长度；与只调整物理布局的 preg.cast 不同。无需额外 mask、mode 或 width 参数。

**调用示例**

src 为 64 lanes，result 为 128 lanes。

```python
@tla.jit
def example():
    with tla.scope.simd():
        src = tla.simd.pset(32)
        result = tla.simd.ppack(src, part=tla.simd.PART.LOWER)
```

仓库测试：[predicate_ops.py](https://github.com/triton-lang/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/predicate_ops.py#L33)、[predicate_ops.py](https://github.com/triton-lang/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/predicate_ops.py#L138)。

<a name="api-tla-simd-punpack"></a>

## tla.simd.punpack

签名：`tla.simd.punpack(src, part=tla.simd.PART.LOWER)`

实现状态：AVE 开发分支已实现；提交和验证边界见[实现基线](../tracking/simd.md#simd-baseline)。

返回：int1 谓词向量；128→64 或 256→128 lanes。

编译表示：ave.hir.preg.unpack。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src` | 源逻辑谓词 | tla.vector（int1） | 数据类型：<br>B1：int1<br>lane 数：128 / 256 | lane 表示逻辑布尔元素；物理寄存器 padding 不参与接口语义。 | 必填 |
| `part` | 选择连续的下半区或上半区 | 编译期 str（使用 PART 常量） | PART.LOWER / PART.HIGHER | 不接受 EVEN / ODD。 | PART.LOWER |

从 N 个源元素中取连续 N/2 个：LOWER 取 src[0:N/2]，HIGHER 取 src[N/2:N]；保留选中元素的布尔值和顺序。

此接口改变逻辑谓词长度；与只调整物理布局的 preg.cast 不同。无需额外 mask、mode 或 width 参数。

**调用示例**

src 为 128 lanes，result 为 64 lanes。

```python
@tla.jit
def example():
    with tla.scope.simd():
        src = tla.simd.pset(16)
        result = tla.simd.punpack(src, part=tla.simd.PART.LOWER)
```

仓库测试：[predicate_ops.py](https://github.com/triton-lang/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/predicate_ops.py#L33)、[predicate_ops.py](https://github.com/triton-lang/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/predicate_ops.py#L176)。

<a name="api-tla-simd-pintlv"></a>

## tla.simd.pintlv

签名：`tla.simd.pintlv(src0, src1)`

实现状态：AVE 开发分支已实现；提交和验证边界见[实现基线](../tracking/simd.md#simd-baseline)。

返回：两个 int1 谓词向量组成的 tuple，二者均与输入同 lane 数。

编译表示：ave.hir.preg.interleave。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src0` | 第一个源谓词 | tla.vector（int1） | 数据类型：<br>B1：int1<br>lane 数：64 / 128 / 256 | 两个源谓词的 lane 数必须相同。 | 必填 |
| `src1` | 第二个源谓词 | tla.vector（int1） | 数据类型：<br>B1：int1<br>lane 数：64 / 128 / 256 | 两个源谓词的 lane 数必须相同。 | 必填 |

按 src0[0]、src1[0]、src0[1]、src1[1]……交错成 2N 个逻辑布尔值；前 N 个放入第一个结果，后 N 个放入第二个结果。

同一谓词密度下，pdintlv(*pintlv(a, b)) 恢复 a、b。宽度由输入 lane 数确定，不接收 width、mask 或 mode 参数；两个输出都可独立使用。

两个输入必须都是 int1 且 lane 数相同；不一致或 lane 数不在 64 / 128 / 256 中时抛 ValueError，不做隐式转换或广播。

**调用示例**

a、b 都是 64 lanes 的逻辑谓词。

```python
@tla.jit
def example():
    with tla.scope.simd():
        a = tla.simd.pge(32, 64)
        b = tla.simd.pge(16, 64)
        first, second = tla.simd.pintlv(a, b)
```

仓库测试：[predicate_ops.py](https://github.com/triton-lang/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/predicate_ops.py#L33)、[predicate_ops.py](https://github.com/triton-lang/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/predicate_ops.py#L176)。

<a name="api-tla-simd-pdintlv"></a>

## tla.simd.pdintlv

签名：`tla.simd.pdintlv(src0, src1)`

实现状态：AVE 开发分支已实现；提交和验证边界见[实现基线](../tracking/simd.md#simd-baseline)。

返回：两个 int1 谓词向量组成的 tuple，二者均与输入同 lane 数。

编译表示：ave.hir.preg.deinterleave。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `src0` | 第一个源谓词 | tla.vector（int1） | 数据类型：<br>B1：int1<br>lane 数：64 / 128 / 256 | 两个源谓词的 lane 数必须相同。 | 必填 |
| `src1` | 第二个源谓词 | tla.vector（int1） | 数据类型：<br>B1：int1<br>lane 数：64 / 128 / 256 | 两个源谓词的 lane 数必须相同。 | 必填 |

先把 src0、src1 按顺序拼成 2N 个逻辑布尔值；第一个结果取偶数位置，第二个取奇数位置。

同一谓词密度下，pdintlv(*pintlv(a, b)) 恢复 a、b。宽度由输入 lane 数确定，不接收 width、mask 或 mode 参数；两个输出都可独立使用。

两个输入必须都是 int1 且 lane 数相同；不一致或 lane 数不在 64 / 128 / 256 中时抛 ValueError，不做隐式转换或广播。

**调用示例**

a、b 都是 64 lanes 的逻辑谓词。

```python
@tla.jit
def example():
    with tla.scope.simd():
        a = tla.simd.pge(32, 64)
        b = tla.simd.pge(16, 64)
        first, second = tla.simd.pdintlv(a, b)
```

仓库测试：[predicate_ops.py](https://github.com/triton-lang/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/predicate_ops.py#L33)、[predicate_ops.py](https://github.com/triton-lang/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/predicate_ops.py#L176)。

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
| `offsets_or_val` | 省略 val 时是待写入值，提供 val 时是显式偏移 | 整数序列 / tla.vector（int1 谓词） | 偏移按元素计<br>数据类型：<br>B1：int1<br>lane 数：256 / 128 / 64 | 偏移元数匹配地址秩。省略 val 时第二个参数是写值。 | 必填 |
| `val` | 待存入的向量或谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：256 / 128 / 64 | 偏移元数匹配地址秩。省略 val 时第二个参数是写值。 | None |

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
| `src0` | 第一个源操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：不支持<br>B32：int32 / uint32<br>B64：不支持 | 两个输入均为满 64 lanes 且同 dtype，不一致直接报错；返回模 2³² 的差与借位谓词，inactive 得到 0 和 false。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：不支持<br>B32：int32 / uint32<br>B64：不支持 | 两个输入均为满 64 lanes 且同 dtype，不一致直接报错；返回模 2³² 的差与借位谓词，inactive 得到 0 和 false。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：64 | None 生成全真 mask。 | None |

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
| `src0` | 左操作数 | tla.vector | 数据类型：<br>B32：int32 / uint32 | 必须为满 64 lanes，与 `src1` 同 dtype。不一致直接报错。 | 必填 |
| `src1` | 右操作数 | tla.vector | 数据类型：<br>B32：int32 / uint32 | 同上。 | 必填 |
| `carry` | 输入进位谓词 | tla.vector（int1 谓词） | 数据类型：<br>B1：int1<br>lane 数：64 | 必填；lane 为真表示该 lane 加 1。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词）或 None | 数据类型：<br>B1：int1<br>lane 数：64 | None 生成全真 mask；inactive lane 的 sum 为 0、carry_out 为假。 | None |

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
| `src0` | 左操作数（被减数） | tla.vector | 数据类型：<br>B32：int32 / uint32 | 必须为满 64 lanes，与 `src1` 同 dtype。不一致直接报错。 | 必填 |
| `src1` | 右操作数（减数） | tla.vector | 数据类型：<br>B32：int32 / uint32 | 同上。 | 必填 |
| `borrow` | 输入借位谓词 | tla.vector（int1 谓词） | 数据类型：<br>B1：int1<br>lane 数：64 | 必填；lane 为真表示该 lane 减 1。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词）或 None | 数据类型：<br>B1：int1<br>lane 数：64 | None 生成全真 mask；inactive lane 的 diff 为 0、borrow_out 为假。 | None |

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
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：不支持<br>B16：float16<br>B32：float32<br>B64：不支持 | lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 | 必填 |
| `alpha` | 负值分支的斜率 | int / float / 标量 tl.tensor | 数据类型：<br>B8：不支持<br>B16：float16<br>B32：float32<br>B64：不支持 | 必须为 Python 数值或 rank-0 tl.tensor；按 src 的元素类型构造，超出该类型量程按该类型舍入（float16 可能得到 inf），类型不可转换直接报错。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |

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
| `src0` | 第一个源操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：float16<br>B32：float32<br>B64：不支持 | 两操作数必须同 dtype、同 lane 数，不一致直接报错，不做隐式转换或广播；lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 | 必填 |
| `src1` | 第二个源操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：float16<br>B32：float32<br>B64：不支持 | 两操作数必须同 dtype、同 lane 数，不一致直接报错，不做隐式转换或广播；lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |

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
| `mask` | 控制各源 lane 是否参与相加 | tla.vector（int1 谓词）或 None | 数据类型：<br>B1：int1<br>lane 数：与 `src` 一致 | None 生成全真 mask；未选中的输入 lane 按 0 参与。 | None |
| `mode` | 未选中 lane 的输出约定 | 编译期 tla.simd.MODE | MODE.ZEROING / MODE.MERGING | ZEROING 时 inactive lane 与结果上半部均为 0；MERGING 时 inactive lane 回填 `src`。 | MODE.ZEROING |

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
| `mask` | 选择活跃 lane 的谓词 | tla.vector（int1 谓词） | 数据类型：<br>B1：int1<br>lane 数：与 `src` 相同 | 必填，不接受 None。 | 必填 |
| `mode` | 未选中 lane 的输出约定 | 编译期 tla.simd.MODE | MODE.ZEROING / MODE.MERGING | ZEROING 时压缩后剩余的尾部 lane 为 0；MERGING 时 inactive lane 回填 `src`。 | MODE.ZEROING |

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
| `mask` | 活跃 lane 谓词 | tla.vector（int1 谓词） | 数据类型：<br>B1：int1<br>lane 数：等于结果 lane 数 | 必填。 | 必填 |
| `dtype` | 结果元素类型 | 编译期 tl.dtype | 数据类型：<br>B8：int8 / uint8<br>B16：int16 / uint16<br>B32：int32 / uint32 | 决定结果位宽与 lane 数：8 位 256 lanes、16 位 128 lanes、32 位 64 lanes。其它 dtype 直接报错。 | 'u32'（uint32） |

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
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。按输出 lane 匹配。 | None |
| `part` | 选择偶数或奇数槽位 | 编译期 str（使用 PART 常量） | PART.EVEN / PART.ODD | 按转换或 packing 的槽位规则解释。 | PART.EVEN |
| `pp` | 选择四路布局中的槽位 | 编译期 str | pp0 / pp1 / pp2 / pp3 | 用于 4 倍扩展。 | 'pp0' |
| `signed` | 是否按有符号值扩展 | 编译期 bool | False / True | True 符号扩展，False 零扩展。 | False |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE 枚举 | MODE.ZEROING / MODE.MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

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

签名：`tla.simd.vgatherb(base, index, mask=None)`

实现状态：AVE 开发分支已实现；提交和验证边界见[实现基线](../tracking/simd.md#simd-baseline)。

返回：64 lanes、与 base 同 dtype 的数据向量。

编译表示：ave.hir.vgatherb。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `base` | 块读取的基地址 | tla.buffer / buffer_access | 地址空间：UB<br>数据类型：<br>B8：不支持<br>B16：不支持<br>B32：int32 / uint32 / float32<br>B64：不支持 | 一维连续；buf 等价于 buf[0]；base 下标及视图偏移的单位为元素。 | 必填 |
| `index` | 八个块起点相对 base 的字节偏移 | tla.vector | 数据类型：<br>B32：uint32<br>lane 数：64；只使用前 8 个索引 | 使用的偏移必须是 4 的倍数；允许重复块。每个活跃输出对应的地址须在缓冲区内。 | 必填 |
| `mask` | 选择实际读取的输出元素 | tla.vector（int1）或 None | 数据类型：<br>B1：int1<br>lane 数：64 | 每个输出元素一个 mask，非每块一个；false 不读取且输出零。 | None |

这是 block gather：按 index[0:8] 收集八个 32 字节块，每块包含 8 个 32 位元素。对活跃输出 i，result[i] = base[index[i // 8] // 4 + i % 8]。它与 vgather 的逐元素索引、元素偏移单位不同。

4 字节对齐是接口要求，不要求块起点一定按 32 字节对齐；现有专用用例使用 32 字节对齐的块起点。AVE 内部组合现有寄存器算子与带 mask 的 gather 来落实逐输出元素的屏蔽。没有 mode 参数，不会自动检查运行时索引边界。

**调用示例**

x_ub 为 float32 连续 UB，index_ub 为 64 个 uint32；前 8 个索引为合法块起点的字节偏移，八块都在可读范围内。

```python
@tla.jit
def example(x_ub, index_ub):
    with tla.scope.simd():
        index = tla.simd.vlds(index_ub[0])
        result = tla.simd.vgatherb(x_ub[0], index)
```

仓库测试：[memory_completion.py](https://github.com/triton-lang/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/memory_completion.py#L53)。

<a name="api-tla-simd-vabsdif"></a>

## tla.simd.vabsdif

签名：`tla.simd.vabsdif(lhs, rhs, mask=None, mode=tla.simd.MODE.ZEROING)`

返回：tla.vector。

编译表示：ave.hir.vabs_diff。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `lhs` | 左操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：float16<br>B32：float32<br>B64：不支持 | 两操作数必须同 dtype、同 lane 数，不一致直接报错，不做隐式转换或广播；lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 | 必填 |
| `rhs` | 右操作数 | tla.vector | 数据类型：<br>B8：不支持<br>B16：float16<br>B32：float32<br>B64：不支持 | 两操作数必须同 dtype、同 lane 数，不一致直接报错，不做隐式转换或广播；lane 数上限为 2048 / 元素位宽（64 位元素为 4096）。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） 或 None | 数据类型：<br>B1：int1<br>lane 数：与相应向量一致 | None 生成全真 mask。 | None |
| `mode` | 选择操作模式 | 编译期 tla.simd.MODE | MODE.ZEROING / MODE.MERGING | MERGING 与 ZEROING 的 inactive 结果均为零。 | MODE.ZEROING |

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
| `acc` | 累加器 | tla.vector | 数据类型：<br>B16：uint16<br>lane 数：128 | acc 固定 128 lanes，src 与 mask 固定 256 lanes。计数累加保持在 uint16 范围内。 | 必填 |
| `src` | 源数据 | tla.vector | 数据类型：<br>B8：uint8<br>lane 数：256 | acc 固定 128 lanes，src 与 mask 固定 256 lanes。计数累加保持在 uint16 范围内。 | 必填 |
| `mask` | 控制各 lane 是否启用的谓词 | tla.vector（int1 谓词） | 数据类型：<br>B1：int1<br>lane 数：256 | — | 必填 |
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
