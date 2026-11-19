# SIMD评审跟踪

[返回接口目录](../README.md)

<a name="simd-baseline"></a>

## 魏子尧负责的新增接口实现基线

这 10 个接口的实现负责人均为魏子尧，补充范围为：vcadd、vcgadd / vcgmax / vcgmin、vld2、ppack / punpack、pintlv / pdintlv、vgatherb。

规格依据为 [TLA d54604c1133d](https://github.com/wzy1214/triton-ascend/tree/d54604c1133df2631c6b32d8649eea79c6b2c975) 与其配套 [AscendNPU-IR bd88eea996b4](https://gitcode.com/Ascend/AscendNPU-IR/tree/bd88eea996b49aee9898465b77815bb32f762bb6)，核对了公开入口、builder、AVE 类型约束及 lowering。各小节提供固定提交的源码和用例链接；开发分支已实现不等于主线已合入或完成转测。

在核对的 [main-dev ff7b02c72795](https://github.com/triton-lang/triton-ascend/tree/ff7b02c727951aa2f279918121025a7decfeb28a) 中，这 10 个接口尚无公开入口；不能以开发分支实现认定该主线基线可执行。

本次仅作源码与文档核对，未重新编译或运行真卡。引用用例不等于本次执行通过；既有联调记录限于 Ascend950 / Process，不能外推至其他目标、后端或最新主线合并版本。

<a name="simd-011"></a>

## SIMD-011 · tla.simd.vadds

[查看规格](../spec/simd.md#api-tla-simd-vadds)

- `scalar`：考虑前端提前拒绝单 lane 向量。

结论：待决。

<a name="simd-013"></a>

## SIMD-013 · tla.simd.vaxpy

[查看规格](../spec/simd.md#api-tla-simd-vaxpy)

- `scalar`：单 lane vector 会进入 AVE 标量操作数，类型契约不符，见 [_scalar_handle](https://github.com/triton-lang/triton-ascend/blob/ff7b02c727951aa2f279918121025a7decfeb28a/third_party/ascend/language/tla/simd.py#L1136) 与 [AVE_VecScalarOp](https://gitcode.com/Ascend/AscendNPU-IR/blob/bd228faec3b574ce116edb54ad8437355d630523/bishengir/include/bishengir/Dialect/HIVMAVE/IR/HIVMAVEOps.td#L111)。正例保留同 dtype 的秩 0 tensor；单 lane vector 的前端拒绝及诊断用例待补。

结论：已确认当前限制；实现修复与回归待完成。

<a name="simd-014"></a>

## SIMD-014 · tla.simd.vcadd

[vcadd 规格](../spec/simd.md#api-tla-simd-vcadd)

入口已实现，复用既有 AVE ReductionOp。[simd.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/language/tla/simd.py#L2645) 与 [HIVMAVEOps.td](https://gitcode.com/Ascend/AscendNPU-IR/blob/bd88eea996b49aee9898465b77815bb32f762bb6/bishengir/include/bishengir/Dialect/HIVMAVE/IR/HIVMAVEOps.td#L767) 均规定返回同型向量，只有第 0 个 lane 是归约结果；不能承诺尾部为零或把内部扩宽当作公开返回类型。

[memory_completion.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/memory_completion.py#L68) 目前覆盖 int32、float32；前端未限制其他 dtype，但不能据此声明完整执行支持。后续需补逐类型、溢出及浮点特殊值矩阵。

结论：开发分支已提供入口，主线合入仍待确认；其他类型的可执行覆盖待补，规格先写明已核对范围。

<a name="simd-015"></a>

## SIMD-015 · tla.simd.vcgadd

[vcgadd 规格](../spec/simd.md#api-tla-simd-vcgadd)

开发分支已提供独立的 AVE GroupReductionOp：Ascend950 的 8 个连续 32 字节组，前 8 个结果有效、尾部补零，mask 选择输入元素。vcgadd 的 int16 / uint16 扩宽为 int32 / uint32；max / min 保持输入类型。

[simd.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/language/tla/simd.py#L3087)、[HIVMAVEOps.td](https://gitcode.com/Ascend/AscendNPU-IR/blob/bd88eea996b49aee9898465b77815bb32f762bb6/bishengir/include/bishengir/Dialect/HIVMAVE/IR/HIVMAVEOps.td#L514) 与 [group_reduce.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/group_reduce.py#L33) 相互对应。verifier 拒绝浮点输入配无符号 cast；Ascend960 当前明确不支持。浮点顺序、NaN、正负零及空 min/max 组遵循目标指令，未另行承诺跨硬件统一值。

结论：开发分支已提供入口，主线合入仍待确认，当前范围已写回规格；新增目标需要独立语义和验证。

<a name="simd-016"></a>

## SIMD-016 · tla.simd.vcgmax

[vcgmax 规格](../spec/simd.md#api-tla-simd-vcgmax)

开发分支已提供独立的 AVE GroupReductionOp：Ascend950 的 8 个连续 32 字节组，前 8 个结果有效、尾部补零，mask 选择输入元素。vcgadd 的 int16 / uint16 扩宽为 int32 / uint32；max / min 保持输入类型。

[simd.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/language/tla/simd.py#L3101)、[HIVMAVEOps.td](https://gitcode.com/Ascend/AscendNPU-IR/blob/bd88eea996b49aee9898465b77815bb32f762bb6/bishengir/include/bishengir/Dialect/HIVMAVE/IR/HIVMAVEOps.td#L514) 与 [group_reduce.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/group_reduce.py#L33) 相互对应。verifier 拒绝浮点输入配无符号 cast；Ascend960 当前明确不支持。浮点顺序、NaN、正负零及空 min/max 组遵循目标指令，未另行承诺跨硬件统一值。

结论：开发分支已提供入口，主线合入仍待确认，当前范围已写回规格；新增目标需要独立语义和验证。

<a name="simd-017"></a>

## SIMD-017 · tla.simd.vcgmin

[vcgmin 规格](../spec/simd.md#api-tla-simd-vcgmin)

开发分支已提供独立的 AVE GroupReductionOp：Ascend950 的 8 个连续 32 字节组，前 8 个结果有效、尾部补零，mask 选择输入元素。vcgadd 的 int16 / uint16 扩宽为 int32 / uint32；max / min 保持输入类型。

[simd.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/language/tla/simd.py#L3112)、[HIVMAVEOps.td](https://gitcode.com/Ascend/AscendNPU-IR/blob/bd88eea996b49aee9898465b77815bb32f762bb6/bishengir/include/bishengir/Dialect/HIVMAVE/IR/HIVMAVEOps.td#L514) 与 [group_reduce.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/group_reduce.py#L33) 相互对应。verifier 拒绝浮点输入配无符号 cast；Ascend960 当前明确不支持。浮点顺序、NaN、正负零及空 min/max 组遵循目标指令，未另行承诺跨硬件统一值。

结论：开发分支已提供入口，主线合入仍待确认，当前范围已写回规格；新增目标需要独立语义和验证。

<a name="simd-023"></a>

## SIMD-023 · tla.simd.vcvt

[查看规格](../spec/simd.md#api-tla-simd-vcvt)

- 按目标维护源/目标类型对 × 舍入 × 半寄存器选择（part）× mask 粒度的设备矩阵，不能用单一类型对代表全部转换。

- `src`：按目标验证所有类型对的舍入、饱和和 lane 映射。

结论：待决。

<a name="simd-032"></a>

## SIMD-032 · tla.simd.vld

[查看规格](../spec/simd.md#api-tla-simd-vld)

- 原需求是按分布加载连续数据，当前 vld 接收索引并委托 vgather。当前连续加载使用 vlds；原始连续 vld 的命名和迁移入口仍待确认。

结论：待决。

<a name="simd-033"></a>

## SIMD-033 · tla.simd.vld2

[vld2 规格](../spec/simd.md#api-tla-simd-vld2)

[simd.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/language/tla/simd.py#L936) 已通过两次 NORM vload 和一次 vdintlv 组合实现，完整读取两个寄存器，再返回偶 / 奇位置数据；off 按元素计。它没有依赖旧的双结果分布 load lowering。

[memory_completion.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/memory_completion.py#L68) 包含 int16、float16、int32、float32。uint16、uint32、bfloat16 虽被入口接受，尚需专用执行覆盖。

结论：开发分支已提供入口，主线合入仍待确认；剩余 dtype 的验证待补。

<a name="simd-042"></a>

## SIMD-042 · tla.simd.vmull

[查看规格](../spec/simd.md#api-tla-simd-vmull)

问题单报告的 950PR 实测：`uint32(0xFFFFFFFF) × uint32(2)` 返回低半部 `0xFFFFFFFE`、高半部 `0xFFFFFFFF`，正确高半部应为 `0x00000001`；相同位型按 `int32` 计算符合有符号语义。本次未重跑该设备用例。

源码复核：[前端](https://github.com/triton-lang/triton-ascend/blob/ff7b02c727951aa2f279918121025a7decfeb28a/third_party/ascend/language/tla/simd.py#L2169) 接受 uint32，但 [builder](https://github.com/triton-lang/triton-ascend/blob/ff7b02c727951aa2f279918121025a7decfeb28a/third_party/ascend/language/tla/cpp/tla_builder.cc#L913) 创建 VFVMULLOp 未传 cast；[ODS](https://gitcode.com/Ascend/AscendNPU-IR/blob/bd228faec3b574ce116edb54ad8437355d630523/bishengir/include/bishengir/Dialect/HIVMAVE/IR/HIVMAVEOps.td#L76) 默认 cast_signed，[库调用类型推导](https://gitcode.com/Ascend/AscendNPU-IR/blob/bd228faec3b574ce116edb54ad8437355d630523/bishengir/lib/Dialect/HIVMAVE/IR/HIVMAVEOps.cpp#L56) 使用此属性。无符号信息未传递到 AVE。

转测处理：保留 int32 用例；uint32 暂不列设备正例，待传递无符号 cast 后回归最高位为 1、普通值、零值及 mask。不能仅凭低半部正确判断完整乘积正确。

结论：已确认当前缺陷；本次仅修正规格，代码尚未修复。

<a name="simd-043"></a>

## SIMD-043 · tla.simd.vmuls

[查看规格](../spec/simd.md#api-tla-simd-vmuls)

- `scalar`：单 lane vector 会进入 AVE 标量操作数，类型契约不符，见 [_scalar_handle](https://github.com/triton-lang/triton-ascend/blob/ff7b02c727951aa2f279918121025a7decfeb28a/third_party/ascend/language/tla/simd.py#L1136) 与 [AVE_VecScalarOp](https://gitcode.com/Ascend/AscendNPU-IR/blob/bd228faec3b574ce116edb54ad8437355d630523/bishengir/include/bishengir/Dialect/HIVMAVE/IR/HIVMAVEOps.td#L111)。正例保留同 dtype 的秩 0 tensor；单 lane vector 的前端拒绝及诊断用例待补。

- B64 的 Python int 常量经 float 和接收 double 的 [create_simd_scalar](https://github.com/triton-lang/triton-ascend/blob/ff7b02c727951aa2f279918121025a7decfeb28a/third_party/ascend/language/tla/cpp/tla_builder.cc#L1048) 构造，`2**53+1` 在建 IR 前已变为 `2**53`。保留同 dtype 的秩 0 tensor 路径；常量、tensor、最高位与 mask 组合需分别回归。

结论：已确认当前限制；实现修复与回归待完成。

<a name="simd-056"></a>

## SIMD-056 · tla.simd.vsts

[查看规格](../spec/simd.md#api-tla-simd-vsts)

- `dist`：按目标维护 dist × dtype × 地址对齐的设备矩阵，分别覆盖 packing、交错和通道合并，不能把某一分布用例外推至其他分布。

结论：待决。

<a name="simd-063"></a>

## SIMD-063 · tla.simd.ppack

[ppack 规格](../spec/simd.md#api-tla-simd-ppack)

[simd.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/language/tla/simd.py#L3001) 已通过独立 AVE 接口实现。逻辑 pack / unpack 使用 LOWER / HIGHER 连续半区；pack 的另一半补 false，unpack 只选连续的一半。不能把它替换成旧 preg.cast。

[predicate_ops.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/predicate_ops.py#L33) 及该文件的生成谓词、组合用例覆盖逻辑长度、半区和双结果；实现与源、结果类型约束已写回规格。

结论：开发分支已提供入口，主线合入仍待确认；验证边界沿用实现基线，转测状态不自动修改。

<a name="simd-064"></a>

## SIMD-064 · tla.simd.punpack

[punpack 规格](../spec/simd.md#api-tla-simd-punpack)

[simd.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/language/tla/simd.py#L3011) 已通过独立 AVE 接口实现。逻辑 pack / unpack 使用 LOWER / HIGHER 连续半区；pack 的另一半补 false，unpack 只选连续的一半。不能把它替换成旧 preg.cast。

[predicate_ops.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/predicate_ops.py#L33) 及该文件的生成谓词、组合用例覆盖逻辑长度、半区和双结果；实现与源、结果类型约束已写回规格。

结论：开发分支已提供入口，主线合入仍待确认；验证边界沿用实现基线，转测状态不自动修改。

<a name="simd-065"></a>

## SIMD-065 · tla.simd.pintlv

[pintlv 规格](../spec/simd.md#api-tla-simd-pintlv)

[simd.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/language/tla/simd.py#L3030) 已通过独立 AVE 接口实现。交错 / 去交错接口返回两个逻辑谓词，输入密度相同；宽度从 lane 数推导，不需要 width 参数。

[predicate_ops.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/predicate_ops.py#L33) 及该文件的生成谓词、组合用例覆盖逻辑长度、半区和双结果；实现与源、结果类型约束已写回规格。

结论：开发分支已提供入口，主线合入仍待确认；验证边界沿用实现基线，转测状态不自动修改。

<a name="simd-066"></a>

## SIMD-066 · tla.simd.pdintlv

[pdintlv 规格](../spec/simd.md#api-tla-simd-pdintlv)

[simd.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/language/tla/simd.py#L3040) 已通过独立 AVE 接口实现。交错 / 去交错接口返回两个逻辑谓词，输入密度相同；宽度从 lane 数推导，不需要 width 参数。

[predicate_ops.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/predicate_ops.py#L33) 及该文件的生成谓词、组合用例覆盖逻辑长度、半区和双结果；实现与源、结果类型约束已写回规格。

结论：开发分支已提供入口，主线合入仍待确认；验证边界沿用实现基线，转测状态不自动修改。

<a name="simd-070"></a>

## SIMD-070 · tla.simd.vaddcs

[查看规格](../spec/simd.md#api-tla-simd-vaddcs)

- 实现已存在：`vaddcs(src0, src1, carry, mask=None)`，返回 `(sum, carry_out)`；64 lane int32/uint32，`carry` 必填。

结论：待决。规格已按实现补齐，请确认签名后关闭。

<a name="simd-071"></a>

## SIMD-071 · tla.simd.vsubcs

[查看规格](../spec/simd.md#api-tla-simd-vsubcs)

- 实现已存在：`vsubcs(src0, src1, borrow, mask=None)`，返回 `(diff, borrow_out)`；约束同 `vaddcs`。
- 注意极性：本接口的 `borrow` 与 AVE 侧 no-borrow 的 `carry_in` 相反，前端已取反。

结论：待决。规格已按实现补齐，请确认签名与极性描述后关闭。

<a name="simd-074"></a>

## SIMD-074 · tla.simd.vcpadd

[查看规格](../spec/simd.md#api-tla-simd-vcpadd)

- 实现已存在：`vcpadd(src, mask=None, mode=...)`，返回 `tla.vector`（原提案的 `-> tla,tensor` 是笔误）。
- 语义为相邻 lane 配对求和：结果 lane `j < N/2` 为 `src[2j] + src[2j+1]`，其余 lane 为 0；只支持 f16/f32 满寄存器。

结论：待决。规格已按实现补齐，请确认后关闭。

<a name="simd-076"></a>

## SIMD-076 · tla.simd.vusqz

[查看规格](../spec/simd.md#api-tla-simd-vusqz)

- 原描述"拟将谓词展开为索引向量"与实现不符：实现返回排他前缀计数（结果 lane `i` 为索引小于 `i` 的活跃 lane 数）。
- 实现签名 `vusqz(mask, dtype='u32')`，`dtype` 决定结果位宽与 lane 数。

结论：待决。规格已按实现更正，请确认后关闭。

<a name="simd-078"></a>

## SIMD-078 · tla.simd.vgatherb

[vgatherb 规格](../spec/simd.md#api-tla-simd-vgatherb)

[simd.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/language/tla/simd.py#L967) 与 [HIVMAVEOps.td](https://gitcode.com/Ascend/AscendNPU-IR/blob/bd88eea996b49aee9898465b77815bb32f762bb6/bishengir/include/bishengir/Dialect/HIVMAVE/IR/HIVMAVEOps.td#L192) 已定义为八个 32 字节块的 gather。数据为 int32 / uint32 / float32；64 个 uint32 索引中使用前 8 个，索引单位为字节，mask 则逐输出元素生效。

AVE 通过现有寄存器操作展开索引并调用 masked gather，因此旧的原生无 mask block-gather 限制不等于当前接口没有 mask。索引只要求 4 字节对齐；[memory_completion.py](https://github.com/wzy1214/triton-ascend/blob/d54604c1133df2631c6b32d8649eea79c6b2c975/third_party/ascend/unittest/tla_ut/simd/memory_completion.py#L108) 的现有块起点均为 32 字节对齐，4 字节对齐但非 32 字节对齐的地址仍需独立执行用例。

结论：开发分支已提供入口，主线合入仍待确认；非 32 字节对齐块起点及最终目标版本回归待补。

<a name="simd-080"></a>

## SIMD-080 · tla.simd.dhistv2

[查看规格](../spec/simd.md#api-tla-simd-dhistv2)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：待决。

<a name="simd-081"></a>

## SIMD-081 · tla.simd.membar

[查看规格](../spec/simd.md#api-tla-simd-membar)

- 规格已把 `scope` 定为枚举 `tla.simd.SCOPE`（16 个成员，与 `ave.hir.membar` 的 i32 编码 0–15 一一对应），不再按字符串描述。
- 实现现状：`tla.simd.membar(scope)` 仍只接受字符串（内部 `str(scope).upper()` 后交给 C++ 白名单校验），没有 `SCOPE` 常量类。
- 待办：前端补 `class SCOPE` 及取值校验（非法值报错）；补完后规格无需再改。

结论：待决。
