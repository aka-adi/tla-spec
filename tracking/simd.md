# SIMD评审跟踪

[返回接口目录](../README.md)

<a name="simd-011"></a>

## SIMD-011 · tla.simd.vadds

[查看规格](../spec/simd.md#api-tla-simd-vadds)

- `scalar`：考虑前端提前拒绝单 lane 向量。

结论：待决。

<a name="simd-013"></a>

## SIMD-013 · tla.simd.vaxpy

[查看规格](../spec/simd.md#api-tla-simd-vaxpy)

- `scalar`：考虑前端提前拒绝单 lane 向量。

结论：待决。

<a name="simd-014"></a>

## SIMD-014 · tla.simd.vcadd

[查看规格](../spec/simd.md#api-tla-simd-vcadd)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：待决。

<a name="simd-015"></a>

## SIMD-015 · tla.simd.vcgadd

[查看规格](../spec/simd.md#api-tla-simd-vcgadd)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：待决。

<a name="simd-016"></a>

## SIMD-016 · tla.simd.vcgmax

[查看规格](../spec/simd.md#api-tla-simd-vcgmax)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：待决。

<a name="simd-017"></a>

## SIMD-017 · tla.simd.vcgmin

[查看规格](../spec/simd.md#api-tla-simd-vcgmin)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：待决。

<a name="simd-023"></a>

## SIMD-023 · tla.simd.vcvt

[查看规格](../spec/simd.md#api-tla-simd-vcvt)

- 建议按源和目标类型对、mask 粒度与舍入组合维护测试矩阵。

- `src`：按目标验证所有类型对的舍入、饱和和 lane 映射。

结论：待决。

<a name="simd-032"></a>

## SIMD-032 · tla.simd.vld

[查看规格](../spec/simd.md#api-tla-simd-vld)

- 原需求是按分布加载连续数据，当前 vld 接收索引并委托 vgather。确认公开命名及旧调用的迁移入口。

结论：待决。

<a name="simd-033"></a>

## SIMD-033 · tla.simd.vld2

[查看规格](../spec/simd.md#api-tla-simd-vld2)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：待决。

<a name="simd-042"></a>

## SIMD-042 · tla.simd.vmull

[查看规格](../spec/simd.md#api-tla-simd-vmull)

- 加一组 uint32 高位为 1 的输入，核对高半部符号解释。

结论：待决。

<a name="simd-043"></a>

## SIMD-043 · tla.simd.vmuls

[查看规格](../spec/simd.md#api-tla-simd-vmuls)

- `scalar`：考虑前端提前拒绝单 lane 向量。

结论：待决。

<a name="simd-056"></a>

## SIMD-056 · tla.simd.vsts

[查看规格](../spec/simd.md#api-tla-simd-vsts)

- `dist`：逐目标补齐 packing、交错和通道合并的分布矩阵。

结论：待决。

<a name="simd-063"></a>

## SIMD-063 · tla.simd.ppack

[查看规格](../spec/simd.md#api-tla-simd-ppack)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：待决。

<a name="simd-064"></a>

## SIMD-064 · tla.simd.punpack

[查看规格](../spec/simd.md#api-tla-simd-punpack)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：待决。

<a name="simd-065"></a>

## SIMD-065 · tla.simd.pintlv

[查看规格](../spec/simd.md#api-tla-simd-pintlv)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：待决。

<a name="simd-066"></a>

## SIMD-066 · tla.simd.pdintlv

[查看规格](../spec/simd.md#api-tla-simd-pdintlv)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：待决。

<a name="simd-070"></a>

## SIMD-070 · tla.simd.vaddcs

[查看规格](../spec/simd.md#api-tla-simd-vaddcs)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：待决。

<a name="simd-071"></a>

## SIMD-071 · tla.simd.vsubcs

[查看规格](../spec/simd.md#api-tla-simd-vsubcs)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：待决。

<a name="simd-074"></a>

## SIMD-074 · tla.simd.vcpadd

[查看规格](../spec/simd.md#api-tla-simd-vcpadd)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：待决。

<a name="simd-076"></a>

## SIMD-076 · tla.simd.vusqz

[查看规格](../spec/simd.md#api-tla-simd-vusqz)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：待决。

<a name="simd-078"></a>

## SIMD-078 · tla.simd.vgatherb

[查看规格](../spec/simd.md#api-tla-simd-vgatherb)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：待决。

<a name="simd-080"></a>

## SIMD-080 · tla.simd.dhistv2

[查看规格](../spec/simd.md#api-tla-simd-dhistv2)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：待决。
