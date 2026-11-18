# DMA评审跟踪

[返回接口目录](../README.md)

<a name="dma-004"></a>

## DMA-004 · tla.copy_l1_to_l0

[查看规格](../spec/dma.md#api-tla-copy-l1-to-l0)

- 对齐 RegBase 模板库的单 batch 搬运形式，以源、目的 buffer 视图表达分块；公开配置仅保留 `is_transpose`。

结论：接口设计已确认。每次调用仅处理单 batch，不提供批量搬运配置；`k_part*` 和 `m` 不作为公开参数。

实现待对齐：现有前端及独立操作 lowering 仍依赖旧分区字段，需改为从 buffer 视图构造模板参数，并采用单 batch 搬运路径。

依据：AscendNPU-IR `c521aba2` 的 RegBase 模板 `copy_l1_to_l0a_a5.hpp`、`copy_l1_to_l0b_a5.hpp`。后续需核对 L0A/L0B 与转置的组合行为；旧分区接口的结果不能替代新签名的覆盖。

<a name="dma-005"></a>

## DMA-005 · tla.make_partition_view

[查看规格](../spec/dma.md#api-tla-make-partition-view)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：已关闭。本轮不需要，待补齐实现后另行评审。

<a name="dma-006"></a>

## DMA-006 · tla.make_strided_view

[查看规格](../spec/dma.md#api-tla-make-strided-view)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：已关闭。本轮不需要，待补齐实现后另行评审。

<a name="dma-007"></a>

## DMA-007 · tla.make_gather_scatter_view

[查看规格](../spec/dma.md#api-tla-make-gather-scatter-view)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：已关闭。本轮不需要，待补齐实现后另行评审。

<a name="dma-008"></a>

## DMA-008 · tla.load_view

[查看规格](../spec/dma.md#api-tla-load-view)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：已关闭。本轮不需要，待补齐实现后另行评审。

<a name="dma-009"></a>

## DMA-009 · tla.store_view

[查看规格](../spec/dma.md#api-tla-store-view)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：已关闭。本轮不需要，待补齐实现后另行评审。

<a name="dma-010"></a>

## DMA-010 · tla.set_pad_value

[查看规格](../spec/dma.md#api-tla-set-pad-value)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：已关闭。本轮不需要，待补齐实现后另行评审。

<a name="dma-011"></a>

## DMA-011 · tla.set_post_copy

[查看规格](../spec/dma.md#api-tla-set-post-copy)

- 确认是否纳入本轮规格并补齐目标实现与可验证的签名。

结论：已关闭。本轮不需要，待补齐实现后另行评审。

<a name="dma-012"></a>

## DMA-012 · tla.stride_store

[查看规格](../spec/dma.md#api-tla-stride-store)

原评审问题：是否纳入本轮规格并补齐实现。

结论：已关闭。确认不需要此接口，退出规格范围。

<a name="dma-013"></a>

## DMA-013 · tla.stride_load

[查看规格](../spec/dma.md#api-tla-stride-load)

原评审问题：是否纳入本轮规格并补齐实现。

结论：已关闭。确认不需要此接口，退出规格范围。
