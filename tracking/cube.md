# Cube评审跟踪

[返回接口目录](../README.md)

<a name="cube-001"></a>

## CUBE-001 · tla.copy_ub_to_l1

[查看规格](../spec/cube.md#api-tla-copy-ub-to-l1)

- `copy_ub_to_l1` 仅保留 `src`、`dst`，不提供填充配置；物理布局由调用方准备。

结论：本轮范围已确定，按规格进入待转测。暂不支持手动设置 `pad_mode` 和 `pad_value`；AIV 写入后供 AIC 消费须建立跨核依赖。

<a name="cube-003"></a>

## CUBE-003 · tla.copy_l1_to_bt

[查看规格](../spec/cube.md#api-tla-copy-l1-to-bt)

结论：已关闭，本轮不需要。L1→BT→MMAD 路径仍存在 AIC trap，尚未定位到搬运、地址规划或消费端；修复后再重新评审。

<a name="cube-004"></a>

## CUBE-004 · tla.copy_l1_to_fb

[查看规格](../spec/cube.md#api-tla-copy-l1-to-fb)

- 目前只有预留入口，建议在完成后端映射前标记为不可转测。

结论：已关闭。本轮不需要，待补齐实现后另行评审。

<a name="cube-005"></a>

## CUBE-005 · tla.copy_l0c_to_ub

[查看规格](../spec/cube.md#api-tla-copy-l0c-to-ub)

- 需限定 `dma_mode`、`pre_quant`、`pre_relu` 与目的子块的合法组合。

结论：本轮范围已确定，按规格进入待转测。源为 FP32；三种目的布局支持 FP32/FP16/BF16 输出及无激活/ReLU。双目标仅提供 NZ2ND，且须显式 `sub_block_idx=None`。额外激活模式、任意 scale 及其他双目标布局暂不开放。

<a name="cube-006"></a>

## CUBE-006 · tla.copy_l0c_to_l1

[查看规格](../spec/cube.md#api-tla-copy-l0c-to-l1)

- 需按目的布局限定类型转换和激活组合。

结论：本轮范围已确定，按规格进入待转测。NZ2ND/NZ2NZ 支持 FP32/FP16/BF16；NZ2DN 仅保留 FP16/BF16，FP32 路径暂不具备目标实现。激活仅提供无激活/ReLU；不提供双目标或子块选择，channel_split/C0 padding 保持内部默认值。

<a name="cube-007"></a>

## CUBE-007 · tla.store_l0c_to_gm

[查看规格](../spec/cube.md#api-tla-store-l0c-to-gm)

- 需明确目的布局、类型转换与激活组合。

结论：本轮范围已确定，按规格进入待转测。NZ2ND/NZ2NZ/NZ2DN 支持 FP32/FP16/BF16 输出及无激活/ReLU；不提供双目标或子块选择。暂不支持手动设置 scale、额外激活模式及手动 UnitFlag。
