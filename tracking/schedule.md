# 调度评审跟踪

[返回接口目录](../README.md)

<a name="schedule-001"></a>

## SCHEDULE-001 · tla.program_id

[查看规格](../spec/schedule.md#api-tla-program-id)

- 在 TLA 命名空间提供 `program_id(axis)`，沿用现有轴语义。核对新入口在 TLA 编译链路中的 0 / 1 / 2 轴映射及非法轴校验。

结论：设计已确认，待实现。当前参照路径为 `tl.program_id`。

<a name="schedule-002"></a>

## SCHEDULE-002 · tla.annotate_multi_buffer

[查看规格](../spec/schedule.md#api-tla-annotate-multi-buffer)

- 将 `tla.annotate_buffer_versions` 改名为 `tla.annotate_multi_buffer`，配合 [MEMORY-002](memory.md#memory-002) 将多缓冲配置移出分配签名。核对新入口的参数及版本标记生成。

结论：设计已确认，待实现。当前代码仍使用旧入口。
