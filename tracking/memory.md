# 内存评审跟踪

[返回接口目录](../README.md)

<a name="memory-001"></a>

## MEMORY-001 · tla.alloc

[查看规格](../spec/memory.md#api-tla-alloc)

- 原表序号 2 要求分配独占并禁止 planmem retry 复用。当前物理内存是否复用由内存规划决定，需确认该场景的要求。

结论：待决。移除分配参数和物理内存能否复用是两个问题。现有实现仍由后端内存规划决定复用，已确认的签名变更记录在 [MEMORY-002](#memory-002)。

<a name="memory-002"></a>

## MEMORY-002 · tla.alloc 签名调整

[查看规格](../spec/memory.md#api-tla-alloc)

- 目标签名为 `tla.alloc(shape, dtype, space=None)`。删除 `scope` 关键字，调用方用 `space` 指定地址空间。
- 移除 `versions`。需要多缓冲时改用 [tla.annotate_multi_buffer](../spec/schedule.md#api-tla-annotate-multi-buffer)。

结论：设计已确认，待实现。当前签名仍接收这两个参数。实现更新后核对旧调用的迁移、新签名和默认 UB 行为。

<a name="memory-003"></a>

## MEMORY-003 · tla.view 的空间范围

[查看规格](../spec/memory.md#api-tla-view)

- `tla.view` 曾通过 `space` 参数选择地址空间，并支持从片上指针构造视图，与 `tla.subview` 的职责重叠。

结论：已确认。`tla.view` 固定为 GM，只接受 GM kernel 指针，不再提供 `space` 参数；已有缓冲区的切片统一由 [tla.subview](../spec/memory.md#api-tla-subview) 承担，空间继承源缓冲区。

实现待对齐：当前 `view.py` 仍保留 `space` 参数和片上指针的地址绑定路径，PR #3224 合入后更新签名与源码链接。

<a name="memory-004"></a>

## MEMORY-004 · tla.view / tla.subview 的 sizes 与 strides

[查看规格](../spec/memory.md#api-tla-view) · [tla.subview](../spec/memory.md#api-tla-subview)

- PR #3224 允许 `sizes` 和 `strides` 为运行时整数标量 tensor，任一维度动态时整视图的边界检查推迟到运行时；当前规格只允许编译期正整数。

结论：待决。需确认本轮规格保留静态限制，还是与实现一致地允许运行时 sizes/strides。空间范围见 [MEMORY-003](#memory-003)。
