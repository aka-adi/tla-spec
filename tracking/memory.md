# 内存评审跟踪

[返回接口目录](../README.md)

<a name="memory-001"></a>

## MEMORY-001 · tla.alloc

[查看规格](../spec/memory.md#api-tla-alloc)

- 原表序号 2 要求分配独占并禁止 planmem retry 复用。当前物理内存是否复用由内存规划决定，需确认该场景的要求。

结论：待决。
