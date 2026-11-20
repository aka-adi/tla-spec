# 调度接口规格

[返回接口目录](../README.md)

<a name="api-tl-program-id"></a>
<a name="api-tla-program-id"></a>

## tla.program_id

计划签名：`tla.program_id(axis)`

返回：指定轴的程序实例编号，为 int32 标量 tl.tensor。

目标沿用现有 `tl.program_id` 的轴语义。当前 TLA 内核仍使用 `tl.program_id`，新入口的实现见 [SCHEDULE-001](../tracking/schedule.md#schedule-001)。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `axis` | launch 网格轴 | 编译期 int | 0 / 1 / 2 | TLA adapter 映射到对应 grid 参数。 | 必填 |

计划调用示例

```python
@tla.jit
def example():
    block = tla.program_id(0)
```

现有 `tl.program_id` 路径的仓库测试：[tla_ut/vector/fused_weight.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/vector/fused_weight.py#L35)。

<a name="api-tla-annotate-buffer-versions"></a>
<a name="api-tla-annotate-multi-buffer"></a>

## tla.annotate_multi_buffer

计划签名：`tla.annotate_multi_buffer(buf, versions)`

返回：None。

多缓冲版本数从 [tla.alloc](memory.md#api-tla-alloc) 的分配签名移到显式标注。当前实现入口为 `tla.annotate_buffer_versions`，计划沿用现有参数与行为，见 [SCHEDULE-002](../tracking/schedule.md#schedule-002)。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `buf` | 要标注的缓冲区 | tla.buffer | 可添加 annotation.mark 的 buffer | — | 必填 |
| `versions` | 缓冲区版本数 | 编译期 int | 正整数 | 1 不添加版本标记，>1 添加 hivm.multi_buffer。 | 必填 |

计划调用示例

```python
@tla.jit
def example():
    tile = tla.alloc((64,), tl.float32)
    tla.annotate_multi_buffer(tile, 2)
```

<a name="api-tla-sub-block-id"></a>

## tla.sub_block_id

签名：`tla.sub_block_id()`

返回：当前 AIV 子块在所属 Cube 块内的索引，为 int64 标量 tl.tensor。

该接口读取当前 AIV 子块的编号，不接收参数，也不产生副作用。

约束：返回值为非负整数，从 0 开始，上界由运行时的 AIV 与 Cube 子块绑定关系决定，AIV 与 Cube 按 1:2 绑定时取 0 或 1。同一子块内重复调用返回相同的值。结果可直接参与整数比较和地址计算，也可作为 tla.view、tla.subview 的偏移。TLA 不提供子块总数查询，调用方不得假定固定的子块数量。仓库现有用例均在 Vector 执行域内调用，Cube 执行域的取值未纳入当前规格。

**调用示例**

前提：`out_ub` 为 16×32 fp16 的 UB 缓冲区，`y` 为同形状的 GM fp16 指针。Fixpipe 与 L12UB 默认只写入子块 0 的 UB，因此只让子块 0 回写 GM。

```python
@tla.jit
def example(out_ub, y):
    with tla.scope.vector():
        if tla.sub_block_id() == 0:
            tla.store_ub_to_gm(out_ub, tla.view(y, (16, 32)))
```

仓库测试：[tla_ut/dma/nd2nz.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dma/nd2nz.py#L28)。

