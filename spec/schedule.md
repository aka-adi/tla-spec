# 调度接口规格

[返回接口目录](../README.md)

<a name="api-tl-program-id"></a>

## tl.program_id

签名：`tl.program_id(axis)`

返回：指定轴的程序实例编号。

沿用 Triton 入口，适用场景仍受 TLA 编译路径和目标约束。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `axis` | launch 网格轴 | 编译期 int | 0 / 1 / 2 | TLA adapter 映射到对应 grid 参数。 | 必填 |

**调用示例**

```python
@tla.jit
def example():
    block = tl.program_id(0)
```

仓库测试：[tla_ut/vector/fused_weight.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/vector/fused_weight.py#L35)。

<a name="api-tla-annotate-buffer-versions"></a>

## tla.annotate_buffer_versions

签名：`tla.annotate_buffer_versions(buf, versions)`

返回：None。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `buf` | 要标注的缓冲区 | tla.buffer | 可添加 annotation.mark 的 buffer | — | 必填 |
| `versions` | 缓冲区版本数 | 编译期 int | 正整数 | 1 不添加版本标记，>1 添加 hivm.multi_buffer。 | 必填 |

**调用示例**

```python
@tla.jit
def example():
    tile = tla.alloc((64,), tl.float32)
    tla.annotate_buffer_versions(tile, 2)
```

## 范围说明

以下条目在当前整理表中没有可定位的 TLA 接口名。

| 原序号 | 范围结论 | 涉及场景 |
| --- | --- | --- |
| 179 | 不涉及 | T.PerCoreTask（开发者模式） |
| 180 | 不涉及 | T.Persistent（开发者模式） |
| 181 | 不涉及 | T.Pipelined（开发者模式） |
| 182 | 不涉及 | T.Stage（开发者模式） |
| 183 | 不涉及 | T.Task（开发者模式） |
| 185 | 暂不支持 | T.annotate_manual_multi_buffer |
| 186 | 暂不支持 | T.annotate_unlimit_memory |
| 187 | 暂不支持 | T.assume_conflict |
| 188 | 暂不支持 | T.assume_no_conflict |
| 189 | 暂不支持 | T.AscendBaseTileScheduler |
| 190 | 暂不支持 | T.AscendTileScheduler |
| 191 | 暂不支持 | T.AscendBatchedTileScheduler |
| 192 | 暂不支持 | T.AscendMGroupedTileScheduler |
| 193 | 暂不支持 | T.AscendKGroupedTileScheduler |
