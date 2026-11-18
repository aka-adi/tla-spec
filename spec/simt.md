# SIMT接口规格

[返回接口目录](../README.md)

<a name="api-tla-scope-simt"></a>

## tla.scope.simt

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`tla.scope.simt(num_warps=32, super_block_factor=1):`

<a name="api-tla-simt-to-ptr"></a>

## tla.simt.to_ptr

**提案。** 当前源码未找到对应入口；签名和目标约束待评审。

原始提案：`tla.simt.to_ptr ( src: tla.buffer ) -> tt.ptr`

## 范围说明

以下条目在当前整理表中没有可定位的 TLA 接口名。

| 原序号 | 范围结论 | 涉及场景 |
| --- | --- | --- |
| 121 | tla暂不开放simt vf内thread级编程 | T.get_thread_binding |
| 128 | tla暂不开放simt vf内thread级编程 | T.warp_reduce_max |
| 129 | tla暂不开放simt vf内thread级编程 | T.warp_reduce_min |
| 130 | tla暂不开放simt vf内thread级编程 | T.warp_reduce_sum |
| 131 | 暂不开放 | T.get_block_binding |
| 132 | 暂不开放 | T.get_block_bindings |
| 133 | 暂不开放 | T.get_block_extent |
| 134 | 暂不开放 | T.get_block_extents |
| 135 | 暂不开放 | T.get_thread_extent |
| 136 | 暂不开放 | T.get_thread_extents |
