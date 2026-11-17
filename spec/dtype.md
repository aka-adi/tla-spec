# 数据类型接口规格

[返回接口目录](../README.md)

<a name="api-tla-bfloat16"></a>

## tla.bfloat16

签名：`tla.bfloat16`

返回：Triton dtype 对象。

**调用示例**

```python
@tla.jit
def example():
    tile = tla.alloc((64,), tla.bfloat16)
```

仓库测试：[tla_ut/dtype/dtype_kernel.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dtype/dtype_kernel.py#L50)。

<a name="api-tla-dtype"></a>

## tla.dtype

签名：`tla.dtype`

返回：Triton dtype 对象。

| 参数名 | 参数含义 | 参数类型 | 合法参数范围 | 其他约束 | 默认值 |
| --- | --- | --- | --- | --- | --- |
| `name` | Triton 的 canonical dtype 名称 | 编译期 str | int1 / int8 / int16 / int32 / int64 / uint8 / uint16 / uint32 / uint64<br>fp8e4b15 / fp8e4nv / fp8e4b8 / fp8e5 / fp8e5b16 / fp16 / bf16 / fp32 / fp64 / void | 仅构造 dtype 对象。具体操作仍须匹配其各自的类型范围。 | 必填 |

**调用示例**

```python
@tla.jit
def example():
    element_type = tla.dtype('fp32')
    tile = tla.alloc((64,), element_type)
```

仓库测试：[tla_ut/dtype/dtype_api.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dtype/dtype_api.py#L55)。

<a name="api-tla-float32"></a>

## tla.float32

签名：`tla.float32`

返回：Triton dtype 对象。

**调用示例**

```python
@tla.jit
def example():
    tile = tla.alloc((64,), tla.float32)
```

仓库测试：[tla_ut/dtype/dtype_kernel.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dtype/dtype_kernel.py#L52)。

<a name="api-tla-int32"></a>

## tla.int32

签名：`tla.int32`

返回：Triton dtype 对象。

**调用示例**

```python
@tla.jit
def example():
    tile = tla.alloc((64,), tla.int32)
```

仓库测试：[tla_ut/dtype/dtype_kernel.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dtype/dtype_kernel.py#L55)。

<a name="api-tla-int64"></a>

## tla.int64

签名：`tla.int64`

返回：Triton dtype 对象。

**调用示例**

```python
@tla.jit
def example():
    tile = tla.alloc((64,), tla.int64)
```

仓库测试：[tla_ut/dtype/dtype_kernel.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dtype/dtype_kernel.py#L56)。

<a name="api-tla-uint16"></a>

## tla.uint16

签名：`tla.uint16`

返回：Triton dtype 对象。

**调用示例**

```python
@tla.jit
def example():
    tile = tla.alloc((64,), tla.uint16)
```

仓库测试：[tla_ut/dtype/dtype_kernel.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dtype/dtype_kernel.py#L58)。

<a name="api-tla-uint32"></a>

## tla.uint32

签名：`tla.uint32`

返回：Triton dtype 对象。

**调用示例**

```python
@tla.jit
def example():
    tile = tla.alloc((64,), tla.uint32)
```

仓库测试：[tla_ut/dtype/dtype_kernel.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dtype/dtype_kernel.py#L59)。

<a name="api-tla-uint8"></a>

## tla.uint8

签名：`tla.uint8`

返回：Triton dtype 对象。

**调用示例**

```python
@tla.jit
def example():
    tile = tla.alloc((64,), tla.uint8)
```

仓库测试：[tla_ut/dtype/dtype_kernel.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dtype/dtype_kernel.py#L57)。

<a name="api-tla-float16"></a>

## tla.float16

签名：`tla.float16`

返回：Triton dtype 对象。

**调用示例**

```python
@tla.jit
def example():
    tile = tla.alloc((64,), tla.float16)
```

仓库测试：[tla_ut/dtype/dtype_kernel.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dtype/dtype_kernel.py#L51)。

<a name="api-tla-int8"></a>

## tla.int8

签名：`tla.int8`

返回：Triton dtype 对象。

**调用示例**

```python
@tla.jit
def example():
    tile = tla.alloc((64,), tla.int8)
```

仓库测试：[tla_ut/dtype/dtype_kernel.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dtype/dtype_kernel.py#L53)。

<a name="api-tla-int16"></a>

## tla.int16

签名：`tla.int16`

返回：Triton dtype 对象。

**调用示例**

```python
@tla.jit
def example():
    tile = tla.alloc((64,), tla.int16)
```

仓库测试：[tla_ut/dtype/dtype_kernel.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dtype/dtype_kernel.py#L54)。

<a name="api-tla-uint64"></a>

## tla.uint64

签名：`tla.uint64`

返回：Triton dtype 对象。

**调用示例**

```python
@tla.jit
def example():
    tile = tla.alloc((64,), tla.uint64)
```

仓库测试：[tla_ut/dtype/dtype_kernel.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dtype/dtype_kernel.py#L60)。

## 范围说明

以下条目在当前整理表中没有可定位的 TLA 接口名。

| 原序号 | 范围结论 | 涉及场景 |
| --- | --- | --- |
| 220 | 暂不支持 | T.StridedTensor 可以优先级低一些 |
| 221 | 暂不支持 | T.Tensor 不一定需要 |
| 222 | 暂不支持 | T.annotate_layout 使用语义化的接口nd2nz，不用标注layout |
