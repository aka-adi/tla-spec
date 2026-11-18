# 语言评审跟踪

[返回接口目录](../README.md)

本轮核对 TA `074cf62907f815694a01e5cdc4ac78a90610ae82` 的 Python 入口、native builder 及已有 IR 用例。

<a name="language-001"></a>

## LANGUAGE-001 · tla.sqrt

[查看规格](../spec/language.md#api-tla-sqrt)

- 需明确标量平方根的参数类型和编译表示。

依据：[公开入口](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/math.py#L79) 的标量分支及 [test_sqrt_ir](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/vector/vector_type.py#L136)。

结论：本轮仅支持 float32 标量，编译表示为 math.sqrt。已同步参数表和用例链接。

<a name="language-002"></a>

## LANGUAGE-002 · tla.symbolic

[查看规格](../spec/language.md#api-tla-symbolic)

原评审问题：是否纳入本轮规格并补齐实现。

结论：已关闭。确认不需要此接口，退出规格范围。
