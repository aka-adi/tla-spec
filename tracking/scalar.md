# 标量评审跟踪

[返回接口目录](../README.md)

<a name="scalar-009"></a>

## SCALAR-009 · tla.exp

[查看规格](../spec/scalar.md#api-tla-exp)

已记录开发分支的 FP32 标量实现。本轮按[转测范围](../spec/scalar.md#scalar-unary-test-scope)验证功能。

结论：本轮范围已限定。标量展开应落在哪一层及性能验收待评审。

<a name="scalar-011"></a>

## SCALAR-011 · tla.log1p

[查看规格](../spec/scalar.md#api-tla-log1p)

已记录开发分支的 FP32 标量实现。本轮按[转测范围](../spec/scalar.md#scalar-unary-test-scope)验证功能，覆盖零附近精度、零的符号和 −1 附近的定义域边界。

结论：本轮范围已限定。标量展开应落在哪一层及性能验收待评审。

<a name="scalar-015"></a>

## SCALAR-015 · tla.sigmoid

[查看规格](../spec/scalar.md#api-tla-sigmoid)

已记录开发分支的 FP32 标量实现。本轮按[转测范围](../spec/scalar.md#scalar-unary-test-scope)验证功能。

结论：本轮范围已限定。随标量指数路径评审 lowering 归属和性能。
