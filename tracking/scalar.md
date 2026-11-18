# 标量评审跟踪

[返回接口目录](../README.md)

<a name="scalar-version-scope"></a>

本轮核对 TA `074cf62907f815694a01e5cdc4ac78a90610ae82` 的前端、builder、adapter、共享标量语义及专用用例；`tla.abs` 使用开发提交 `33819a5b6a9bb503e46da2a6e5962f0cd0c44393`。

参考版本的 [scalar.py](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/scalar.py#L29) 未统一拒绝 SIMD VF 内的标量调用，也无后续版本的 FP64 数值窄化拒绝检查。

<a name="scalar-001"></a>

## SCALAR-001 · tla.set_atomic

[查看规格](../spec/scalar.md#api-tla-set-atomic)

- 需明确模式的持续范围、dtype 匹配、切换模式的同步及复位要求。

依据：[公开入口](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/dma.py#L900)、[native builder](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/cpp/tla_builder.cc#L1592) 及[专用用例](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/dma/set_atomic.py#L26)。用例在原子写回后显式同步并用 NONE 复位，再检查普通写回。

结论：已修订。模式影响后续相关 GM 写回，scope 不自动恢复状态，dtype 不作转换，NONE 仍须合法 dtype。示例补齐 UB 准备、同步与复位；关联 store 已按 [DMA-003](dma.md#dma-003) 改为两参数调用。

<a name="scalar-002"></a>

## SCALAR-002 · tla.select

[查看规格](../spec/scalar.md#api-tla-select)

- 需明确两分支提前求值的语义，并定位专用写回用例。

依据：[公开入口](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/scalar.py#L43) 复用 `tl.where`；[select_scalar](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scalar/select.py#L35) 覆盖专用写回路径。

结论：已修订。不能用 select 保护越界访存或副作用，未选分支仍须合法；返回类型按两分支提升，并改用专用用例链接。

<a name="scalar-003"></a>

## SCALAR-003 · tla.abs

[查看规格](../spec/scalar.md#api-tla-abs)

- 需限定为标量接口，补齐整数、浮点特殊值语义及专用测试链接。

依据：开发提交的 [abs 入口](https://github.com/triton-lang/triton-ascend/blob/33819a5b6a9bb503e46da2a6e5962f0cd0c44393/third_party/ascend/language/tla/scalar.py#L59) 与 [abs.py](https://github.com/triton-lang/triton-ascend/blob/33819a5b6a9bb503e46da2a6e5962f0cd0c44393/third_party/ascend/unittest/tla_ut/scalar/abs.py#L36)。

结论：已修订。运行时返回标量且保持 dtype；bool 和无符号整数不变，有符号最小整数不饱和、不扩宽，浮点结果清除符号位且不保证 NaN payload。补齐拒绝类型和专用用例链接。

<a name="scalar-005"></a>

## SCALAR-005 · tla.cast

[查看规格](../spec/scalar.md#api-tla-cast)

- 需修正误链至 cdiv.py 的测试链接，并明确返回类型。

依据：[cast 专用 kernel](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scalar/cast.py#L40) 直接调用 tla.cast，并断言结果 dtype 等于目标 dtype。

结论：已修订。链接改为 cast.py，数值转换和等宽 bitcast 均返回目标 dtype 的标量；后续 FP64 窄化拒绝检查不追溯到参考提交。

<a name="scalar-007"></a>

## SCALAR-007 · tla.clamp

[查看规格](../spec/scalar.md#api-tla-clamp)

- 需明确三输入的联合浮点约束、NaN 策略及上下界条件。

依据：[TritonSemantic.clamp](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/python/triton/language/semantic.py#L391) 在类型提升后拒绝非浮点计算；[前端负例](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scalar/scalar_frontend_contract.py#L775) 覆盖整数和 bool 组合；[公开入口](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/scalar.py#L175) 说明 NaN 策略和上下界约束。

结论：已修订。纯整数或 bool 组合拒绝，混合输入须形成浮点计算，原始 FP16/BF16 输入仍拒绝。NONE 下 NaN x 暂不支持，ALL 只传播 x 的 NaN；NaN 上下界或 min > max 为未定义行为，由调用者保证运行时条件。示例标注 float32 输入。

<a name="scalar-012"></a>

## SCALAR-012 · tla.maximum

[查看规格](../spec/scalar.md#api-tla-maximum)

- 需确认双 NaN 的目标结果与支持范围。

依据：[入口契约](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/scalar.py#L58) 限定 NONE 下单 NaN 暂不支持、ALL 传播任一输入的 NaN；[设备用例](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scalar/maximum.py#L115) 仍排除双 NaN，并留有目标结果的 TODO。

结论：已修订 NaN 策略及原始 FP16/BF16 操作数的拒绝约束；双 NaN 的目标支持待决。

<a name="scalar-014"></a>

## SCALAR-014 · tla.minimum

[查看规格](../spec/scalar.md#api-tla-minimum)

- 需确认双 NaN 的目标结果与支持范围。

依据：[入口契约](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/language/tla/scalar.py#L74) 限定 NONE 下单 NaN 暂不支持、ALL 传播任一输入的 NaN；[设备用例](https://github.com/triton-lang/triton-ascend/blob/074cf62907f815694a01e5cdc4ac78a90610ae82/third_party/ascend/unittest/tla_ut/scalar/minimum.py#L115) 仍排除双 NaN，并留有目标结果的 TODO。

结论：已修订 NaN 策略及原始 FP16/BF16 操作数的拒绝约束；双 NaN 的目标支持待决。
