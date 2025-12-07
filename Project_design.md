
# GDP Nowcast 项目技术文档（精简版）

## 1. 设计目标与原则

本项目旨在构建一套可维护、模块化、可测试的 GDP Nowcast 系统，用于：

* 利用日度宏观数据构建当前季度 GDP 的 nowcast；
* 通过可插拔模型（AR baseline 和 Bridge 模型）生成点预测与不确定度；
* 生成供上层策略（如 All Weather）使用的信号；
* 实现严格的 real-time/vintage 流程，避免未来信息泄露。

设计原则：

1. 数据、特征、模型、训练、信号模块相互隔离。
2. 通过统一接口（BaseNowcastModel、FeatureBuilder）实现模型可替换性。
3. 所有步骤均基于 vintage 宏观面板，严格避免时间穿越。
4. 所有功能模块都提供可独立测试的输入/输出接口。

---

## 2. 项目目录结构

```
gdp_nowcast/
  config/
    series.yaml
    bridge.yaml
    rolling.yaml
    signal.yaml

  data/
    raw/
    processed/
    results/

  src/
    data/
      series_config.py
      loaders.py
      calendar.py
      panel.py
      validators.py

    features/
      bridge_base.py
      bridge_stage.py

    models/
      base.py
      bridge.py
      ar_baseline.py

    training/
      bridge_dataset.py
      rolling_trainer.py

    engine/
      pipeline.py
      io.py

    signal/
      mapper.py
      history.py

  test/
    test_data_layer.py
    test_feature_layer.py
    test_model_ar.py
    test_model_bridge.py
    test_pipeline.py
```

---

## 3. 数据流（Data Flow Pipeline）

数据流顺序如下：

1. series.yaml → series_config 解析元数据
2. loaders 读取原始序列 → raw DataFrame
3. calendar 根据元数据生成发布日 → release table
4. panel 合并 release tables → MacroPanel（日频 ragged panel）
5. FeatureBuilder 将 MacroPanel 转换为 design_df（训练用）
6. Model.fit(design_df) → 模型训练
7. Model.predict_row(row) → NowcastResult
8. SignalMapper 将 NowcastResult 映射为增长信号和置信度
9. 结果写入 data/results/

---

## 4. 模块职责与接口说明

### 4.1 config/

存放四类配置文件：

* series.yaml：宏观序列元数据（频率、发布日期规则、滞后、变换方式）
* bridge.yaml：Bridge 模型使用的输入指标与阶段划分规则
* rolling.yaml：Rolling 训练参数（窗口长度、重估频率）
* signal.yaml：信号映射阈值与置信度参数

### 4.2 src/data/

#### series_config.py

解析 series.yaml，输出：

```
dict[str, SeriesMeta]
```

#### loaders.py

从本地或外部源读取序列。输出 DataFrame：

* index：ref_period（季度或月份）
* column：value

#### calendar.py

根据元数据计算每条观测的发布日，输出 DataFrame：

* series, ref_period, value, release_date

#### panel.py

构造 MacroPanel：

* data：日频、按发布日展开的 ragged DataFrame
* meta：序列元数据
  提供函数：

```
get_macro_panel_as_of(asof_date) -> MacroPanel
```

要求严格截断未来信息。

#### validators.py

检查数据的完整性、时间顺序与发布结构。

---

## 5. 特征层（src/features/）

特征构造模块将 MacroPanel 转换为模型训练用 design_df。

### BridgeFeatureBuilder（抽象类）

接口：

```
build_design_table(panel: MacroPanel) -> DataFrame
```

输出 design_df：

* index：origin_date（日频）
* columns：

  * target_quarter
  * stage
  * x_*（桥接模型所需特征）
  * y（季度 GDP 真值；训练时填充）

### StageAwareBridgeFeatureBuilder

根据 bridge.yaml 生成阶段型 Bridge 特征，包括：

* 按季度内可见月份构建 partial aggregators
* 生成 stage
* 对齐季度 GDP 以形成 y 列

---

## 6. 模型层（src/models/）

### BaseNowcastModel（抽象类）

必须实现：

```
fit(dataset)
predict_row(row) -> NowcastResult
```

### NowcastResult 数据结构

包含：

* origin_date
* target_quarter
* point（点预测）
* std（预测不确定度，可为空）
* stage
* info_coverage
* meta（附加信息）

### ARBaselineModel

只使用 GDP 的滞后项进行预测。可直接从 design_df 中读取 y 并构建滞后。

### BridgeNowcastModel

使用 design_df 中的 X_* 特征进行线性回归，可按 stage 训练多组子模型。

---

## 7. 训练层（src/training/）

### bridge_dataset.py

负责从 MacroPanel 中构建训练用 design_df（调用 FeatureBuilder）。

### RollingTrainer

执行 walk-forward 训练流程：

1. 根据 rolling.yaml 构造重估点
2. 对每个重估点：

   * 提取训练窗口
   * 训练模型（model.fit）
   * 对未来区间执行 model.predict_row
3. 输出包含 y_true 与 y_pred 的 OOS DataFrame

要求 rolling 必须严格基于真实 vintage 数据。

---

## 8. Engine（src/engine/）

### pipeline.py

核心职责：

* `fit_history(asof_date)`：
  加载数据 → 构建 MacroPanel → 构建 design_df → 训练模型

* `run_daily(asof_date)`：
  读取当日 MacroPanel → 构建单日特征 → 调用 model.predict_row → 调用 SignalMapper → 返回结果

接口：

```
run_daily(date) -> (NowcastResult, signal, confidence)
```

### io.py

负责读写模型预测结果与信号。

---

## 9. 信号层（src/signal/）

### mapper.py

将 NowcastResult 映射为：

* signal（-1 / 0 / +1）
* confidence（0–1）

映射规则来自 signal.yaml。

### history.py

存储信号历史记录。

---

## 10. 测试规范（test/ 文件夹）

测试内容不属于模型评估，而是代码层面的单元测试。

### 测试分类

1. 数据层测试（test_data_layer.py）

   * 测试 series_config, loaders, calendar, panel 的输入输出结构
   * 测试 get_macro_panel_as_of 的时间截断正确性
   * 测试 validators 的异常检测

2. 特征层测试（test_feature_layer.py）

   * design_df 是否包含 target_quarter、stage、x_*、y
   * 特征是否符合 bridge.yaml 的配置要求

3. 模型单元测试

   * AR 模型：fit + predict_row 能否正常运行
   * Bridge 模型：fit + predict_row 是否输出 NowcastResult 所需字段

4. Pipeline 测试（test_pipeline.py）

   * pipeline.fit_history 能否构建 MacroPanel 与模型
   * pipeline.run_daily 能否生成有效的 NowcastResult、signal、confidence

---

如需我进一步将其压缩成“最简洁的系统文档版本（比如给 LLM 作为 system prompt）”，我可以继续精简。
