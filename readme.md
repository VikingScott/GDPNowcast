# GDP Nowcast System for All Weather Strategy

## 1. 项目概述
这是一个轻量级、工程化的美国名义 GDP (Real GDP) 临现预测 (Nowcast) 系统。
它的核心目标不是追求学术上的完美 $R^2$，而是**为全天候 (All Weather) 策略提供及时、稳健的宏观增长信号**。

**核心特性：**
* **Point-in-Time (Vintage) 仿真**：严格避免未来数据泄露，模拟历史每一天“当时”能看到的数据状态（Ragged Edge）。
* **抗黑天鹅设计**：通过 4-sigma Clipping 和 Robust Scaling，有效应对 2020 年疫情导致的极端数据冲击。
* **自动化流水线**：一键更新数据、训练模型、生成信号。
* **纯本地化**：基于本地 CSV 缓存运行，速度极快且不依赖实时 API 连接。

### 1.1 注意事项

* nowcast/features/asof_dataset.py：最容易引入未来函数，因为使用硬编码lag，需要精细化

---

## 2. 快速开始 (Quick Start)

### 2.1 环境配置
1.  安装依赖：
    ```bash
    pip install pandas numpy scikit-learn scipy fredapi python-dotenv tqdm matplotlib
    ```
2.  配置密钥：
    在根目录创建 `.env` 文件，填入你的 FRED API Key：
    ```env
    FRED_API_KEY=你的32位key
    ```

### 2.2 日常使用 (Daily Workflow)
每天只需要运行

scripts下的 `data_update.py`

根目录下的 `run_daily.py`：
```bash
python run_daily.py