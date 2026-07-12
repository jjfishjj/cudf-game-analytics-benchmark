# 🎮 cuDF Game Analytics Benchmark — pandas vs GPU, zero code change

用 **RAPIDS `cudf.pandas`** 加速遊戲營運三大日常分析 — **RFM 分群、留存 Cohort、付費漏斗** — 在 2,000 萬筆合成遊戲事件上對比 CPU/GPU 執行時間，且**分析程式碼一行都不用改**。

*Benchmarking classic game-ops analytics (RFM segmentation, retention cohorts, purchase funnel) on 20M synthetic game events: stock pandas vs RAPIDS `cudf.pandas` — with zero code changes.*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjfishjj/cudf-game-analytics-benchmark/blob/main/notebooks/game_analytics_benchmark.ipynb)

## 為什麼做這個

我在 Gamania 與 Imperium 做遊戲數據分析（GA event tracking、RFM、留存、漏斗）。這些分析在千萬級事件量下，CPU pipeline 以分鐘計 — 分析師的迭代節奏被等待時間綁死。RAPIDS 的 `cudf.pandas` 宣稱零改碼 GPU 加速，這個 repo 用真實的遊戲營運分析邏輯來驗證它。

## 內容

- `notebooks/game_analytics_benchmark.ipynb` — 自含式 notebook：
  1. 產生 50 萬玩家、90 天、2,000 萬筆事件（冪律活躍度分佈，貼近真實）
  2. RFM 分群（whale / core / churning payer / at-risk / casual）
  3. D1/D7/D30 留存 cohort
  4. login → stage_clear → gacha → purchase 漏斗
  5. CPU vs GPU 計時對比表

## How to run

Google Colab（免費 T4 GPU）：
1. 點上方 Colab badge
2. Runtime → Change runtime type → **T4 GPU**
3. `USE_GPU = False` 跑一次（CPU baseline）→ 重啟 → `USE_GPU = True` 再跑一次
4. 對比各步驟計時

## Key insight（PM 視角）

GPU 加速對數據團隊的價值不是「省幾分鐘」，而是**把日報表級的分析變成即時互動查詢** — 營運會議上可以當場改分群條件重跑，決策迴圈從「明天給你」變成「現在就看」。`cudf.pandas` 零改碼的特性讓導入成本趨近於零：不需重寫 pipeline、不需重訓團隊。

## Stack

RAPIDS cuDF (`cudf.pandas`) · pandas · NumPy · Google Colab T4

*Author: Jerome Kuo — gaming data PM, NVIDIA DLI certified (深度學習基礎理論與實踐).*
