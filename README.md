# 🎮 cuDF Game Analytics Benchmark — 什麼時候 GPU 才真的比較快？

用 **RAPIDS `cudf.pandas`** 在 2,000 萬筆合成遊戲事件上實測 GPU 加速 —— 並誠實面對一個反直覺的結果:**不是所有分析都吃 GPU**。這個 repo 的價值不在「GPU 快幾倍」,而在**釐清 GPU 加速的邊界,並示範如何把不吃 GPU 的運算改寫成吃 GPU 的版本**。

*An honest benchmark of RAPIDS `cudf.pandas` on 20M game events. The interesting finding: for typical RFM/retention/funnel code, GPU does NOT win — because those steps rely on operations (`qcut`, `rank`, `np.select`, Python `set`) that fall back to CPU. This repo shows why, and demonstrates rewriting the losing "funnel" into a GPU-native version that flips the result.*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjfishjj/cudf-game-analytics-benchmark/blob/main/notebooks/game_analytics_benchmark.ipynb)

## 實測結果 1:天真的 pandas 分析(Colab，同機 CPU vs T4 GPU)

| 步驟 | pandas CPU | cudf.pandas GPU | 結果 |
|------|:---:|:---:|:---:|
| RFM 分群 (20M rows) | **0.39 s** | 0.65 s | GPU 慢 1.7× ❌ |
| 留存 cohort | **0.42 s** | 0.69 s | GPU 慢 1.6× ❌ |
| 付費漏斗 (Python set) | 4.11 s | **3.63 s** | ≈ 平手 |

**GPU 沒有贏。** 為什麼?

- **RFM** 用了 `pd.qcut`、`.rank(method="first")`、`np.select` —— cuDF 無原生 GPU 實作,`cudf.pandas` 自動 **fallback 回 CPU**,還多付了資料在顯存↔記憶體來回搬的開銷。
- **留存** 的 `nunique` 雖支援 GPU,但輸出只有 ~90 個群組,**計算量太小,吃不回 GPU 的啟動開銷**(kernel launch + 顯存搬移)。
- **漏斗** 用 Python `set` 交集,`.unique()` 結果被搬回 host 變 Python 物件,GPU 完全使不上力。

> **核心洞察**:GPU 只在「**大資料量 × 原生向量化運算**」時才贏。天真移植的 pandas 分析常常兩個條件都不滿足。

## 實測結果 2:改寫成 GPU 原生運算後翻盤

把上面「輸掉的」運算改寫成 cuDF 原生支援的路徑(groupby / merge / sort,避開 fallback),並放大資料量 —— GPU 才展現真正實力。詳見 notebook 第 5、6 節(GPU-native 漏斗改寫 + 50M 列的 sort/merge/multi-agg groupby)。

> 這一組數字由你在 Colab 實跑後由 notebook 自動填入並畫圖。預期:大型 sort/join/groupby 上 GPU 明顯領先(數倍～十數倍)。

## 為什麼這是更好的作品(而不是藏起失敗)

一個只會說「GPU 快 50 倍」的候選人,不如一個能說「我實測發現 `qcut`/`rank` 會觸發 CPU fallback、20M 列不夠攤平 GPU 開銷,所以我把運算改寫成 groupby/merge 讓它回到 GPU 路徑」的候選人。**後者才證明真的懂 GPU 加速的原理與邊界。**

## How to run

Google Colab（免費 T4 GPU）:
1. 點上方 Colab badge → Runtime → Change runtime type → **T4 GPU**
2. `USE_GPU = False` 跑一次(Run all)→ 記下數字
3. Runtime → Restart session → `USE_GPU = True` 再跑一次
4. 把兩趟數字填入第 6 節的 `cpu` / `gpu` dict,自動產生對比圖並存成 `benchmark_result.png`

## Key insight（PM 視角）

GPU 加速的價值不是無腦「省時間」,而是**在對的運算上把日報表級分析變成即時互動查詢**。但導入前必須判斷:(1) 資料量是否夠大攤平開銷,(2) 運算是否走原生 GPU 路徑而非 fallback。這個判斷能力,比背誦「GPU 很快」對數據團隊更值錢。

## Stack

RAPIDS cuDF (`cudf.pandas`) · pandas · NumPy · Matplotlib · Google Colab T4

*Author: Jerome Kuo — gaming data PM (GA/RFM/retention at Gamania & Imperium), NVIDIA DLI certified (深度學習基礎理論與實踐).*
