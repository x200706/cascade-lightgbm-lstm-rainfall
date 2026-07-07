# 海報版兩階段降雨預測流程

這個版本整理海報發表用的精簡版程式，只保留兩階段架構：

1. 第一階段：Dual LightGBM 雨／無雨分類閘門。
2. 第二階段：只用雨事件訓練的 LSTM 降雨量回歸模型。
3. LSTM loss 固定使用 MSE。若第一階段判定無雨，最終預測直接輸出 `0 mm`；若判定有雨，才交給 LSTM 預測降雨量。

主程式是 `cascade_model.py`，把原本 `prepare.py` 與 `train.py` 中海報需要的資料整理、雙分類器、雨事件 LSTM、MSE loss 與輸出流程合併成單一檔案。

## 資料來源

本程式不直接附上可重現的完整原始資料。使用者需自行下載「各氣象站逐時資料集」，再將單一測站 CSV 放到專案的 `data/` 目錄。可使用下列任一方式取得資料：

- 中央氣象署 Climate Data Service（CoDiS）：https://codis.cwa.gov.tw/
- 國立中興大學開發的歷史天氣資料下載器：https://mycolab.pp.nchu.edu.tw/historical_weather/

下載後的檔案可放成：

```text
data/466930.csv
data/467080.csv
```

CoDiS 下載的原始欄位名稱可能是中文欄位；程式會將目前專案中使用過的中央氣象署欄位對應到 canonical names，例如：

```text
datetime, PP01, PP02, RH01, WD07, PS01, TX01, PS02, WD08, TD01, WD02, WD01, GR01
```

其中 `PP01` 是逐時降雨量欄位，`datetime` 是觀測時間欄位，這兩個欄位是必要欄位。

## 執行方式

先整理單一測站資料：

```bash
python cascade_model.py prepare --station 466930
```

再訓練海報版兩階段模型：

```bash
python cascade_model.py train --station 466930 --epochs 150 --run-name cascade_mse
```

預設輸出位置：

```text
prepared/<station>/processed.csv
outputs/<station>/<run_id>_<run_name>/
  metrics.json
  models/
    dual_lgbm_gate.joblib
    feature_scaler.joblib
    pp01_scaler.joblib
    rain_lstm_mse.keras
  predictions/
    train_predictions.csv
    test_predictions.csv
  plots/
    train_prediction.png
    test_prediction.png
```

`metrics.json` 會直接列出訓練集與測試集評估指標，包括：

- `overall_mae`
- `rmse`
- `ge_10mm_mae`
- `ge_10mm_rmse`
- `ge_10mm_samples`
- `rain_only_mae`
- `heavy_mae_40mm`
- `heavy_rmse_40mm`

其中 `ge_10mm_*` 是針對真實降雨量大於等於 `10 mm` 的樣本計算，方便海報或口頭報告時呈現較明顯降雨事件的預測表現。

圖表也由同一支 `cascade_model.py` 直接輸出，不需要另外呼叫 `plot_results.py`：

- `train_prediction.png`：訓練集真實值與預測值時間序列。
- `test_prediction.png`：測試集真實值與預測值時間序列。

## 架構摘要

Dual LightGBM 分類閘門包含兩個分類器：

- Classifier 1：使用 `scale_pos_weight` 的 LightGBM binary classifier。
- Classifier 2：針對類別不平衡建立的 under-sampled LightGBM ensemble。

最終雨／無雨機率使用兩者機率的最大值，並以內部分割資料選擇 F1 較佳的 threshold。這個設計目標是降低雨事件被漏判的機率。

LSTM 回歸器只用訓練期間的雨事件序列訓練，loss 為：

```python
loss="mse"
```

預測時，整體 cascade 規則為：

- dual classifier 預測無雨：最終降雨量為 `0 mm`。
- dual classifier 預測有雨：用 LSTM 輸出並 inverse transform 回毫米，再將物理上不合理的負值 clip 到 `0 mm`。

## 海報補充資訊

海報補充頁面：

https://x200706.github.io/cascade-lightgbm-lstm-rainfall/

該頁主要收錄海報版面未完整列出的 references 與 acknowledgements。引用與致謝摘要如下。

### References

- Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting, 22*(4), 679-688. https://doi.org/10.1016/j.ijforecast.2006.03.001
- World Meteorological Organization. (2024). *Guide to instruments and methods of observation: Volume I - Measurement of meteorological variables* (WMO-No. 8). World Meteorological Organization.
- Chen, Y.-C. (2023). *Two XGBoost classifiers for imbalanced data classification* [Master's thesis, University of Taipei]. National Digital Library of Theses and Dissertations in Taiwan. https://hdl.handle.net/11296/nz43w7

補充頁也註明，降水強度標準參考 WMO-No. 8 中「Criteria for slight, moderate and heavy precipitation intensity」附錄。

### Acknowledgements

本研究感謝國家科學及技術委員會（NSTC）計畫「結合人工智慧方法建立豪雨誘發淹水預警系統（子計畫六）（I）」經費支持，並感謝中央氣象署（CWA）透過 Climate Data Service（CoDiS）提供氣象資料，以支援本研究資料分析。
