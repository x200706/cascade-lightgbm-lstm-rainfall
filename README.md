# cascade-lightgbm-lstm-rainfall

Release code for a cascade LightGBM-LSTM rainfall prediction model.

The release implementation is in [cascade_model.py](cascade_model.py). It combines data preparation, model training, metric reporting, and prediction plots in a single script.

Traditional Chinese documentation is available in [README.zh-TW.md](README.zh-TW.md).

## Model

The cascade model has two stages:

1. Dual LightGBM rain/no-rain classification gate.
2. Rain-event-only LSTM rainfall amount regressor.

The LSTM loss is fixed to MSE. At inference time, the gate decides whether rainfall should be predicted:

- No rain: output `0 mm`.
- Rain: use the LSTM rainfall amount prediction, inverse-transform it to millimeters, and clip negative values to `0 mm`.

## Data

The raw station observations are not bundled as a directly reusable dataset. Download hourly station data, then place each station CSV under `data/`.

Recommended data sources:

- Central Weather Administration Climate Data Service (CoDiS): https://codis.cwa.gov.tw/
- Historical weather downloader developed by National Chung Hsing University: https://mycolab.pp.nchu.edu.tw/historical_weather/

Example:

```text
data/466930.csv
data/467080.csv
```

The loader accepts common CWA Chinese column names and maps them to canonical names such as:

```text
datetime, PP01, PP02, RH01, WD07, PS01, TX01, PS02, WD08, TD01, WD02, WD01, GR01
```

`datetime` and `PP01` are required. `PP01` is hourly precipitation amount.

## Usage

Prepare one station:

```bash
python cascade_model.py prepare --station 466930
```

Train the cascade model:

```bash
python cascade_model.py train --station 466930 --epochs 150 --run-name cascade_mse
```

## Outputs

Default outputs:

```text
prepared/<station>/processed.csv
outputs/<station>/<run_id>_<run_name>/
  metrics.json
  models/
  predictions/
    train_predictions.csv
    test_predictions.csv
  plots/
    train_prediction.png
    test_prediction.png
```

`metrics.json` reports train and test metrics, including overall MAE/RMSE and rainfall-event metrics such as `ge_10mm_mae`, `ge_10mm_rmse`, `rain_only_mae`, `heavy_mae_40mm`, and `heavy_rmse_40mm`.

## Poster Supplement

Supplementary acknowledgements and references:

https://x200706.github.io/cascade-lightgbm-lstm-rainfall/
