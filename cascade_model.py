import argparse
import json
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


try:
    import lightgbm as lgb
except ImportError as exc:
    raise SystemExit("Missing dependency: lightgbm. Install requirements.txt.") from exc

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.layers import Dense, Dropout, LSTM
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.optimizers import Adam
except ImportError as exc:
    raise SystemExit("Missing dependency: tensorflow. Install requirements.txt.") from exc


COL_MAPPING = {
    "觀測時間(hour)": "datetime",
    "測站氣壓(hPa)": "PS01",
    "海平面氣壓(hPa)": "PS02",
    "氣溫(℃)": "TX01",
    "氣溫(°C)": "TX01",
    "露點溫度(℃)": "TD01",
    "露點溫度(°C)": "TD01",
    "相對溼度(%)": "RH01",
    "相對濕度(%)": "RH01",
    "風速(m/s)": "WD01",
    "風向(360degree)": "WD02",
    "最大陣風(m/s)": "WD07",
    "最大陣風風向(360degree)": "WD08",
    "降水量(mm)": "PP01",
    "降雨量(mm)": "PP01",
    "降水時數(hr)": "PP02",
    "降雨時數(hr)": "PP02",
    "全天空日射量(MJ/㎡)": "GR01",
    "天全空日射量(MJ/㎡)": "GR01",
    "全天空日射量(MJ/m2)": "GR01",
    "datetime": "datetime",
    "PS01": "PS01",
    "PS02": "PS02",
    "TX01": "TX01",
    "TD01": "TD01",
    "RH01": "RH01",
    "WD01": "WD01",
    "WD02": "WD02",
    "WD07": "WD07",
    "WD08": "WD08",
    "PP01": "PP01",
    "PP02": "PP02",
    "GR01": "GR01",
    "TxSoil0cm": "TxSoil0cm",
    "TxSoil10cm": "TxSoil10cm",
    "TxSoil20cm": "TxSoil20cm",
    "TxSoil50cm": "TxSoil50cm",
    "TxSoil100cm": "TxSoil100cm",
}
REQUIRED_COLUMNS = ["datetime", "PP01"]

REGRESSION_FEATURES = [
    "PS01", "PS02", "TX01", "TD01", "RH01", "WD01", "WD07", "PP01", "PP02", "GR01",
    "WindDirection.TenMinutelyMaximum", "WindSpeed.TenMinutelyMaximum",
    "WD02_sin", "WD02_cos", "WD08_sin", "WD08_cos", "year", "hour_sin", "hour_cos",
    "dayofyear_sin", "dayofyear_cos", "month_sin", "month_cos",
]

NON_SCALING_FEATURES = [
    "WD02_sin", "WD02_cos", "WD08_sin", "WD08_cos",
    "hour_sin", "hour_cos", "dayofyear_sin", "dayofyear_cos",
    "month_sin", "month_cos",
]


@dataclass
class TrainConfig:
    station: str
    prepared_dir: str = "prepared"
    outputs_dir: str = "outputs"
    train_start: str = "2009-01-01 00:00:00"
    train_end: str = "2020-12-31 23:00:00"
    test_start: str = "2021-01-01 00:00:00"
    test_end: str = "2025-12-31 23:00:00"
    rain_threshold: float = 0.5
    moderate_threshold: float = 10.0
    heavy_threshold: float = 40.0
    classifier_lag_hours: int = 3
    max_undersample_models: int = 8
    time_step: int = 4
    lstm_units: int = 64
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 150
    patience: int = 12
    run_name: str = "cascade_mse"
    seed: int = 42


class DualLGBMClassifier:
    """Two-classifier rain/no-rain gate used by the cascade pipeline.

    Classifier 1 is a weighted LightGBM binary classifier.
    Classifier 2 is an under-sampled LightGBM ensemble.
    The final rain probability is max(p1, p2), making the gate recall-friendly.
    """

    def __init__(
        self,
        n_estimators=500,
        learning_rate=0.07,
        max_depth=7,
        subsample=0.72,
        colsample_bytree=0.66,
        num_leaves=60,
        min_child_samples=20,
        max_undersample_models=8,
        random_state=42,
    ):
        self.params = {
            "objective": "binary",
            "metric": "auc",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
        self.max_undersample_models = max_undersample_models
        self.random_state = random_state
        self.classifier1 = None
        self.classifier2_ensemble = []
        self.best_n_classifier2 = 0
        self.threshold = 0.5

    @staticmethod
    def _scale_pos_weight(y):
        pos = int(np.sum(y))
        neg = int(len(y) - pos)
        return 1.0 if pos == 0 else neg / pos

    def fit(self, X, y):
        y = pd.Series(y)
        params = dict(self.params)
        params["scale_pos_weight"] = self._scale_pos_weight(y)
        self.classifier1 = lgb.LGBMClassifier(**params)
        self.classifier1.fit(X, y)

        X_inner, X_val, y_inner, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        pos_idx = y_inner[y_inner == 1].index.to_numpy()
        neg_idx = y_inner[y_inner == 0].index.to_numpy()
        rng = np.random.default_rng(self.random_state)
        n_models = min(
            self.max_undersample_models,
            max(1, len(neg_idx) // max(1, len(pos_idx))),
        )

        self.classifier2_ensemble = []
        for _ in range(n_models):
            sampled_neg = rng.choice(neg_idx, size=len(pos_idx), replace=False)
            sampled_idx = rng.permutation(np.concatenate([pos_idx, sampled_neg]))
            model = lgb.LGBMClassifier(**self.params)
            model.fit(X_inner.loc[sampled_idx], y_inner.loc[sampled_idx])
            self.classifier2_ensemble.append(model)

        best_score = -1.0
        best_n = 1
        best_threshold = 0.5
        for n in range(1, len(self.classifier2_ensemble) + 1):
            proba = self._ensemble_proba(X_val, n)
            for threshold in np.arange(0.30, 0.71, 0.01):
                pred = (proba >= threshold).astype(int)
                actual = y_val.to_numpy()
                tp = np.sum((pred == 1) & (actual == 1))
                fp = np.sum((pred == 1) & (actual == 0))
                fn = np.sum((pred == 0) & (actual == 1))
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-12)
                if f1 > best_score:
                    best_score = f1
                    best_n = n
                    best_threshold = float(threshold)

        self.best_n_classifier2 = best_n
        self.threshold = best_threshold
        return self

    def _ensemble_proba(self, X, n=None):
        n = n or self.best_n_classifier2 or len(self.classifier2_ensemble)
        models = self.classifier2_ensemble[:n]
        if not models:
            return self.classifier1.predict_proba(X)[:, 1]
        return np.mean([model.predict_proba(X)[:, 1] for model in models], axis=0)

    def predict_proba(self, X):
        p1 = self.classifier1.predict_proba(X)[:, 1]
        p2 = self._ensemble_proba(X)
        return np.maximum(p1, p2)

    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(int)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def read_station_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={k: v for k, v in COL_MAPPING.items() if k in df.columns})
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def clean_data(df: pd.DataFrame, nan_threshold_ratio: float) -> pd.DataFrame:
    df = df.copy()

    if "PP01" in df.columns:
        df.loc[df["PP01"] == -9.8, "PP01"] = 0

    for col in df.columns:
        if col not in ["datetime", "PP01"] and pd.api.types.is_numeric_dtype(df[col]):
            df.loc[df[col] < 0, col] = np.nan

    if "PP02" in df.columns:
        missing_duration = df["PP02"].isna()
        df.loc[missing_duration & (df["PP01"] == 0), "PP02"] = 0
        df.loc[missing_duration & (df["PP01"] > 0), "PP02"] = 1

    df = df.dropna(subset=["PP01"]).sort_values("datetime").reset_index(drop=True)

    protected = {"datetime", "PP01"}
    nan_ratio = df.isna().mean()
    drop_cols = [
        col for col, ratio in nan_ratio.items()
        if col not in protected and ratio > nan_threshold_ratio
    ]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


def add_features(df: pd.DataFrame, rain_threshold: float) -> pd.DataFrame:
    df = df.copy().sort_values("datetime").set_index("datetime")
    impute_features = [col for col in df.columns if col != "PP01"]

    for col in impute_features:
        if df[col].isna().any():
            df[col] = df[col].interpolate(method="time").bfill().ffill()

    if "RH01" in df.columns:
        df["RH01"] = df["RH01"].clip(0, 100)

    df = df.reset_index()
    df["year"] = df["datetime"].dt.year.astype(int)
    df["month"] = df["datetime"].dt.month.astype(int)
    df["hour"] = df["datetime"].dt.hour.astype(int)
    df["dayofyear"] = df["datetime"].dt.dayofyear.astype(int)

    if "WD02" in df.columns:
        df["WD02_sin"] = np.sin(2 * np.pi * df["WD02"] / 360)
        df["WD02_cos"] = np.cos(2 * np.pi * df["WD02"] / 360)
    if "WD08" in df.columns:
        df["WD08_sin"] = np.sin(2 * np.pi * df["WD08"] / 360)
        df["WD08_cos"] = np.cos(2 * np.pi * df["WD08"] / 360)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["is_rain"] = (df["PP01"] > rain_threshold).astype(int)
    return df


def prepare_station(args) -> None:
    data_path = Path(args.data_dir) / f"{args.station}.csv"
    out_dir = Path(args.out_dir) / args.station
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_station_csv(data_path)
    df = clean_data(df, args.nan_threshold_ratio)
    df = add_features(df, args.rain_threshold)

    processed_path = out_dir / "processed.csv"
    df.to_csv(processed_path, index=False, encoding="utf-8-sig")

    metadata = {
        "station": args.station,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "rain_threshold": args.rain_threshold,
        "datetime_min": str(df["datetime"].min()),
        "datetime_max": str(df["datetime"].max()),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {processed_path}")


def load_processed(config: TrainConfig) -> pd.DataFrame:
    path = Path(config.prepared_dir) / config.station / "processed.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run prepare first. Missing: {path}")
    return pd.read_csv(path, parse_dates=["datetime"], encoding="utf-8-sig").sort_values("datetime")


def build_classifier_dataset(df: pd.DataFrame, lag_hours: int):
    exclude = {"stno", "datetime", "is_rain", "WD02", "WD08", "month", "hour", "dayofyear"}
    base_features = [col for col in df.columns if col not in exclude and col != "PP01_log"]
    X_temp = df[base_features].copy()

    time_features = {
        "year", "hour_sin", "hour_cos", "dayofyear_sin", "dayofyear_cos", "month_sin", "month_cos"
    }
    lag_features = [col for col in base_features if col not in time_features]
    for feature in lag_features:
        for lag in range(1, lag_hours + 1):
            X_temp[f"{feature}_lag{lag}"] = df[feature].shift(lag)

    y_target = df["is_rain"].shift(-1).rename("y_target")
    target_datetime = df["datetime"].shift(-1).rename("target_datetime")
    combined = pd.concat([X_temp, y_target, target_datetime], axis=1)
    combined = combined.iloc[lag_hours:-1].dropna().copy()
    X = combined.drop(columns=["y_target", "target_datetime"])
    y = combined["y_target"].astype(int)
    dt = pd.to_datetime(combined["target_datetime"])
    return X, y, dt


def present_features(df: pd.DataFrame, features: list[str]) -> list[str]:
    return [feature for feature in features if feature in df.columns]


def scale_for_rain_lstm(
    df: pd.DataFrame,
    train_mask: np.ndarray,
    rain_threshold: float,
    features: list[str],
):
    df = df.copy()
    scaling_features = [f for f in features if f not in NON_SCALING_FEATURES and f != "PP01"]
    train_rain_mask = train_mask & (df["PP01"].to_numpy() > rain_threshold)
    if not np.any(train_rain_mask):
        raise ValueError("No rain events in training data for LSTM scaler fitting.")

    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    pp01_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler.fit(df.loc[train_rain_mask, scaling_features])
    pp01_scaler.fit(df.loc[train_rain_mask, ["PP01"]])

    scaled_parts = {}
    scaled_other = feature_scaler.transform(df[scaling_features])
    for idx, feature in enumerate(scaling_features):
        scaled_parts[feature] = scaled_other[:, idx]

    scaled_pp01 = np.zeros(len(df), dtype=float)
    positive = df["PP01"].to_numpy() > 0
    scaled_pp01[positive] = pp01_scaler.transform(df.loc[positive, ["PP01"]]).ravel()
    scaled_parts["PP01"] = scaled_pp01

    for feature in features:
        if feature in NON_SCALING_FEATURES:
            scaled_parts[feature] = df[feature].to_numpy()

    scaled = pd.DataFrame(scaled_parts)[features].to_numpy(dtype=np.float32)
    return scaled, feature_scaler, pp01_scaler


def create_sequences(data: np.ndarray, target: np.ndarray, datetimes: pd.Series, time_step: int):
    X, y, dt = [], [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i + time_step])
        y.append(target[i + time_step])
        dt.append(datetimes.iloc[i + time_step])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), pd.to_datetime(dt)


def build_rain_lstm_model(input_shape, config: TrainConfig):
    model = Sequential(
        [
            LSTM(config.lstm_units, input_shape=input_shape),
            Dropout(config.dropout),
            Dense(1),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss="mse",
        metrics=["mae", "mse"],
    )
    return model


def make_pipeline_predictions(
    label: str,
    start,
    end,
    gate,
    X_clf,
    dt_clf,
    X_seq,
    dt_seq,
    actual_seq,
    lstm_model,
    pp01_scaler,
    batch_size: int,
) -> pd.DataFrame:
    split_start = pd.Timestamp(start)
    split_end = pd.Timestamp(end)
    gate_mask = (dt_clf >= split_start) & (dt_clf <= split_end)
    seq_mask = (dt_seq >= split_start) & (dt_seq <= split_end)

    gate_df = pd.DataFrame(
        {
            "datetime": dt_clf[gate_mask].to_numpy(),
            "has_rain_pred": gate.predict(X_clf.loc[gate_mask]),
        }
    )
    seq_df = pd.DataFrame(
        {
            "datetime": dt_seq[seq_mask].to_numpy(),
            "actual": actual_seq[seq_mask],
            "seq_index": np.arange(len(dt_seq))[seq_mask],
        }
    )
    pred_df = seq_df.merge(gate_df, on="datetime", how="inner")
    pred_df["split"] = label

    final_pred = np.zeros(len(pred_df), dtype=float)
    rain_rows = pred_df["has_rain_pred"].to_numpy(dtype=int) == 1
    if np.any(rain_rows):
        seq_indices = pred_df.loc[rain_rows, "seq_index"].to_numpy(dtype=int)
        scaled_pred = lstm_model.predict(X_seq[seq_indices], batch_size=batch_size, verbose=0)
        mm_pred = pp01_scaler.inverse_transform(scaled_pred).ravel()
        final_pred[rain_rows] = np.maximum(mm_pred, 0)

    pred_df["pred"] = final_pred
    pred_df["abs_error"] = np.abs(pred_df["actual"] - pred_df["pred"])
    return pred_df


def metrics_for(y_true, y_pred, moderate_threshold, heavy_threshold):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    nonzero_mask = np.abs(y_true) > 1e-8
    rain_mask = y_true > 0.5
    moderate_mask = y_true >= moderate_threshold
    heavy_mask = y_true >= heavy_threshold
    return {
        "overall_mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": (
            float(np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100.0)
            if np.any(nonzero_mask)
            else float("nan")
        ),
        "rain_only_mae": (
            float(mean_absolute_error(y_true[rain_mask], y_pred[rain_mask])) if np.any(rain_mask) else float("nan")
        ),
        "ge_10mm_mae": (
            float(mean_absolute_error(y_true[moderate_mask], y_pred[moderate_mask]))
            if np.any(moderate_mask)
            else float("nan")
        ),
        "ge_10mm_rmse": (
            float(np.sqrt(mean_squared_error(y_true[moderate_mask], y_pred[moderate_mask])))
            if np.any(moderate_mask)
            else float("nan")
        ),
        "ge_10mm_samples": int(np.sum(moderate_mask)),
        "heavy_mae_40mm": (
            float(mean_absolute_error(y_true[heavy_mask], y_pred[heavy_mask])) if np.any(heavy_mask) else float("nan")
        ),
        "heavy_rmse_40mm": (
            float(np.sqrt(mean_squared_error(y_true[heavy_mask], y_pred[heavy_mask]))) if np.any(heavy_mask) else float("nan")
        ),
        "heavy_samples_40mm": int(np.sum(heavy_mask)),
    }


def plot_prediction_series(df: pd.DataFrame, title: str, output_path: Path) -> None:
    if df.empty:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = df.sort_values("datetime")
    plt.figure(figsize=(26, 12))
    plt.plot(plot_df["datetime"], plot_df["actual"], label="Actual rainfall", color="#0057ff", linewidth=2.8)
    plt.plot(plot_df["datetime"], plot_df["pred"], label="Predicted rainfall", color="#e60000", linewidth=2.5)
    plt.axhline(0, color="#555555", linewidth=1.4, alpha=0.55)
    plt.title(title, fontsize=28, pad=18)
    plt.xlabel("Datetime", fontsize=22, labelpad=12)
    plt.ylabel("Rainfall (mm)", fontsize=26, labelpad=16)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=22)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close()


def history_metrics(history) -> dict:
    history_dict = getattr(history, "history", {}) or {}
    return {
        "loss_name": "mse",
        "train_mse_best": float(np.min(history_dict.get("loss", [np.nan]))),
        "train_mse_final": float(history_dict.get("loss", [np.nan])[-1]),
        "val_mse_best": float(np.min(history_dict.get("val_loss", [np.nan]))),
        "val_mse_final": float(history_dict.get("val_loss", [np.nan])[-1]),
        "epochs_ran": len(history_dict.get("loss", [])),
    }


def train_two_stage(args) -> None:
    config_fields = {field.name for field in fields(TrainConfig)}
    config = TrainConfig(**{key: value for key, value in vars(args).items() if key in config_fields})
    set_seed(config.seed)

    df = load_processed(config).reset_index(drop=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_run_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in config.run_name)
    experiment_dir = Path(config.outputs_dir) / config.station / f"{run_id}_{safe_run_name}"
    model_dir = experiment_dir / "models"
    pred_dir = experiment_dir / "predictions"
    plot_dir = experiment_dir / "plots"
    model_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    train_mask_df = (
        (df["datetime"] >= pd.Timestamp(config.train_start)) &
        (df["datetime"] <= pd.Timestamp(config.train_end))
    ).to_numpy()

    X_clf, y_clf, dt_clf = build_classifier_dataset(df, config.classifier_lag_hours)
    clf_train_mask = (
        (dt_clf >= pd.Timestamp(config.train_start)) &
        (dt_clf <= pd.Timestamp(config.train_end))
    )
    gate = DualLGBMClassifier(
        max_undersample_models=config.max_undersample_models,
        random_state=config.seed,
    )
    gate.fit(X_clf.loc[clf_train_mask], y_clf.loc[clf_train_mask])
    joblib.dump(gate, model_dir / "dual_lgbm_gate.joblib")

    features = present_features(df, REGRESSION_FEATURES)
    scaled, feature_scaler, pp01_scaler = scale_for_rain_lstm(
        df, train_mask_df, config.rain_threshold, features
    )
    joblib.dump(feature_scaler, model_dir / "feature_scaler.joblib")
    joblib.dump(pp01_scaler, model_dir / "pp01_scaler.joblib")

    X_seq, y_scaled_seq, dt_seq = create_sequences(
        scaled, scaled[:, features.index("PP01")], df["datetime"], config.time_step
    )
    actual_seq = df["PP01"].iloc[config.time_step:].to_numpy(dtype=float)
    seq_train_mask = (
        (dt_seq >= pd.Timestamp(config.train_start)) &
        (dt_seq <= pd.Timestamp(config.train_end)) &
        (actual_seq > config.rain_threshold)
    )
    if not np.any(seq_train_mask):
        raise ValueError("No LSTM rain-event training sequences were created.")

    model_path = model_dir / "rain_lstm_mse.keras"
    lstm_model = build_rain_lstm_model((X_seq.shape[1], X_seq.shape[2]), config)
    callbacks = [
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, mode="min", verbose=0),
        EarlyStopping(monitor="val_loss", patience=config.patience, restore_best_weights=True, mode="min", verbose=1),
    ]
    history = lstm_model.fit(
        X_seq[seq_train_mask],
        y_scaled_seq[seq_train_mask],
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        shuffle=False,
        verbose=1,
    )
    if model_path.exists():
        lstm_model = load_model(model_path, compile=False)

    train_predictions = make_pipeline_predictions(
        "train",
        config.train_start,
        config.train_end,
        gate,
        X_clf,
        dt_clf,
        X_seq,
        dt_seq,
        actual_seq,
        lstm_model,
        pp01_scaler,
        config.batch_size,
    )
    test_predictions = make_pipeline_predictions(
        "test",
        config.test_start,
        config.test_end,
        gate,
        X_clf,
        dt_clf,
        X_seq,
        dt_seq,
        actual_seq,
        lstm_model,
        pp01_scaler,
        config.batch_size,
    )

    train_predictions.to_csv(pred_dir / "train_predictions.csv", index=False, encoding="utf-8-sig")
    test_predictions.to_csv(pred_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    plot_prediction_series(
        train_predictions,
        f"{config.station} train rainfall prediction ({config.run_name})",
        plot_dir / "train_prediction.png",
    )
    plot_prediction_series(
        test_predictions,
        f"{config.station} test rainfall prediction ({config.run_name})",
        plot_dir / "test_prediction.png",
    )

    metrics = {
        "config": asdict(config),
        "architecture": {
            "stage_1": "Dual LightGBM rain/no-rain classifier gate",
            "stage_2": "Rain-event-only LSTM rainfall amount regressor",
            "lstm_loss": "mse",
            "final_rule": "No-rain gate output maps to 0 mm; rain gate output uses inverse-scaled LSTM prediction.",
        },
        "train_metrics": metrics_for(
            train_predictions["actual"],
            train_predictions["pred"],
            config.moderate_threshold,
            config.heavy_threshold,
        ),
        "test_metrics": metrics_for(
            test_predictions["actual"],
            test_predictions["pred"],
            config.moderate_threshold,
            config.heavy_threshold,
        ),
        "lstm_history": history_metrics(history),
    }
    (experiment_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote outputs in {experiment_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cascade LightGBM-LSTM rainfall pipeline: Dual LightGBM gate + rain-event LSTM with MSE loss."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Clean one CWA CoDiS station CSV.")
    prepare.add_argument("--station", required=True)
    prepare.add_argument("--data-dir", default="data")
    prepare.add_argument("--out-dir", default="prepared")
    prepare.add_argument("--rain-threshold", type=float, default=0.5)
    prepare.add_argument("--nan-threshold-ratio", type=float, default=0.20)
    prepare.set_defaults(func=prepare_station)

    train = subparsers.add_parser("train", help="Train the cascade two-stage model.")
    train.add_argument("--station", required=True)
    train.add_argument("--prepared-dir", default="prepared")
    train.add_argument("--outputs-dir", default="outputs")
    train.add_argument("--train-start", default="2009-01-01 00:00:00")
    train.add_argument("--train-end", default="2020-12-31 23:00:00")
    train.add_argument("--test-start", default="2021-01-01 00:00:00")
    train.add_argument("--test-end", default="2025-12-31 23:00:00")
    train.add_argument("--rain-threshold", type=float, default=0.5)
    train.add_argument("--moderate-threshold", type=float, default=10.0)
    train.add_argument("--heavy-threshold", type=float, default=40.0)
    train.add_argument("--classifier-lag-hours", type=int, default=3)
    train.add_argument("--max-undersample-models", type=int, default=8)
    train.add_argument("--time-step", type=int, default=4)
    train.add_argument("--lstm-units", type=int, default=64)
    train.add_argument("--dropout", type=float, default=0.2)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--epochs", type=int, default=150)
    train.add_argument("--patience", type=int, default=12)
    train.add_argument("--run-name", default="cascade_mse")
    train.add_argument("--seed", type=int, default=42)
    train.set_defaults(func=train_two_stage)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
