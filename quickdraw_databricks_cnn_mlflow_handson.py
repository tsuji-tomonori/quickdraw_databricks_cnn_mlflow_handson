# Databricks notebook source
# MAGIC %md
# MAGIC # Databricksハンズオン: Quick, Draw! `.npy` 10クラスCNN分類
# MAGIC
# MAGIC このノートブックでは、Google Creative Lab の **Quick, Draw! Dataset** のうち、28×28グレースケール画像として公開されている `.npy` 形式を使い、10クラスの落書き分類CNNをDatabricks上で学習・検証します。
# MAGIC
# MAGIC
# MAGIC ## データソース
# MAGIC
# MAGIC - 公式リポジトリ: https://github.com/googlecreativelab/quickdraw-dataset
# MAGIC - `.npy` 取得元: `https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/<class>.npy`
# MAGIC - ライセンス: Creative Commons Attribution 4.0 International
# MAGIC - 注意: 公式READMEにも記載がある通り、データはmoderation済みですが不適切な内容が含まれる可能性があります。
# MAGIC
# MAGIC ## このハンズオンで触るDatabricks機能
# MAGIC
# MAGIC - **Widgets**: クラス、サンプル数、エポック数などをUIから変更する
# MAGIC - **DBFS / Unity Catalog Volume**: データと実験成果物を保存する
# MAGIC - **Spark / Delta Lake**: データセットメタデータと学習履歴をDelta形式で保存・表示する
# MAGIC - **display**: DataFrameや画像をノートブック上で可視化する
# MAGIC - **MLflow Tracking**: パラメータ、メトリクス、成果物、PyTorchモデルを記録する
# MAGIC - **MLflow Model Registry**: 任意で学習済みモデルを登録する
# MAGIC
# MAGIC ## 前提
# MAGIC
# MAGIC 推奨クラスタは **Databricks Runtime for Machine Learning** です。CPUでも動きますが、GPUクラスタではより速く学習できます。クラスタから `storage.googleapis.com` へHTTPSアクセスできる必要があります。
# MAGIC
# MAGIC このノートブックの初期設定では、10クラス × 12,000枚を使い、80%を学習、20%を検証に使います。より短いハンズオンにしたい場合は `samples_per_class` や `epochs` を小さくしてください。

# COMMAND ----------
# Databricks Runtime for Machine Learning では通常、このセルの追加インストールは不要です。
# 標準Runtimeなどで torch が入っていない場合のみ、次の2行のコメントを外して実行してください。
# %pip install -q torch pandas matplotlib mlflow
# dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. ライブラリの読み込みと実行環境の確認
# MAGIC
# MAGIC まず、PyTorch、MLflow、NumPy、Pandas、Matplotlibを読み込みます。Databricksでは `spark` と `dbutils` があらかじめ利用できます。

# COMMAND ----------
import copy
import json
import math
import os
import random
import shutil
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature

print(f"PyTorch: {torch.__version__}")
print(f"MLflow : {mlflow.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Databricks widgetsでハンズオン用パラメータを作る
# MAGIC
# MAGIC Widgetsを使うと、ノートブックを編集せずに実験条件を変えられます。Run Allを何度も試す前提で、既存widgetがあれば値を保持し、なければ作成する実装にしています。

# COMMAND ----------
def ensure_text_widget(name: str, default: str, label: str) -> None:
    try:
        dbutils.widgets.get(name)
    except Exception:
        dbutils.widgets.text(name, default, label)


def ensure_dropdown_widget(name: str, default: str, choices: list[str], label: str) -> None:
    try:
        dbutils.widgets.get(name)
    except Exception:
        dbutils.widgets.dropdown(name, default, choices, label)


ensure_text_widget(
    "classes_csv",
    "airplane,apple,banana,bicycle,car,cat,dog,house,tree,umbrella",
    "10 classes, comma separated",
)
ensure_text_widget("storage_root", "dbfs:/tmp/quickdraw_cnn_handson", "DBFS or UC Volume root")
ensure_text_widget("samples_per_class", "12000", "samples per class used in this handson")
ensure_text_widget("val_ratio", "0.20", "validation ratio")
ensure_text_widget("epochs", "5", "epochs")
ensure_text_widget("batch_size", "256", "batch size")
ensure_text_widget("learning_rate", "0.001", "learning rate")
ensure_text_widget("seed", "42", "random seed")
ensure_text_widget("num_workers", "0", "PyTorch DataLoader workers")
ensure_dropdown_widget("use_gpu", "auto", ["auto", "cpu", "cuda"], "device selection")
ensure_text_widget(
    "registered_model_name",
    "",
    "optional MLflow registered model name, e.g. catalog.schema.quickdraw_cnn",
)

print("Widgets are ready. Change values from the widget UI if needed, then re-run from this cell onward.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. パラメータを読み取り、10クラスであることを検証する
# MAGIC
# MAGIC Quick, Draw! には多数のカテゴリがありますが、このハンズオンでは10クラスに固定します。`classes_csv` を変える場合も、必ず10クラスにしてください。

# COMMAND ----------
classes_csv = dbutils.widgets.get("classes_csv")
CLASS_NAMES = [x.strip() for x in classes_csv.split(",") if x.strip()]

assert len(CLASS_NAMES) == 10, f"このハンズオンでは10クラスにしてください。現在: {len(CLASS_NAMES)} classes"
assert len(set(CLASS_NAMES)) == len(CLASS_NAMES), "クラス名が重複しています。"

STORAGE_ROOT = dbutils.widgets.get("storage_root").rstrip("/")
SAMPLES_PER_CLASS = int(dbutils.widgets.get("samples_per_class"))
VAL_RATIO = float(dbutils.widgets.get("val_ratio"))
EPOCHS = int(dbutils.widgets.get("epochs"))
BATCH_SIZE = int(dbutils.widgets.get("batch_size"))
LEARNING_RATE = float(dbutils.widgets.get("learning_rate"))
SEED = int(dbutils.widgets.get("seed"))
NUM_WORKERS = int(dbutils.widgets.get("num_workers"))
USE_GPU = dbutils.widgets.get("use_gpu")
REGISTERED_MODEL_NAME = dbutils.widgets.get("registered_model_name").strip()

assert 0.05 <= VAL_RATIO <= 0.50, "val_ratioは0.05〜0.50程度にしてください。"
assert SAMPLES_PER_CLASS >= 1000, "学習が不安定になるため、samples_per_classは1000以上を推奨します。"
assert EPOCHS >= 1
assert BATCH_SIZE >= 16
assert NUM_WORKERS >= 0

params_pdf = pd.DataFrame(
    [
        ("classes", ", ".join(CLASS_NAMES)),
        ("storage_root", STORAGE_ROOT),
        ("samples_per_class", SAMPLES_PER_CLASS),
        ("val_ratio", VAL_RATIO),
        ("epochs", EPOCHS),
        ("batch_size", BATCH_SIZE),
        ("learning_rate", LEARNING_RATE),
        ("seed", SEED),
        ("num_workers", NUM_WORKERS),
        ("use_gpu", USE_GPU),
        ("registered_model_name", REGISTERED_MODEL_NAME or "<skip>"),
    ],
    columns=["parameter", "value"],
)

display(spark.createDataFrame(params_pdf.astype(str)))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. 保存先とドライバローカルキャッシュを準備する
# MAGIC
# MAGIC Databricksでは永続化先としてDBFSまたはUnity Catalog Volumeを使い、学習時の読み込み速度を上げるためにドライバのローカルディスク `/local_disk0` をキャッシュとして使います。
# MAGIC
# MAGIC - `STORAGE_ROOT`: ダウンロード済みデータ、Delta、MLflow成果物補助ファイルの永続化先
# MAGIC - `DRIVER_CACHE_ROOT`: PyTorchのmemmap読み込み用ローカルキャッシュ

# COMMAND ----------
def dbfs_uri_to_local_path(uri: str) -> str:
    """dbfs:/... を /dbfs/... に変換する。/Volumes/... はそのまま返す。"""
    if uri.startswith("dbfs:/"):
        return "/dbfs/" + uri[len("dbfs:/"):].lstrip("/")
    if uri.startswith("file:/"):
        return uri[len("file:"):]
    return uri


def local_path_to_spark_path(path: str) -> str:
    """/dbfs/... を Spark が扱いやすい dbfs:/... に戻す。"""
    if path.startswith("/dbfs/"):
        return "dbfs:/" + path[len("/dbfs/"):].lstrip("/")
    return path


# 永続化先を作成
try:
    dbutils.fs.mkdirs(STORAGE_ROOT)
except Exception:
    os.makedirs(dbfs_uri_to_local_path(STORAGE_ROOT), exist_ok=True)

STORAGE_ROOT_LOCAL = dbfs_uri_to_local_path(STORAGE_ROOT)
PERSISTENT_NPY_DIR = os.path.join(STORAGE_ROOT_LOCAL, "raw_npy")
os.makedirs(PERSISTENT_NPY_DIR, exist_ok=True)

# /local_disk0 が使える場合は学習時のキャッシュに利用
if os.path.isdir("/local_disk0"):
    DRIVER_CACHE_ROOT = "/local_disk0/quickdraw_cnn_handson"
else:
    DRIVER_CACHE_ROOT = os.path.join(STORAGE_ROOT_LOCAL, "_driver_cache")

DRIVER_NPY_DIR = os.path.join(DRIVER_CACHE_ROOT, "raw_npy")
RUN_OUTPUT_DIR = os.path.join(STORAGE_ROOT_LOCAL, "run_outputs")
for p in [DRIVER_NPY_DIR, RUN_OUTPUT_DIR]:
    os.makedirs(p, exist_ok=True)

print("Persistent data dir:", PERSISTENT_NPY_DIR)
print("Driver cache dir   :", DRIVER_NPY_DIR)
print("Run output dir     :", RUN_OUTPUT_DIR)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Quick, Draw! `.npy` をダウンロードする
# MAGIC
# MAGIC `.npy` はカテゴリごとに分かれており、各行が1枚の28×28画像を784次元にflattenした配列です。ここでは選んだ10クラス分だけダウンロードします。
# MAGIC
# MAGIC 初回だけ時間がかかります。2回目以降はDBFS/Volumeまたはドライバキャッシュにファイルがあれば再利用します。

# COMMAND ----------
QUICKDRAW_NPY_BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"


def safe_filename_for_class(class_name: str) -> str:
    return class_name.replace("/", "_").replace(" ", "_") + ".npy"


def npy_url_for_class(class_name: str) -> str:
    # クラス名に空白が含まれる場合にも対応できるようURLエンコードする
    return f"{QUICKDRAW_NPY_BASE_URL}/{quote(class_name, safe='')}.npy"


def is_valid_quickdraw_npy(path: str) -> bool:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        arr = np.load(path, mmap_mode="r")
        return arr.ndim == 2 and arr.shape[1] == 784
    except Exception:
        return False


def download_to_path(url: str, dest_path: str) -> None:
    tmp_path = dest_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as f:
        shutil.copyfileobj(response, f, length=1024 * 1024)
    os.replace(tmp_path, dest_path)


metadata_records = []
CLASS_TO_NPY_PATH = {}

for class_name in CLASS_NAMES:
    file_name = safe_filename_for_class(class_name)
    persistent_path = os.path.join(PERSISTENT_NPY_DIR, file_name)
    driver_path = os.path.join(DRIVER_NPY_DIR, file_name)
    url = npy_url_for_class(class_name)

    if is_valid_quickdraw_npy(driver_path):
        source = "driver_cache"
    elif is_valid_quickdraw_npy(persistent_path):
        shutil.copy2(persistent_path, driver_path)
        source = "persistent_cache_copied_to_driver"
    else:
        print(f"Downloading {class_name}: {url}")
        download_to_path(url, driver_path)
        if not is_valid_quickdraw_npy(driver_path):
            raise RuntimeError(f"Downloaded file is not a valid QuickDraw npy: {driver_path}")
        shutil.copy2(driver_path, persistent_path)
        source = "downloaded"

    arr = np.load(driver_path, mmap_mode="r")
    CLASS_TO_NPY_PATH[class_name] = driver_path
    metadata_records.append(
        {
            "class_name": class_name,
            "label_id": CLASS_NAMES.index(class_name),
            "rows_available": int(arr.shape[0]),
            "npy_shape": str(tuple(arr.shape)),
            "file_size_mb": round(os.path.getsize(driver_path) / 1024 / 1024, 2),
            "source": source,
            "driver_path": driver_path,
            "persistent_path": persistent_path,
            "url": url,
        }
    )

metadata_pdf = pd.DataFrame(metadata_records)
metadata_sdf = spark.createDataFrame(metadata_pdf)
display(metadata_sdf.select("label_id", "class_name", "rows_available", "file_size_mb", "source"))

metadata_delta_path = f"{STORAGE_ROOT}/delta/class_metadata"
metadata_sdf.write.format("delta").mode("overwrite").save(metadata_delta_path)
metadata_sdf.createOrReplaceTempView("quickdraw_class_metadata")
print("Saved metadata as Delta:", metadata_delta_path)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. サンプル画像を確認する
# MAGIC
# MAGIC Quick, Draw! の `.npy` は28×28の小さなビットマップです。モデルに入れる前に、各クラスから1枚ずつ確認します。

# COMMAND ----------
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.flatten()

for ax, class_name in zip(axes, CLASS_NAMES):
    arr = np.load(CLASS_TO_NPY_PATH[class_name], mmap_mode="r")
    img = arr[0].reshape(28, 28)
    ax.imshow(img, cmap="gray")
    ax.set_title(class_name)
    ax.axis("off")

plt.tight_layout()
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. 学習データと検証データを分割する
# MAGIC
# MAGIC ここでは各クラスから `samples_per_class` 枚をランダムに選び、`val_ratio` 分を検証データ、残りを学習データにします。分割結果もDeltaに保存します。
# MAGIC
# MAGIC 全データを使うとかなり大きくなるため、ハンズオンではサンプリングして短時間で回せる構成にしています。

# COMMAND ----------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

rng = np.random.default_rng(SEED)
LABEL_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_LABEL = {i: name for name, i in LABEL_TO_ID.items()}

train_indices_by_class: dict[str, np.ndarray] = {}
val_indices_by_class: dict[str, np.ndarray] = {}
split_records = []

for class_name in CLASS_NAMES:
    arr = np.load(CLASS_TO_NPY_PATH[class_name], mmap_mode="r")
    rows_available = int(arr.shape[0])
    n_total = min(SAMPLES_PER_CLASS, rows_available)
    n_val = max(1, int(round(n_total * VAL_RATIO)))
    n_train = n_total - n_val

    indices = rng.choice(rows_available, size=n_total, replace=False)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    train_indices_by_class[class_name] = train_indices.astype(np.int64)
    val_indices_by_class[class_name] = val_indices.astype(np.int64)

    split_records.append(
        {
            "class_name": class_name,
            "label_id": LABEL_TO_ID[class_name],
            "rows_available": rows_available,
            "used_total": int(n_total),
            "train_rows": int(n_train),
            "validation_rows": int(n_val),
        }
    )

split_pdf = pd.DataFrame(split_records)
split_sdf = spark.createDataFrame(split_pdf)
display(split_sdf)

split_delta_path = f"{STORAGE_ROOT}/delta/split_summary"
split_sdf.write.format("delta").mode("overwrite").save(split_delta_path)
print("Saved split summary as Delta:", split_delta_path)
print("Total train rows     :", int(split_pdf["train_rows"].sum()))
print("Total validation rows:", int(split_pdf["validation_rows"].sum()))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. PyTorch Dataset / DataLoaderを作る
# MAGIC
# MAGIC `.npy` ファイルは大きいので、`np.load(..., mmap_mode="r")` でメモリマップとして読みます。これにより、全画像を一度にメモリへ載せず、必要な行だけを読み出せます。

# COMMAND ----------
class QuickDrawBitmapDataset(Dataset):
    """QuickDraw numpy_bitmap .npy を読むPyTorch Dataset。"""

    def __init__(
        self,
        class_to_path: dict[str, str],
        class_to_indices: dict[str, np.ndarray],
        label_to_id: dict[str, int],
    ):
        self.class_to_path = dict(class_to_path)
        self.label_to_id = dict(label_to_id)
        self.samples: list[tuple[str, int, int]] = []
        for class_name, indices in class_to_indices.items():
            label = self.label_to_id[class_name]
            self.samples.extend((class_name, int(i), label) for i in indices)
        self._mmap_cache = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getstate__(self):
        # DataLoader workersにpickleされるとき、memmap自体は渡さずworker側で再オープンする
        state = self.__dict__.copy()
        state["_mmap_cache"] = None
        return state

    def _get_mmaps(self) -> dict[str, np.ndarray]:
        if self._mmap_cache is None:
            self._mmap_cache = {
                class_name: np.load(path, mmap_mode="r")
                for class_name, path in self.class_to_path.items()
            }
        return self._mmap_cache

    def __getitem__(self, idx: int):
        class_name, row_idx, label = self.samples[idx]
        arr = self._get_mmaps()[class_name][row_idx]
        image = arr.astype(np.float32).reshape(1, 28, 28) / 255.0
        return torch.from_numpy(image), torch.tensor(label, dtype=torch.long)


train_ds = QuickDrawBitmapDataset(CLASS_TO_NPY_PATH, train_indices_by_class, LABEL_TO_ID)
val_ds = QuickDrawBitmapDataset(CLASS_TO_NPY_PATH, val_indices_by_class, LABEL_TO_ID)

# deviceを決める
if USE_GPU == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
elif USE_GPU == "cuda":
    if not torch.cuda.is_available():
        raise RuntimeError("use_gpu=cuda が指定されていますが、CUDAが利用できません。")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

loader_kwargs = {
    "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "pin_memory": device.type == "cuda",
}
if NUM_WORKERS > 0:
    loader_kwargs["persistent_workers"] = True
    loader_kwargs["prefetch_factor"] = 2

train_loader = DataLoader(
    train_ds,
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED),
    **loader_kwargs,
)
val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

x_batch, y_batch = next(iter(train_loader))
print("device:", device)
print("train dataset:", len(train_ds))
print("validation dataset:", len(val_ds))
print("batch image shape:", tuple(x_batch.shape))
print("batch label shape:", tuple(y_batch.shape))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9. CNNモデルを定義する
# MAGIC
# MAGIC 28×28の単一チャネル画像を入力し、10クラスのlogitを出力します。構造は小さめですが、BatchNorm、MaxPool、Dropoutを入れて、ハンズオンでも学習の改善が見えやすい構成にしています。

# COMMAND ----------
class QuickDrawCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14 -> 7
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = QuickDrawCNN(num_classes=len(CLASS_NAMES)).to(device)
num_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(model)
print(f"parameters: {num_params:,}")
print(f"trainable : {trainable_params:,}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 10. MLflow ExperimentをDatabricks上に作成する
# MAGIC
# MAGIC DatabricksのMLflow Tracking UIから、エポックごとのloss/accuracy、成果物、学習済みモデルを確認できます。Experimentは現在のユーザー配下に作成します。

# COMMAND ----------
try:
    current_user = spark.sql("SELECT current_user()").first()[0]
    experiment_name = f"/Users/{current_user}/quickdraw_cnn_handson"
except Exception:
    experiment_name = "quickdraw_cnn_handson"

mlflow.set_experiment(experiment_name)
print("MLflow experiment:", experiment_name)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 11. 学習・評価関数を定義する
# MAGIC
# MAGIC MLflowへはエポックごとに以下を記録します。
# MAGIC
# MAGIC - `train_loss`, `train_accuracy`
# MAGIC - `val_loss`, `val_accuracy`
# MAGIC - `learning_rate`
# MAGIC
# MAGIC さらに、ベストモデルのcheckpoint、学習履歴CSV、PyTorchモデル本体、混同行列、予測サンプル画像をartifactとして保存します。

# COMMAND ----------
def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> int:
    preds = logits.argmax(dim=1)
    return int((preds == labels).sum().item())


def train_one_epoch(model, loader, optimizer, criterion, device, use_amp: bool):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_count = 0

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        running_correct += accuracy_from_logits(logits, labels)
        running_count += batch_size

    return {
        "loss": running_loss / running_count,
        "accuracy": running_correct / running_count,
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_predictions: bool = False):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_count = 0
    all_true = []
    all_pred = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        running_correct += int((preds == labels).sum().item())
        running_count += batch_size

        if return_predictions:
            all_true.append(labels.cpu().numpy())
            all_pred.append(preds.cpu().numpy())

    metrics = {
        "loss": running_loss / running_count,
        "accuracy": running_correct / running_count,
    }
    if return_predictions:
        return metrics, np.concatenate(all_true), np.concatenate(all_pred)
    return metrics

# COMMAND ----------
# MAGIC %md
# MAGIC ## 12. 実際に学習する
# MAGIC
# MAGIC このセルでCNNを学習します。実行中はノートブック出力にも進捗が出ますが、主な可視化はMLflow Tracking UIで確認します。
# MAGIC
# MAGIC Databricks画面右側または上部の **Experiment / MLflow** からrunを開くと、各epochのメトリクス推移、artifact、モデルを確認できます。

# COMMAND ----------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
use_amp = device.type == "cuda"

run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
run_name = f"quickdraw-cnn-{run_timestamp}"
checkpoint_path = os.path.join(RUN_OUTPUT_DIR, f"best_model_{run_timestamp}.pt")
history_csv_path = os.path.join(RUN_OUTPUT_DIR, f"history_{run_timestamp}.csv")

best_val_accuracy = -1.0
best_epoch = -1
history = []

with mlflow.start_run(run_name=run_name) as run:
    RUN_ID = run.info.run_id
    mlflow.set_tags(
        {
            "project": "quickdraw-cnn-databricks-handson",
            "framework": "pytorch",
            "data_format": "quickdraw-numpy-bitmap-npy",
        }
    )
    mlflow.log_params(
        {
            "classes_csv": ",".join(CLASS_NAMES),
            "num_classes": len(CLASS_NAMES),
            "samples_per_class": SAMPLES_PER_CLASS,
            "val_ratio": VAL_RATIO,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "optimizer": "AdamW",
            "weight_decay": 1e-4,
            "scheduler": "CosineAnnealingLR",
            "model": "QuickDrawCNN",
            "num_params": num_params,
            "trainable_params": trainable_params,
            "device": str(device),
            "seed": SEED,
        }
    )
    mlflow.log_dict(
        {"label_to_id": LABEL_TO_ID, "id_to_label": ID_TO_LABEL},
        "metadata/class_mapping.json",
    )
    mlflow.log_dict(
        {
            "storage_root": STORAGE_ROOT,
            "metadata_delta_path": metadata_delta_path,
            "split_delta_path": split_delta_path,
        },
        "metadata/databricks_paths.json",
    )

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, use_amp)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed_sec = time.time() - start_time
        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "learning_rate": lr,
            "elapsed_sec": elapsed_sec,
        }
        history.append(row)

        mlflow.log_metrics(
            {
                "train_loss": row["train_loss"],
                "train_accuracy": row["train_accuracy"],
                "val_loss": row["val_loss"],
                "val_accuracy": row["val_accuracy"],
                "learning_rate": row["learning_rate"],
                "epoch_elapsed_sec": row["elapsed_sec"],
            },
            step=epoch,
        )

        if row["val_accuracy"] > best_val_accuracy:
            best_val_accuracy = row["val_accuracy"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "label_to_id": LABEL_TO_ID,
                    "id_to_label": ID_TO_LABEL,
                    "class_names": CLASS_NAMES,
                    "epoch": epoch,
                    "val_accuracy": best_val_accuracy,
                    "params": {
                        "samples_per_class": SAMPLES_PER_CLASS,
                        "val_ratio": VAL_RATIO,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "seed": SEED,
                    },
                },
                checkpoint_path,
            )

        print(
            f"epoch={epoch:02d}/{EPOCHS} "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_accuracy']:.4f} "
            f"lr={lr:.6f} elapsed={elapsed_sec:.1f}s"
        )

    # ベストモデルをロードしてからMLflowへ保存
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    history_pdf = pd.DataFrame(history)
    history_pdf.to_csv(history_csv_path, index=False)
    mlflow.log_artifact(history_csv_path, artifact_path="training")
    mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
    mlflow.log_metrics(
        {
            "best_val_accuracy": best_val_accuracy,
            "best_epoch": float(best_epoch),
        }
    )

    # Deltaにも学習履歴を保存する
    history_sdf = spark.createDataFrame(history_pdf)
    history_delta_path = f"{STORAGE_ROOT}/delta/training_history/{RUN_ID}"
    history_sdf.write.format("delta").mode("overwrite").save(history_delta_path)
    mlflow.log_dict({"history_delta_path": history_delta_path}, "metadata/history_delta_path.json")

    # PyTorchモデルをMLflow Modelとして保存する
    model_cpu = copy.deepcopy(model).cpu().eval()
    input_example = np.zeros((1, 1, 28, 28), dtype=np.float32)
    with torch.no_grad():
        output_example = model_cpu(torch.from_numpy(input_example)).numpy()
    signature = infer_signature(input_example, output_example)
    mlflow.pytorch.log_model(
        pytorch_model=model_cpu,
        artifact_path="model",
        input_example=input_example,
        signature=signature,
    )

print("Run ID:", RUN_ID)
print("Best epoch:", best_epoch)
print("Best validation accuracy:", best_val_accuracy)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 13. 学習履歴をDatabricks上で確認する
# MAGIC
# MAGIC 同じ値はMLflow UIでも見られますが、ここではDeltaに保存した学習履歴をSparkで読み、ノートブック上でもグラフ化します。

# COMMAND ----------
history_sdf = spark.read.format("delta").load(history_delta_path)
display(history_sdf)

history_pd = history_sdf.orderBy("epoch").toPandas()

plt.figure(figsize=(8, 5))
plt.plot(history_pd["epoch"], history_pd["train_loss"], marker="o", label="train_loss")
plt.plot(history_pd["epoch"], history_pd["val_loss"], marker="o", label="val_loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Training / validation loss")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history_pd["epoch"], history_pd["train_accuracy"], marker="o", label="train_accuracy")
plt.plot(history_pd["epoch"], history_pd["val_accuracy"], marker="o", label="val_accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("Training / validation accuracy")
plt.legend()
plt.grid(True)
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 14. 検証データで最終評価し、混同行列を作る
# MAGIC
# MAGIC ここでは学習には使っていない検証データで予測し、クラスごとのPrecision、Recall、F1、混同行列を確認します。結果はMLflow artifactにも保存します。

# COMMAND ----------
final_metrics, y_true, y_pred = evaluate(model, val_loader, criterion, device, return_predictions=True)
print("Final validation metrics:", final_metrics)

num_classes = len(CLASS_NAMES)
cm = np.zeros((num_classes, num_classes), dtype=np.int64)
for t, p in zip(y_true, y_pred):
    cm[int(t), int(p)] += 1

metric_rows = []
for i, class_name in enumerate(CLASS_NAMES):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    support = cm[i, :].sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    metric_rows.append(
        {
            "label_id": i,
            "class_name": class_name,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }
    )

class_metrics_pdf = pd.DataFrame(metric_rows)
class_metrics_sdf = spark.createDataFrame(class_metrics_pdf)
display(class_metrics_sdf)

class_metrics_path = os.path.join(RUN_OUTPUT_DIR, f"class_metrics_{RUN_ID}.csv")
class_metrics_pdf.to_csv(class_metrics_path, index=False)

cm_normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(cm_normalized)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(num_classes))
ax.set_yticks(range(num_classes))
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("Normalized confusion matrix")

for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j, i, f"{cm_normalized[i, j]:.2f}", ha="center", va="center", fontsize=8)

plt.tight_layout()

with mlflow.start_run(run_id=RUN_ID):
    mlflow.log_metric("final_val_loss", final_metrics["loss"])
    mlflow.log_metric("final_val_accuracy", final_metrics["accuracy"])
    mlflow.log_artifact(class_metrics_path, artifact_path="evaluation")
    mlflow.log_figure(fig, "evaluation/confusion_matrix.png")

plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 15. 検証データの予測例を可視化する
# MAGIC
# MAGIC 正解ラベルと予測ラベルを並べて確認します。誤分類がある場合、どのクラスが紛らわしいかを人間の目でも確認できます。

# COMMAND ----------
model.eval()
images, labels = next(iter(val_loader))
images_device = images.to(device)

with torch.no_grad():
    logits = model(images_device)
    preds = logits.argmax(dim=1).cpu()

n_show = min(20, images.size(0))
fig, axes = plt.subplots(4, 5, figsize=(12, 10))
axes = axes.flatten()

for idx in range(n_show):
    ax = axes[idx]
    img = images[idx].squeeze(0).numpy()
    true_label = ID_TO_LABEL[int(labels[idx])]
    pred_label = ID_TO_LABEL[int(preds[idx])]
    mark = "✓" if true_label == pred_label else "✗"
    ax.imshow(img, cmap="gray")
    ax.set_title(f"{mark} true: {true_label}\npred: {pred_label}", fontsize=9)
    ax.axis("off")

for idx in range(n_show, len(axes)):
    axes[idx].axis("off")

plt.tight_layout()

with mlflow.start_run(run_id=RUN_ID):
    mlflow.log_figure(fig, "evaluation/sample_predictions.png")

plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 16. MLflowに保存したモデルを読み戻して推論する
# MAGIC
# MAGIC MLflow Modelとして保存したモデルは、同じrunから再ロードできます。Databricks上ではこのモデルをModel Registryへ登録し、ServingやJobにつなげることもできます。

# COMMAND ----------
model_uri = f"runs:/{RUN_ID}/model"
loaded_model = mlflow.pytorch.load_model(model_uri)
loaded_model.eval()

sample_image, sample_label = val_ds[0]
with torch.no_grad():
    logits = loaded_model(sample_image.unsqueeze(0))
    probabilities = torch.softmax(logits, dim=1).squeeze(0).numpy()
    pred_id = int(probabilities.argmax())

topk = min(5, len(CLASS_NAMES))
top_indices = probabilities.argsort()[::-1][:topk]
result_pdf = pd.DataFrame(
    [
        {
            "rank": rank + 1,
            "class_name": ID_TO_LABEL[int(i)],
            "probability": float(probabilities[i]),
        }
        for rank, i in enumerate(top_indices)
    ]
)

print("model_uri:", model_uri)
print("true label:", ID_TO_LABEL[int(sample_label)])
print("pred label:", ID_TO_LABEL[pred_id])
display(spark.createDataFrame(result_pdf))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 17. 任意: MLflow Model Registryへ登録する
# MAGIC
# MAGIC `registered_model_name` widgetに値を入れてからこのセルを実行すると、学習済みモデルをRegistryへ登録します。
# MAGIC
# MAGIC Unity Catalogを使っている場合は、例として `catalog.schema.quickdraw_cnn` のような3階層名を指定します。空欄のままなら何もしません。

# COMMAND ----------
if REGISTERED_MODEL_NAME:
    print("Registering model:", REGISTERED_MODEL_NAME)
    registration_result = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)
    print(registration_result)
else:
    print("registered_model_name が空欄のため、Model Registry登録はスキップしました。")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 18. ふりかえりと発展課題
# MAGIC
# MAGIC このノートブックでは、Databricks上で次の流れを一通り実行しました。
# MAGIC
# MAGIC 1. 10クラスをwidgetsで定義
# MAGIC 2. Quick, Draw! `.npy` をDBFS/Volumeへ保存し、ドライバローカルへキャッシュ
# MAGIC 3. Spark DataFrameとDeltaでデータセットメタデータ・分割情報・学習履歴を管理
# MAGIC 4. PyTorch Dataset/DataLoaderでmemmap読み込み
# MAGIC 5. CNNを学習し、検証データで評価
# MAGIC 6. MLflow Trackingへパラメータ、メトリクス、artifact、PyTorchモデルを記録
# MAGIC 7. 任意でModel Registryへ登録
# MAGIC
# MAGIC ### 発展課題
# MAGIC
# MAGIC - `samples_per_class` を増やして精度と学習時間の変化を見る
# MAGIC - `classes_csv` を別の10クラスに変更する
# MAGIC - CNNの層数、Dropout、learning rateを変えてMLflowで比較する
# MAGIC - Databricks Jobsで定期実行する
# MAGIC - Unity Catalog Volumeを `storage_root` に指定し、チーム共有しやすい形にする
# MAGIC - Model RegistryからDatabricks Model Servingへつなげる
