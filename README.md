# QuickDraw Databricks CNN MLflow Handson

このZIPには、Databricksにインポートして実行できるQuick, Draw! CNN分類ハンズオン用ノートブックが含まれています。

## ファイル

- `quickdraw_databricks_cnn_mlflow_handson.py`: Databricks source形式。Databricks Workspaceで **Import > File** から取り込めます。
- `quickdraw_databricks_cnn_mlflow_handson.ipynb`: Jupyter notebook形式。Databricksにもインポートできます。

## 推奨実行環境

- Databricks Runtime for Machine Learning
- CPUでも実行可能、GPU推奨
- クラスタから `storage.googleapis.com` へアクセス可能であること

## 実行方法

1. Databricks WorkspaceでノートブックをImportします。
2. ML RuntimeクラスタへAttachします。
3. まずWidgetsの値を確認します。初期設定は10クラス、各12,000枚、5 epochsです。
4. Run Allします。
5. 右側または上部のMLflow/Experiment UIで、メトリクス、artifact、モデルを確認します。

