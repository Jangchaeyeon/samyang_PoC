# Databricks notebook source
# MAGIC %md # Databricks ML Quickstart: 모델학습
# MAGIC 
# MAGIC 이 노트북은 Databricks에서 머신러닝 모델을 학습하기 위한 간단한 핸즈온 자료입니다. 모델을 학습하기 위해서는 Databricks ML Runtime에 이미 설치가 되어 있는 Scikit-learn과 같은 라이브러리를 사용할 수 있습니다. 추가로 MLflow를 사용하실 경우 학습된 모델을 기록하거나 Hyperopt와 SparkTrial을 사용한 하이퍼파라미터 튜닝 기능도 사용할 수 있습니다.
# MAGIC 
# MAGIC 이 핸즈온에서는 두 가지를 실습하게 됩니다:
# MAGIC - Part 1: MLflow 기록을 통해 간단한 분류모델 학습하기
# MAGIC - Part 2: 더 나은 성능 모델 학습을 위해 Hyperopt를 사용하여 하이퍼파라미터 튜닝하기
# MAGIC 
# MAGIC 모델 라이프사이클 관리나 모델 추론 등과 같이 다양한 프로덕션 레벨의 ML은 두 번째 ML End to End 실습에서 확인할 수 있습니다. ([AWS](https://docs.databricks.com/applications/mlflow/end-to-end-example.html)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/end-to-end-example)).
# MAGIC 
# MAGIC ### 요구사항
# MAGIC - Databricks Runtime 7.5 ML 이상을 사용하셔야 합니다.

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 라이브러리
# MAGIC 필요한 라이브러리를 import하실 수 있습니다. 아래 라이브러리들은 Databricks ML Runtime에 미리 설치되어 있고 ([AWS](https://docs.databricks.com/runtime/mlruntime.html)|[Azure](https://docs.microsoft.com/azure/databricks/runtime/mlruntime)) 클러스터는 이미 안정성과 성능을 최적화하기 위해 튜닝이 되어 있습니다.

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.ensemble

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 데이터 적재
# MAGIC 이번 실습에서는 다른 종류의 와인 샘플을 묘사하는 데이터를 사용하게 됩니다. 데이터셋은 UCI Machine Learning 리포지토리와 DBFS에 있습니다 [dataset](https://archive.ics.uci.edu/ml/datasets/Wine)([AWS](https://docs.databricks.com/data/databricks-file-system.html)|[Azure](https://docs.microsoft.com/azure/databricks/data/databricks-file-system)).
# MAGIC 이번 실습의 목표는 와인의 퀄리티에 따라 레드와인 또는 화이트화인을 분류하는 것입니다.
# MAGIC 
# MAGIC 데이터를 다른 소스로부터 업로딩하고 로딩하는 부분에 있어 더 많은 내용을 확인하고 싶으실 경우 ([AWS](https://docs.databricks.com/data/index.html)|[Azure](https://docs.microsoft.com/azure/databricks/data/index)) 에서 확인해주세요.

# COMMAND ----------

# 데이터 로딩 및 전처리
white_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-white.csv", sep=';')
red_wine = pd.read_csv("/dbfs/databricks-datasets/wine-quality/winequality-red.csv", sep=';')
white_wine['is_red'] = 0.0
red_wine['is_red'] = 1.0
data_df = pd.concat([white_wine, red_wine], axis=0)

# 와인 품질에 따른 레이블 분류 기준 생성
data_labels = data_df['quality'] >= 7
data_df = data_df.drop(['quality'], axis=1)

# 80 대 20 비율로 학습 및 테스트 데이터 분산
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
  data_df,
  data_labels,
  test_size=0.2,
  random_state=1
)

# COMMAND ----------

# MAGIC %md ## Part 1. 분류 모델 학습하기

# COMMAND ----------

# MAGIC %md ### MLflow 추적
# MAGIC [MLflow](https://www.mlflow.org/docs/latest/tracking.html)를 통해 머신러닝 학습용 코드, 파라미터, 그리고 코드를 관리/정리할 수 있습니다.
# MAGIC 
# MAGIC ML추적을 자동으로 사용하고자 할 경우 아래와 같이 [*autologging*](https://www.mlflow.org/docs/latest/tracking.html#automatic-logging)을 사용하시면 됩니다.

# COMMAND ----------

# 이 노트북에 MLflow autologging 기능 사용
mlflow.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC 다음으로 MLflow run에서 분류기를 학습할 수 있습니다. 아래와 같이 사용하실 경우 학습된 모델에 대한 로그를 남길 수 있고 그 외에 다양한 메트릭이나 파라미터를 자동으로 로깅할 수 있습니다.
# MAGIC 
# MAGIC 아래와 같이 테스트 데이터셋에 대한 AUC 스코어를 추가할 수도 있습니다.

# COMMAND ----------

with mlflow.start_run(run_name='gradient_boost') as run:
  model = sklearn.ensemble.GradientBoostingClassifier(random_state=0)
  
  # 모델, 파라미터, 그리고 학습 메트릭이 자동으로 추적
  model.fit(X_train, y_train)

  predicted_probs = model.predict_proba(X_test)
  roc_auc = sklearn.metrics.roc_auc_score(y_test, predicted_probs[:,1])
  
  # 테스트 데이터에 AUC 스코어가 자동으로 기록되지 않습니다. 아래와 같이 할 경우 수동으로 로깅할 수 있습니다.
  mlflow.log_metric("test_auc", roc_auc)
  print("Test AUC of: {}".format(roc_auc))

# COMMAND ----------

predicted_probs

# COMMAND ----------

# MAGIC %md
# MAGIC 만약 성능이 충분치 않다면 다른 하이퍼파라미터를 활용하여 다른 모델을 학습할 수 있습니다.

# COMMAND ----------

# 새로운 run 생성 및 추후 사용을 위해 run 이름 지정할 수 있습니다.
with mlflow.start_run(run_name='gradient_boost') as run:
  model_2 = sklearn.ensemble.GradientBoostingClassifier(
    random_state=0, 
    
    # n_estimators를 사용하여 새로운 파라미터 지정
    n_estimators=200,
  )
  model_2.fit(X_train, y_train)

  predicted_probs = model_2.predict_proba(X_test)
  roc_auc = sklearn.metrics.roc_auc_score(y_test, predicted_probs[:,1])
  mlflow.log_metric("test_auc", roc_auc)
  print("Test AUC of: {}".format(roc_auc))

# COMMAND ----------

# MAGIC %md ### MLflow 확인
# MAGIC 학습이 진행된 run을 확인하기 위해서는 우측 상단에 있는 **Experiment** 아이콘을 클릭해주세요. 필요하다면 refresh 아이콘을 클릭하여 가장 최신 run에 대한 결과를 확인하고 모니터링하실 수 있습니다.
# MAGIC 
# MAGIC <img width="350" src="https://docs.databricks.com/_static/images/mlflow/quickstart/experiment-sidebar-icons.png"/>
# MAGIC 
# MAGIC Experiment 페이지 아이콘을 클릭하실 경우 더욱 상세한 MLflow 실험 페이지를 확인하실 수 있습니다. ([AWS](https://docs.databricks.com/applications/mlflow/tracking.html#notebook-experiments)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/tracking#notebook-experiments)). 이 페이지는 다양한 run을 비교하고 특정한 run에 대한 상세한 정보 확인이 가능합니다.
# MAGIC 
# MAGIC <img width="800" src="https://docs.databricks.com/_static/images/mlflow/quickstart/compare-runs.png"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ### 모델 로딩
# MAGIC 특정한 run에 대한 결과는 MLflow API를 통해서도 확인하실 수 있습니다. 아래 셀의 코드는 MLflow run을 통해 학습된 모델을 어떻게 로딩하는지 그리고 예측에 사용되는지 보여줍니다. MLflow run 페이지에서 특정한 모델을 로딩하기 위한 코드 샘플은 ([AWS](https://docs.databricks.com/applications/mlflow/tracking.html#view-notebook-experiment)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/tracking#view-notebook-experiment))에 있습니다.

# COMMAND ----------

# 모델이 로깅된 이후에는 다른 노트북 또는 job에서도 로딩할 수 있습니다.
# mlflow.pyfunc.load_model의 방법을 사용할 경우 common API 아래 모델 예측을 할 수 있도록 지원합니다.
model_loaded = mlflow.pyfunc.load_model(
  'runs:/{run_id}/model'.format(
    run_id=run.info.run_id
  )
)

predictions_loaded = model_loaded.predict(X_test)
predictions_original = model_2.predict(X_test)

# 로딩된 모델은 기존 모델의 결과와 동일합니다.
assert(np.array_equal(predictions_loaded, predictions_original))

# COMMAND ----------

# MAGIC %md ## Part 2. 하이퍼파라미터 튜닝
# MAGIC 지금까지는 간단하게 모델을 학습해보았고 MLflow 추적을 통해 작업한 내용을 확인해봤습니다. 이번 세션에서는 Hyperopt라는 더 나은 튜닝 방법을 통해 어떻게 더 나은 성능을 낼 수 있는지 실습해볼 예정입니다.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hyperopt와 SparkTrials를 사용한 평행학습 (Parallel training)
# MAGIC [Hyperopt](http://hyperopt.github.io/hyperopt/)는 하이퍼파라미터 튜닝을 위한 파이썬 라이브러리입니다. Databricks에서 Hyperopt를 사용하는 더 많은 정보를 알기 원하신다면 ([AWS](https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/index.html#hyperparameter-tuning-with-hyperopt)|[Azure](https://docs.microsoft.com/azure/databricks/applications/machine-learning/automl-hyperparam-tuning/index#hyperparameter-tuning-with-hyperopt))를 참조해주세요.
# MAGIC 
# MAGIC Hyperopt와 SparkTrials를 함께 사용한다면 하이퍼파라미터 스윕 (Sweep)과 여러 모델을 동시에 활용할 수 있습니다. 이런 방법을 사용할 경우 모델 성능을 최적화하기 위한 시간을 줄일 수 있습니다. MLflow 추적은 Hyperopt와 연계되어 자동으로 모델과 파라미터를 로깅합니다.

# COMMAND ----------

# 탐색할 구간 (search space) 지정
search_space = {
  'n_estimators': scope.int(hp.quniform('n_estimators', 20, 1000, 1)),
  'learning_rate': hp.loguniform('learning_rate', -3, 0),
  'max_depth': scope.int(hp.quniform('max_depth', 2, 5, 1)),
}

def train_model(params):
  # 각각의 워커 노드에 autologging 지정
  mlflow.autolog()
  with mlflow.start_run(nested=True):
    model_hp = sklearn.ensemble.GradientBoostingClassifier(
      random_state=0,
      **params
    )
    model_hp.fit(X_train, y_train)
    predicted_probs = model_hp.predict_proba(X_test)
    # 테스트 데이터의 AUC 값을 통해 튜닝
    # 실제 production 환경에서는 별도의 valiation set을 사용할 수도 있습니다.
    roc_auc = sklearn.metrics.roc_auc_score(y_test, predicted_probs[:,1])
    mlflow.log_metric('test_auc', roc_auc)
    
    # Loss 값을 -1*auc_score로 지정하여 fmin 값이 auc_score를 최대로 가질 수 있게 합니다.
    return {'status': STATUS_OK, 'loss': -1*roc_auc}

# SparkTrials은 여러 Spark 워커 노드에 튜닝 값을 분산합니다.
# 더 많은 parallelism은 빠르게 처리할 수 있게 하지만 각각의 하이퍼파라미터 trial은 서로 다른 trial에 대한 정보를 적게 가지게 됩니다.
# 작은 규모의 클러스터나 Databricks Community Edition 에서는 parallelism=2를 사용하시기 바랍니다.
spark_trials = SparkTrials(
  parallelism=8
)

with mlflow.start_run(run_name='gb_hyperopt') as run:
  # Hyperopt를 통해 가장 높은 AUC 값을 나타내는 파라미터를 찾습니다.
  best_params = fmin(
    fn=train_model, 
    space=search_space, 
    algo=tpe.suggest, 
    max_evals=20,
    trials=spark_trials)

# COMMAND ----------

# MAGIC %md ### 최적의 모델 확인
# MAGIC MLflow에서는 모든 run이 기록되기 때문에 최적의 모델의 메트릭과 파라미터를 찾기 위해서는 MLflow search run API를 사용할 수 있고 이를 통해 가장 높은 테스트 AUC 값을 가진 튜닝된 run을 확인할 수 있습니다.
# MAGIC 
# MAGIC 이 튜닝된 모델은 Part1에서 만들어진 간단한 버전의 모델보다 더 나은 성능을 보일 것입니다.

# COMMAND ----------

# 테스트 AUC를 기준으로 정렬을 하고 동률이 있을 경우 가장 최신 run을 사용합니다.
best_run = mlflow.search_runs(
  order_by=['metrics.test_auc DESC', 'start_time DESC'],
  max_results=10,
).iloc[0]
print('Best Run')
print('AUC: {}'.format(best_run["metrics.test_auc"]))
print('Num Estimators: {}'.format(best_run["params.n_estimators"]))
print('Max Depth: {}'.format(best_run["params.max_depth"]))
print('Learning Rate: {}'.format(best_run["params.learning_rate"]))

best_model_pyfunc = mlflow.pyfunc.load_model(
  'runs:/{run_id}/model'.format(
    run_id=best_run.run_id
  )
)
best_model_predictions = best_model_pyfunc.predict(X_test[:5])
print("Test Predictions: {}".format(best_model_predictions))

# COMMAND ----------

# MAGIC %md ### UI 화면에서 다양한 run 비교하기
# MAGIC Part 1에서와 같이 MLflow experiment 상세 페이지에서 다양한 run을 비교하고 확인할 수 있으며 이는 상단의 **Experiment** 사이드 바에 있는 외부 링크를 통해 들어갈 수 있습니다.
# MAGIC 
# MAGIC Experiment 상세 페이지에서 "+" 아이콘을 클릭하여 parent run을 확장하고 parent를 제외한 나머지 run을 모두 선택한 후 **Compare**를 클릭하면 아래와 같이 시각화 할 수 있습니다. 다양한 run에 대한 시각화를 통해 어떤 파라미터가 metric에 어떻게 영향을 주는지도 확인할 수 있습니다.
# MAGIC 
# MAGIC <img width="800" src="https://docs.databricks.com/_static/images/mlflow/quickstart/parallel-plot.png"/>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>

# COMMAND ----------


