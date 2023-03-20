import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from plotnine import *
from mizani.breaks import date_breaks

from feature_engine.creation import CyclicalFeatures
from feature_engine.transformation import LogCpTransformer
from feature_engine.datetime import DatetimeFeatures
from feature_engine.imputation import DropMissingData
from feature_engine.selection import DropFeatures
from feature_engine.timeseries.forecasting import (LagFeatures, WindowFeatures)
from feature_engine.encoding import OneHotEncoder
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import (mean_squared_error, 
                             r2_score, 
                             mean_absolute_error, 
                             mean_absolute_percentage_error
                            )

from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import optuna.integration.lightgbm as lgb
from optuna.samplers import TPESampler

from lightgbm import early_stopping, log_evaluation
import lightgbm as lgbm
import datetime

# LOAD DATASET ----

bitcoin_data = pd.read_csv("../data/bitcoin_prices.csv", parse_dates=["Datetime"])
bitcoin_data.head()

bitcoin_data.info()

# SELECT TARGET ----

bitcoin_adj_close = bitcoin_data[["Datetime", "Adj Close"]]

bitcoin_adj_close = bitcoin_adj_close.set_index("Datetime") \
                                     .asfreq("5Min") \
                                     .sort_index()

bitcoin_adj_close.plot()

# VALIDATION SPLIT ----

val_len = 5
in_sample_df = bitcoin_adj_close.iloc[:-val_len, :]
out_of_sample_df = bitcoin_adj_close.iloc[-val_len:, :]

in_sample_df = in_sample_df.reset_index()
out_of_sample_df = out_of_sample_df.reset_index()

# TRAIN TEST SPLIT ----

test_time = pd.Timedelta(25, unit = "minutes")
split_point = in_sample_df["Datetime"].max() - test_time

X_train = in_sample_df[in_sample_df['Datetime'] < split_point]
X_test = in_sample_df[in_sample_df['Datetime'] >= split_point - pd.Timedelta(35, unit = 'minutes')]

y_train = in_sample_df[in_sample_df['Datetime'] < split_point]
y_test = in_sample_df[in_sample_df['Datetime'] >= split_point - pd.Timedelta(35, unit='minutes')]

X_train = X_train.set_index('Datetime')
X_test = X_test.set_index('Datetime')
y_train = y_train.set_index('Datetime')
y_test = y_test.set_index('Datetime')

# FEATURE ENGINEERING ----

# Fourier Features Class
class AddFourierFeatures(BaseEstimator, TransformerMixin):
    seconds_per_day = 24*60*60     # Daily dataset
    seconds_per_hour = 60*60       # Hourly dataset

    def __init__(self, K, periods: list, by = "day"):
        self.K = K
        self.periods = periods
        self.by = by

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):
        X = X.copy()
        dates = X.index
        
        for period in self.periods:
            term = self.K / period
            timestamps = dates.map(datetime.datetime.timestamp)
            ts_scaled = []

            for ts in timestamps:
                if self.by == "day":
                    x_scaled = round(ts / self.seconds_per_day)
                    ts_scaled.append(x_scaled)
                else:
                    x_scaled = round(ts / self.seconds_per_hour)
                    ts_scaled.append(x_scaled)

            X["fourier_sin"] = [np.sin(2 * np.pi * term * ts) for ts in ts_scaled]
            X["fourier_cos"] = [np.cos(2 * np.pi * term * ts) for ts in ts_scaled]

        return X

# Transformers
horizon = 5

dtf = DatetimeFeatures(
    variables="index",
    features_to_extract=[
        "minute",
        "hour",
        "day_of_month",
        "month",
        "year",
        "day_of_year",
        "week",
        "day_of_week",
        "weekend"
    ],
    drop_original = False,
    utc = True
)

cyclicf = CyclicalFeatures(
    variables=["minute","hour", "month", "day_of_year"],
    drop_original= True
)

fourierf = AddFourierFeatures(
    K = 1,
    periods=[horizon, horizon*2],
    by = "hour"
)

lagf = LagFeatures(
    variables="Adj Close",
    periods=list(range(1,horizon+1)),
    missing_values = "ignore"
)

windf24 = WindowFeatures(
    variables="Adj Close",
    functions=["mean", "std"],
    periods=horizon,
    missing_values="ignore"
)

imputer = DropMissingData()

drop_features = DropFeatures(features_to_drop=["Adj Close"])

prep_pipeline = Pipeline([
    ('datetime features', dtf),
    ('cyclic features', cyclicf),
    ("fourier features", fourierf),
    ('lag features', lagf),
    ('window features', windf24),
    ('imputer', imputer),
    ('drop features', drop_features)
])

X_train_prep = prep_pipeline.fit_transform(X_train)
X_test_prep = prep_pipeline.transform(X_test)

# Align

y_train_prep = y_train.loc[X_train_prep.index]
y_test_prep = y_test.loc[X_test_prep.index]

X_train_prep.columns

# MODELING ----
# 1. LGBM : Bayesian Hyperparameter Tunning

dtrain = lgb.Dataset(X_train_prep, y_train_prep)
dtest = lgb.Dataset(X_test_prep, y_test_prep)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

params = {
    "objective":"regression",
    "metric": "rmse",
    "verbosity":-1,
    "boosting_type":"gbdt",
    "seed":42
}

tuner = lgb.LightGBMTunerCV(
    params,
    dtrain,
    folds = cv,
    callbacks=[early_stopping(200), log_evaluation(100)]
)

tuner.run()

# Build model

best_params = tuner.best_params
best_params

model = lgbm.LGBMRegressor(**best_params)
model.fit(X_train_prep, y_train_prep)

# Predict on test
preds = model.predict(X_test_prep)

# Predict on train
train_preds = model.predict(X_train_prep)

# EVALUATION ----

# Check test MAE
mean_absolute_error(y_test_prep, preds)

# Check train MAE
mean_absolute_error(y_train_prep, train_preds)

# Check results

results = pd.DataFrame(X_test_prep.copy(), columns=X_test_prep.columns)
results["pred"] = preds
results["actual"] = y_test
results["error"] = results["actual"] - results["pred"]
results = results.reset_index()

# Plot model performance

def plot_model_performance(df):
    plt.figure(figsize = (8,4))
    # Histogram
    plt.subplot(1,2,1)
    plt.hist(df["error"], bins=20)
    plt.xlabel("Forecast Error")
    plt.ylabel("Density")
    
    # Real - Preds
    plt.subplot(1,2,2)
    plt.scatter(df.actual, df.pred)
    p1 = max(max(df.pred), max(df.actual))
    p2 = min(min(df.pred), min(df.actual))
    plt.plot([p1, p2], [p1, p2], 'r-')
    plt.xlabel('y')
    plt.ylabel('y_pred')
    
    plt.tight_layout()

plot_model_performance(results)

# Plot predictions

results[['Datetime','pred', 'actual']].set_index('Datetime').plot()
plt.xlabel('Test timestamp')
plt.show()

# Metrics

def give_me_metrics(df):
    mae_test = mean_absolute_error(df.actual, df.pred)
    mse_test = mean_squared_error(df.actual, df.pred)
    rmse_test = mean_squared_error(df.actual, df.pred, squared=False)
    mape_test = mean_absolute_percentage_error(df.actual, df.pred)
    r2_test = r2_score(df.actual, df.pred)

    metric_df = pd.DataFrame({'MAE': [mae_test], 'MSE' : [mse_test], 'RMSE' : [rmse_test], 'MAPE': [mape_test], 'R2':[r2_test]})
    return metric_df

give_me_metrics(results)

lgbm.plot_importance(model, figsize=(10,7), height=0.9, title="Feature Importances", xlabel="Feature importance score", ylabel="Features")
plt.show()


# 2 ElasticNet ----

def objective(trial):
    params = {
        "alpha" : trial.suggest_float("alpha", 1e-5, 100),
        "l1_ratio" : trial.suggest_float("l1_ratio", 0, 1),
        "max_iter" : trial.suggest_int("max_iter", 1000, 2000)
    }

    model = ElasticNet(random_state = 42, **params)
    cv = KFold(n_splits = 5, shuffle = True, random_state = 42)
    
    train_pipeline = Pipeline([
        ("numeric scaler", StandardScaler()),
        ("model", model)
    ])

    scores = cross_val_score(train_pipeline, X_train_prep, y_train_prep, cv = cv, scoring = "neg_mean_absolute_error")
    return scores.mean()

sampler = TPESampler(seed = 42, multivariate = True)
study = optuna.create_study(direction = "maximize", sampler = sampler)
study.optimize(objective, n_trials = 100)

best_params = study.best_params
best_params

linear_model = ElasticNet(random_state = 42, **best_params)
linear_model.fit(X_train_prep, y_train_prep)

linear_preds = linear_model.predict(X_test_prep)
linear_train_preds = linear_model.predict(X_train_prep)

# Check test MAE
mean_absolute_error(y_test_prep, linear_preds)

# Check train MAE
mean_absolute_error(y_train_prep, linear_train_preds)

# Check results

results = pd.DataFrame(X_test_prep.copy(), columns=X_test_prep.columns)
results["pred"] = linear_preds
results["actual"] = y_test
results["error"] = results["actual"] - results["pred"]
results = results.reset_index()

# Model performance
plot_model_performance(results)

# Plot results
results[['Datetime','pred', 'actual']].set_index('Datetime').plot()
plt.xlabel('Test timestamp')
plt.show()

# Metrics
give_me_metrics(results)

