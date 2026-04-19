import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error

def fitur_baru(df):
    df_eng = df.copy()
    
    df_eng['academic_index'] = (df_eng['ssc_percentage'] + 
                                df_eng['hsc_percentage'] + 
                                df_eng['degree_percentage'] + 
                                (df_eng['cgpa'] * 10)) / 4
        
    df_eng['job_readiness'] = (df_eng['internship_count'] + 
                               df_eng['live_projects'] + 
                               df_eng['certifications'])
    
    df_eng['total_comp_score'] = df_eng['technical_skill_score'] + df_eng['soft_skill_score']

    return df_eng

def train_model_classification(x_train, y_train):
    ARTIFACTS_DIR = Path("artifacts")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    feature_engineer = FunctionTransformer(fitur_baru, validate=False)

    preprocess = ColumnTransformer(transformers=[
        ('num', StandardScaler(), make_column_selector(dtype_include=np.number)),
        ('cat', OneHotEncoder(handle_unknown='ignore'), make_column_selector(dtype_include=['object', 'category']))
    ], remainder='drop')

    count_neg = np.sum(y_train == 0)
    count_pos = np.sum(y_train == 1)
    ratio = count_neg / count_pos

    model_params = {
        "colsample_bytree": 0.8,
        "gamma": 0,
        "learning_rate": 0.1,
        "max_depth": 3,
        "n_estimators": 100,
        "subsample": 0.7,
        "random_state": 42,
        "scale_pos_weight" : ratio,
        "use_label_encoder" : False,
        "eval_metric" : "logloss"
    }


    pipeline = Pipeline([
        ('engineering', feature_engineer),
        ('preprocessing', preprocess),
        ('regressor', XGBClassifier(**model_params))
    ])


    mlflow.log_params(model_params)

    cv_scores = cross_val_score(pipeline, x_train, y_train, scoring = 'f1')
    mlflow.log_metric("cv_avg_f1", -cv_scores.mean())

    pipeline.fit(x_train, y_train)

    # Save and Log Model
    joblib.dump(pipeline, "artifacts/placement_classification.pkl")
    mlflow.sklearn.log_model(pipeline, artifact_path="model")

    return mlflow.active_run().info.run_id

def train_model_reg(x_train, y_train):
    ARTIFACTS_DIR = Path("artifacts")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    feature_engineer = FunctionTransformer(fitur_baru, validate=False)

    preprocess = ColumnTransformer(transformers=[
        ('num', StandardScaler(), make_column_selector(dtype_include=np.number)),
        ('cat', OneHotEncoder(handle_unknown='ignore'), make_column_selector(dtype_include=['object', 'category']))
    ], remainder='drop')

    model_params = {
        "learning_rate" : 0.01, 
        "n_estimators": 100,
        "max_depth": 3,
        "random_state": 42,
        "subsample": 1.0 
    }

    pipeline = Pipeline([
        ('engineering', feature_engineer),
        ('preprocessing', preprocess),
        ('regressor', XGBRegressor(**model_params))
    ])


    mlflow.log_params(model_params)

    cv_scores = cross_val_score(pipeline, x_train, y_train, scoring = 'neg_mean_absolute_percentage_error')
    mlflow.log_metric("cv_avg_mape", -cv_scores.mean())

    pipeline.fit(x_train, y_train)

    # Save and Log Model
    joblib.dump(pipeline, "artifacts/salary_regression.pkl")
    mlflow.sklearn.log_model(pipeline, artifact_path="model")

    return mlflow.active_run().info.run_id