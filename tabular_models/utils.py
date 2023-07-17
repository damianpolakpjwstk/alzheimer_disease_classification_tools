import re

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             roc_auc_score, roc_curve)


def get_freesurfer_columns(df_freesurfer: pd.DataFrame, nan_threshold: float = 0.2) -> list[str]:
    """
    Filter out columns with more than nan_threshold% of NaN values and return a list of columns containing
    volumetric data.

    :param df_freesurfer: loaded pd.DataFrame containing volumetric data from FreeSurfer.
    :param nan_threshold: threshold for NaN values. Columns with more than nan_threshold% of NaN values are dropped.
    :return: List of columns names containing meaningful volumetric data from FreeSurfer.
    """
    isna_columns = (df_freesurfer.isna().sum() / len(df_freesurfer))
    isna_columns = isna_columns[isna_columns > nan_threshold].index
    df_freesurfer.drop(isna_columns, axis=1, inplace=True)
    freesurfer_datacolumns = [col for col in list(df_freesurfer.columns) if col.startswith("ST")]
    freesurfer_datacolumns.remove("STATUS")
    return freesurfer_datacolumns


def join_dfs(df_adnimerge: pd.DataFrame, df_freesurfer: pd.DataFrame,
             join_columns: list[str] | None = None) -> pd.DataFrame:
    """
    Join ADNIMERGE.csv and FreeSurfer volumetric (UCSFFSL_02_01_16_09Jun2023.csv) dataframes on IMAGEUID column.
    Return a dataframe containing only the columns specified in join_columns. If join_columns is None, return
    all columns.

    :param df_adnimerge: loaded pd.DataFrame containing ADNIMERGE.csv data.
    :param df_freesurfer: loaded pd.DataFrame containing volumetric data from FreeSurfer.
    :param join_columns:
    :return:
    """
    df = df_adnimerge.merge(df_freesurfer, on=["IMAGEUID"])

    if join_columns is not None:
        df[join_columns] = df[join_columns].replace('[^0-9.]', '', regex=True)
        df[join_columns] = df[join_columns].astype(float)
    return df


def get_xgboost_explanations(model, columns: list[str, ...]) -> pd.DataFrame:
    """
    Get feature importance from XGBoost model and return a dataframe containing the 10 most important features.

    :param model: XGBoost model object.
    :param columns: List of column names.
    :return: 10 most important features.
    """
    explainations = []
    for col, val in model.get_booster().get_score().items():
        col_id = int(str(col)[1:])
        col_name = columns[col_id]
        explainations.append({"feature": col_name, "importance": val})
    explain_df = pd.DataFrame(explainations).sort_values("importance", ascending=False)
    return explain_df.head(10)


def get_svm_explanations(model, columns: list[str, ...], X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """
    Get feature importance from SVM model and return a dataframe containing the 10 most important features.

    :param model: SVM model object.
    :param columns: List of column names used for training.
    :param X: Data used for training.
    :param y: labels used for training.
    :return: 10 most important features.
    """
    explainations = []
    result = permutation_importance(model, X, y, random_state=0, n_repeats=5)
    for i in result.importances_mean.argsort()[::-1]:
        col_name = columns[i]
        explainations.append({"feature": col_name, "importance": result.importances_mean[i]})
    explain_df = pd.DataFrame(explainations).sort_values("importance", ascending=False)
    return explain_df.head(10)


def filter_column_name(column_name: str) -> str:
    """
    Filter explain string from ADNI Freesurfer to make it more readable.

    Remove parentheses and everything inside, filter excess spaces and map "Standard Deviation" to "st. dev.".
    :param column_name: Name of the column.
    :return: Redacted column name.
    """
    column_name = re.sub(r'\([^)]*\)', '', column_name)
    column_name = re.sub(' +', ' ', column_name)
    column_name = re.sub(r'Standard Deviation', 'st. dev.', column_name)
    return column_name


def evaluate_model(model, test_X: np.ndarray, test_y: np.ndarray, val_X: np.ndarray,
                   val_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Evaluate model on test and validation set and print the results.

    Use accuracy_score, ROC AUC, and balanced_accuracy_score. Return ROC curve data to plot it later.
    :param model: Model object with .predict() method.
    :param test_X: X data for test set.
    :param test_y: y data for test set.
    :param val_X: X data for validation set.
    :param val_y: y data for validation set.
    :return: ROC curve data.
    """
    y_pred_val = model.predict(val_X)
    y_pred_test = model.predict(test_X)
    y_pred_test_proba = model.predict_proba(test_X)
    y_pred_val_proba = model.predict_proba(val_X)
    acc_val = accuracy_score(val_y, y_pred_val)
    acc_test = accuracy_score(test_y, y_pred_test)
    acc_val_plus_test = accuracy_score(np.concatenate((val_y, test_y)), np.concatenate((y_pred_val, y_pred_test)))
    print(f"Accuracy on validation set: {acc_val.__round__(3)}")
    print(f"Accuracy on test set: {acc_test.__round__(3)}")
    print(f"Accuracy on validation + test set: {acc_val_plus_test.__round__(3)}")
    balanced_accuracy_val = balanced_accuracy_score(val_y, y_pred_val)
    balanced_accuracy_test = balanced_accuracy_score(test_y, y_pred_test)
    balanced_accuracy_val_plus_test = balanced_accuracy_score(np.concatenate((val_y, test_y)),
                                                              np.concatenate((y_pred_val, y_pred_test)))
    print(f"Balanced accuracy on validation set: {balanced_accuracy_val.__round__(3)}")
    print(f"Balanced accuracy on test set: {balanced_accuracy_test.__round__(3)}")
    print(f"Balanced accuracy on validation + test set: {balanced_accuracy_val_plus_test.__round__(3)}")
    if len(np.unique(val_y)) == 2:
        roc_auc_val = roc_auc_score(val_y, y_pred_val)
        roc_auc_test = roc_auc_score(test_y, y_pred_test)
        roc_auc_val_plus_test = roc_auc_score(np.concatenate((val_y, test_y)),
                                              np.concatenate((y_pred_val, y_pred_test)))
        roc = roc_curve(test_y, y_pred_test)
    else:
        roc_auc_val = roc_auc_score(val_y, y_pred_val_proba, multi_class='ovr')
        roc_auc_test = roc_auc_score(test_y, y_pred_test_proba, multi_class='ovr')
        roc_auc_val_plus_test = roc_auc_score(np.concatenate((val_y, test_y)),
                                              np.concatenate((y_pred_val_proba, y_pred_test_proba)), multi_class='ovo')
        roc = None

    print(f"ROC AUC on validation set: {roc_auc_val.__round__(3)}")
    print(f"ROC AUC on test set: {roc_auc_test.__round__(3)}")
    print(f"ROC AUC on validation + test set: {roc_auc_val_plus_test.__round__(3)}")



    return roc
