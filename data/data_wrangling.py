import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def convert_col_to_cat(df: pd.core.frame.DataFrame, cat_col: list):
    """Convert categorical columns to type 'category'"""
    df_with_cat = pd.DataFrame()
    for col in df:
        if col in cat_col:
            df_with_cat[col] = df[col].astype('category')
        else:
            df_with_cat[col] = df[col]
    return df_with_cat


def convert_num_to_obj(df: pd.core.frame.DataFrame, numeric_cols: list):
    """Convert numeric columns to categorical columns"""
    df_with_converted_cols = pd.DataFrame()
    for col in df:
        if col in numeric_cols:
            df_with_converted_cols[col] = pd.cut(df[col], bins=10, include_lowest=True).astype(str).astype('category')
        else:
            df_with_converted_cols[col] = df[col]
    return df_with_converted_cols


def convert_cat_to_numeric(df: pd.core.frame.DataFrame, cat_with_more_than_2: list, cat_with_2_or_fewer: list):
    """Convert categorical columns to numeric columns"""
    df_with_converted_cols = pd.get_dummies(df, columns=cat_with_more_than_2)
    for col in cat_with_2_or_fewer:
        le = LabelEncoder()
        le.fit(df_with_converted_cols[col])
        df_with_converted_cols[col] = le.transform(df_with_converted_cols[col])
    return df_with_converted_cols


def standardize_numeric_variables(df: pd.core.frame.DataFrame, numeric_cols: list):
    """Standardize numeric columns"""
    df_with_standardized_cols = df.copy() 
    for col in numeric_cols:
        scaler = StandardScaler()
        df_with_standardized_cols[numeric_cols] = scaler.fit_transform(df_with_standardized_cols[numeric_cols])
    return df_with_standardized_cols


def convert_all_cat_to_labels(df: pd.core.frame.DataFrame, cat_col: list):
    """Convert all categorical columns to numeric by label encoding"""
    df_with_converted_cols = pd.DataFrame()
    for col in df:
        if col in cat_col:
            le = LabelEncoder()
            le.fit(df[col])
            df_with_converted_cols[col] = le.transform(df[col])
            df_with_converted_cols[col] = df_with_converted_cols[col].astype('category')
        else:
            df_with_converted_cols[col] = df[col]
    return df_with_converted_cols
