# -*- coding: utf-8 -*-
"""
Created on 2026/04/22 11:35:49

@File    :   rf_imputation.py
@Author  :   Ronghong Ji
"""

# %%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 区分分类变量和数值型变量
def categorize_columns(df, threshold=5):
    categorical_cols = []
    numeric_cols = []
    
    for col in df.columns:
        # 如果唯一值的数量少于阈值，则视为分类变量
        if df[col].nunique() < threshold:
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)
    
    return categorical_cols, numeric_cols

# 将分类变量转换为 'category' 类型
def convert_to_category(df, categorical_cols):
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    return df

df_select_2 = pd.read_excel(r'../output/process_data/df_select_2.xlsx')

# 根据唯一值的数量来区分分类变量和数值变量
categorical_cols, continuous_cols = categorize_columns(df_select_2.drop(columns=['Cmin']), threshold=6)
df_convert = convert_to_category(df_select_2, categorical_cols)

# %%
categorical_cols

# %%
continuous_cols

# %%
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def random_forest_imputation(df, categorical_cols, continuous_cols):
    df_filled = df.copy()
    
    # 获取所有列
    all_cols = df.columns
    
    for col in all_cols:
        print(f"Processing column: {col}")
        
        # 检查列是否存在于数据框中
        if col not in df_filled.columns:
            print(f"Column {col} not found in the dataframe. Skipping.")
            continue
        
        # 训练数据
        train_data = df_filled[df_filled[col].notna()]
        X_train = train_data.drop(columns=[col])
        y_train = train_data[col]
        
        # 测试数据
        test_data = df_filled[df_filled[col].isna()]
        X_test = test_data.drop(columns=[col])
        
        if X_test.shape[0] == 0:
            print(f"No missing values to predict for column: {col}")
            continue
        
        # 根据列类型选择模型
        if col in categorical_cols:
            model = RandomForestClassifier(n_estimators=100, random_state=292)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=292)
        
        # 创建和训练模型
        model.fit(X_train, y_train)
        
        # 预测缺失值
        predicted_values = model.predict(X_test)
        
        # 填充缺失值
        df_filled.loc[df_filled[col].isna(), col] = predicted_values
    
    return df_filled

df_select_3 = random_forest_imputation(df_convert, categorical_cols, continuous_cols)
df_select_3.to_excel(r"../output/process_data/df_select_3.xlsx", index=False)
print(df_select_3.isnull().sum())


