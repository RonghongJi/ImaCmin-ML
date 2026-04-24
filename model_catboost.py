# -*- coding: utf-8 -*-
"""
Created on 2026/04/22 11:36:10

@File    :   model_catboost.py
@Author  :   Ronghong Ji
"""

# %%
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
from catboost import CatBoostRegressor
import numpy as np
from sklearn.preprocessing import StandardScaler

# %%
seed=292

# 加载数据
df = pd.read_excel(r'../output/process_data/df_select_3.xlsx')

X = df.drop(columns=['Cmin'])
y = df[['Cmin']]  # 只使用 'Cmin' 作为目标变量

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# %%
# 设置固定的参数（固定 iterations 和早停策略）
fixed_params = {
    'random_seed': seed,
    'verbose': 0,
    'iterations': 1000,  # 设置较大的 iteration 值，依赖早停
    'early_stopping_rounds': 100,
}

# %%
# 调整 learning_rate 参数
print("Optimizing learning_rate...")
param_grid = {'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1, 0.3]}  # 选择不同的 learning_rate

catboost_model = CatBoostRegressor(**fixed_params)

# 使用 GridSearchCV 进行调参
grid_search = GridSearchCV(catboost_model, param_grid=param_grid, scoring='r2', cv=10, n_jobs=-1, return_train_score=True)
grid_search.fit(X_train, y_train['Cmin'])

# 输出每轮的测试分数
for i, mean_test_score in enumerate(grid_search.cv_results_['mean_test_score']):
    print(f"Iteration {i + 1} - Mean Test Score for learning_rate={grid_search.cv_results_['param_learning_rate'][i]}: {mean_test_score:.4f}")

best_learning_rate = grid_search.best_params_['learning_rate']
fixed_params['learning_rate'] = best_learning_rate
print(f"Best learning_rate: {best_learning_rate}")
print(f"Best model score for learning_rate: {grid_search.best_score_:.4f}")

# %%
# 调整 depth 参数
print("Optimizing depth...")
param_grid = {'depth': [3, 4, 5, 6]}  # 选择不同的 depth
catboost_model = CatBoostRegressor(**fixed_params)

# 使用 GridSearchCV 进行调参
grid_search = GridSearchCV(catboost_model, param_grid=param_grid, scoring='r2', cv=10, n_jobs=-1, return_train_score=True)
grid_search.fit(X_train, y_train['Cmin'])

# 输出每轮的测试分数
for i, mean_test_score in enumerate(grid_search.cv_results_['mean_test_score']):
    print(f"Iteration {i + 1} - Mean Test Score for depth={grid_search.cv_results_['param_depth'][i]}: {mean_test_score:.4f}")

best_depth = grid_search.best_params_['depth']
fixed_params['depth'] = best_depth
print(f"Best depth: {best_depth}")
print(f"Best model score for depth: {grid_search.best_score_:.4f}")

# %%
# 调整 l2_leaf_reg 参数
print("Optimizing l2_leaf_reg...")
param_grid = {'l2_leaf_reg': [1,2,3,4,5,6,7,8,9,10]}  # 选择不同的 l2_leaf_reg
catboost_model = CatBoostRegressor(**fixed_params)

# 使用 GridSearchCV 进行调参
grid_search = GridSearchCV(catboost_model, param_grid=param_grid, scoring='r2', cv=10, n_jobs=-1, return_train_score=True)
grid_search.fit(X_train, y_train['Cmin'])

# 输出每轮的测试分数
for i, mean_test_score in enumerate(grid_search.cv_results_['mean_test_score']):
    print(f"Iteration {i + 1} - Mean Test Score for l2_leaf_reg={grid_search.cv_results_['param_l2_leaf_reg'][i]}: {mean_test_score:.4f}")

best_l2_leaf_reg = grid_search.best_params_['l2_leaf_reg']
fixed_params['l2_leaf_reg'] = best_l2_leaf_reg
print(f"Best l2_leaf_reg: {best_l2_leaf_reg}")
print(f"Best model score for l2_leaf_reg: {grid_search.best_score_:.4f}")

# %%
# 调整 subsample 参数
print("Optimizing subsample...")
param_grid = {'subsample': [0.8, 0.9, 1.0]}  # 选择不同的 subsample
catboost_model = CatBoostRegressor(**fixed_params)

# 使用 GridSearchCV 进行调参
grid_search = GridSearchCV(catboost_model, param_grid=param_grid, scoring='r2', cv=10, n_jobs=-1, return_train_score=True)
grid_search.fit(X_train, y_train['Cmin'])

# 输出每轮的测试分数
for i, mean_test_score in enumerate(grid_search.cv_results_['mean_test_score']):
    print(f"Iteration {i + 1} - Mean Test Score for subsample={grid_search.cv_results_['param_subsample'][i]}: {mean_test_score:.4f}")

best_subsample = grid_search.best_params_['subsample']
fixed_params['subsample'] = best_subsample
print(f"Best subsample: {best_subsample}")
print(f"Best model score for subsample: {grid_search.best_score_:.4f}")

# %%
# 调整 rsm 参数
print("Optimizing rsm...")
param_grid = {'rsm': [0.8, 0.9, 1.0]}  # 选择不同的 rsm
catboost_model = CatBoostRegressor(**fixed_params)

# 使用 GridSearchCV 进行调参
grid_search = GridSearchCV(catboost_model, param_grid=param_grid, scoring='r2', cv=10, n_jobs=-1, return_train_score=True)
grid_search.fit(X_train, y_train['Cmin'])

# 输出每轮的测试分数
for i, mean_test_score in enumerate(grid_search.cv_results_['mean_test_score']):
    print(f"Iteration {i + 1} - Mean Test Score for rsm={grid_search.cv_results_['param_rsm'][i]}: {mean_test_score:.4f}")

best_rsm = grid_search.best_params_['rsm']
fixed_params['rsm'] = best_rsm
print(f"Best rsm: {best_rsm}")
print(f"Best model score for rsm: {grid_search.best_score_:.4f}")

# %%
# 输出最终的最佳参数
print(f"Best parameters after optimization: {fixed_params}")

# 使用逐个调参后的最佳参数训练最终模型
catboost_best_model = CatBoostRegressor(**fixed_params)
catboost_best_model.fit(X_train, y_train['Cmin'])

# 保存训练好的 CatBoost 模型到 pickle 文件
with open(r'../output/machine_learning/pickle/CatBoost_Model.pkl', 'wb') as model_file:
    pickle.dump(catboost_best_model, model_file)

# %%
# --------------------------------------
# 预测部分：加载模型并进行预测
# --------------------------------------

# 加载保存的模型
with open(r'../output/machine_learning/pickle/CatBoost_Model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# %%
loaded_model.get_params()

# %%
# 定义评估指标函数，计算 Cmin 的指标
def calculate_regression_metrics(y_true, y_pred):
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MPE(%)': (np.sum((y_pred - y_true.values.ravel())/y_true.values.ravel())/y_true.shape[0])*100,
        'RMSE(%)': np.sqrt(np.sum(((y_pred - y_true.values.ravel())/y_true.values.ravel())**2)/y_true.shape[0])*100
    }
    
# 十折交叉验证部分，返回 CL 的回归指标
def cross_val_regression_metrics(model, X, y, cv):
    metrics_list = []
    for train_idx, val_idx in cv.split(X):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train_fold, y_train_fold)
        val_predictions = model.predict(X_val_fold)
        fold_metrics = calculate_regression_metrics(y_val_fold, val_predictions)
        metrics_list.append(fold_metrics)
    
    # 计算平均值和标准差
    final_metrics = {
        'RMSE': (np.mean([m['RMSE'] for m in metrics_list]), np.std([m['RMSE'] for m in metrics_list])),
        'MAE': (np.mean([m['MAE'] for m in metrics_list]), np.std([m['MAE'] for m in metrics_list])),
        'MPE(%)': (np.mean([m['MPE(%)'] for m in metrics_list]), np.std([m['MPE(%)'] for m in metrics_list])),
        'RMSE(%)': (np.mean([m['RMSE(%)'] for m in metrics_list]), np.std([m['RMSE(%)'] for m in metrics_list])),
    }
    return final_metrics

# %%
# 定义十折交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=seed)

# 计算十折交叉验证的回归指标
catboost_cv_metrics = cross_val_regression_metrics(loaded_model, X_train, y_train, kf)
metrics_cv_df = {
    'Ten_fold_CV_RMSE': f"{catboost_cv_metrics['RMSE'][0]:.2f} ± {catboost_cv_metrics['RMSE'][1]:.2f}",
    'Ten_fold_CV_MAE': f"{catboost_cv_metrics['MAE'][0]:.2f} ± {catboost_cv_metrics['MAE'][1]:.2f}",
    'Ten_fold_CV_MPE(%)': f"{catboost_cv_metrics['MPE(%)'][0]:.2f} ± {catboost_cv_metrics['MPE(%)'][1]:.2f}",
    'Ten_fold_CV_RMSE(%)': f"{catboost_cv_metrics['RMSE(%)'][0]:.2f} ± {catboost_cv_metrics['RMSE(%)'][1]:.2f}",
}
metrics_cv_df = pd.DataFrame(metrics_cv_df, index=[0])
metrics_cv_df

# %%
# 训练最终模型并预测
loaded_model.fit(X_train, y_train['Cmin'])
catboost_y_pred = loaded_model.predict(X_test)

# 计算测试集上的回归指标
catboost_test_metrics = calculate_regression_metrics(y_test, catboost_y_pred)

# 合并十折交叉验证和测试集的回归指标
metrics_test = {
    'Test_RMSE': f"{catboost_test_metrics['RMSE']:.2f}",
    'Test_MAE': f"{catboost_test_metrics['MAE']:.2f}",
    'Test_MPE(%)': f"{catboost_test_metrics['MPE(%)']:.2f}",
    'Test_RMSE(%)': f"{catboost_test_metrics['RMSE(%)']:.2f}"
}


# 保存到 XLSX 文件
metrics_test_df = pd.DataFrame(metrics_test, index=[0])
metrics_test_df

# %%
loaded_model.feature_importances_

# %%
# 计算特征重要性
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': loaded_model.feature_importances_
})
feature_importance = feature_importance.sort_values(by='importance', ascending=False)
feature_importance

# %%
catboost_test_metrics

# %%
metrics_df = pd.concat([metrics_cv_df, metrics_test_df], axis=1)
metrics_df.to_excel(r'../output/machine_learning/result/CatBoost_result.xlsx', index=False)
metrics_df

# %%
# # 保存预测结果

y_test_df = pd.DataFrame(y_test).reset_index(drop=True)
xgb_y_pred_df = pd.DataFrame(catboost_y_pred).reset_index(drop=True)

y_test_df.columns = ['True Values Cmin']
xgb_y_pred_df.columns = ['Predicted Values Cmin']

output_df = pd.concat([y_test_df, xgb_y_pred_df], axis=1)
output_df.to_excel(r'../output/machine_learning/pred/CatBoost_predictions.xlsx', index=False)

print("ok!")