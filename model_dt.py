# -*- coding: utf-8 -*-
"""
Created on 2026/04/22 11:36:27

@File    :   model_dt.py
@Author  :   Ronghong Ji
"""

# %%
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import numpy as np

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
    'random_state': seed
    }

# %%
# 调整 max_depth 参数
print("Optimizing max_depth...")
param_grid = {'max_depth': [3, 4, 5]}  # 选择不同的 max_depth
dt_model = DecisionTreeRegressor(**fixed_params)

# 使用 GridSearchCV 进行调参
grid_search = GridSearchCV(dt_model, param_grid=param_grid, scoring='r2', cv=10, n_jobs=-1, return_train_score=True)
grid_search.fit(X_train, y_train['Cmin'])

# 输出每轮的测试分数
for i, mean_test_score in enumerate(grid_search.cv_results_['mean_test_score']):
    print(f"Iteration {i + 1} - Mean Test Score for max_depth={grid_search.cv_results_['param_max_depth'][i]}: {mean_test_score:.4f}")

best_max_depth = grid_search.best_params_['max_depth']
fixed_params['max_depth'] = best_max_depth
print(f"Best max_depth: {best_max_depth}")
print(f"Best model score for max_depth: {grid_search.best_score_:.4f}")

# %%
# # 调整 max_features 参数
# print("Optimizing max_features...")
# param_grid = {'max_features': ['log2', 'sqrt', 'auto', None]}  # 选择不同的 max_features
# dt_model = DecisionTreeRegressor(**fixed_params)

# # 使用 GridSearchCV 进行调参
# grid_search = GridSearchCV(dt_model, param_grid=param_grid, scoring='r2', cv=10, n_jobs=-1, return_train_score=True)
# grid_search.fit(X_train, y_train['Cmin'])

# # 输出每轮的测试分数
# for i, mean_test_score in enumerate(grid_search.cv_results_['mean_test_score']):
#     print(f"Iteration {i + 1} - Mean Test Score for max_features={grid_search.cv_results_['param_max_features'][i]}: {mean_test_score:.4f}")

# best_max_features = grid_search.best_params_['max_features']
# fixed_params['max_features'] = best_max_features
# print(f"Best max_features: {best_max_features}")
# print(f"Best model score for max_features: {grid_search.best_score_:.4f}")

# %%
# 调整 min_samples_split 参数
print("Optimizing min_samples_split...")
param_grid = {'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10]}  # 选择不同的 min_samples_split
dt_model = DecisionTreeRegressor(**fixed_params)

# 使用 GridSearchCV 进行调参
grid_search = GridSearchCV(dt_model, param_grid=param_grid, scoring='r2', cv=10, n_jobs=-1, return_train_score=True)
grid_search.fit(X_train, y_train['Cmin'])

# 输出每轮的测试分数
for i, mean_test_score in enumerate(grid_search.cv_results_['mean_test_score']):
    print(f"Iteration {i + 1} - Mean Test Score for min_samples_split={grid_search.cv_results_['param_min_samples_split'][i]}: {mean_test_score:.4f}")

best_min_samples_split = grid_search.best_params_['min_samples_split']
fixed_params['min_samples_split'] = best_min_samples_split
print(f"Best min_samples_split: {best_min_samples_split}")
print(f"Best model score for min_samples_split: {grid_search.best_score_:.4f}")

# %%
# 调整 min_samples_leaf 参数
print("Optimizing min_samples_leaf...")
param_grid = {'min_samples_leaf': [1, 2, 3, 4, 5]}  # 选择不同的 min_samples_leaf
dt_model = DecisionTreeRegressor(**fixed_params)

# 使用 GridSearchCV 进行调参
grid_search = GridSearchCV(dt_model, param_grid=param_grid, scoring='r2', cv=10, n_jobs=-1, return_train_score=True)
grid_search.fit(X_train, y_train['Cmin'])

# 输出每轮的测试分数
for i, mean_test_score in enumerate(grid_search.cv_results_['mean_test_score']):
    print(f"Iteration {i + 1} - Mean Test Score for min_samples_leaf={grid_search.cv_results_['param_min_samples_leaf'][i]}: {mean_test_score:.4f}")

best_min_samples_leaf = grid_search.best_params_['min_samples_leaf']
fixed_params['min_samples_leaf'] = best_min_samples_leaf
print(f"Best min_samples_leaf: {best_min_samples_leaf}")
print(f"Best model score for min_samples_leaf: {grid_search.best_score_:.4f}")

# %%
# 输出最终的最佳参数
print(f"Best parameters after optimization: {fixed_params}")

# 使用逐个调参后的最佳参数训练最终模型
dt_best_model = DecisionTreeRegressor(**fixed_params)
dt_best_model.fit(X_train, y_train['Cmin'])

# 保存训练好的 DecisionTreeRegressor 模型到 pickle 文件
with open(r'../output/machine_learning/pickle/DT_Model.pkl', 'wb') as model_file:
    pickle.dump(dt_best_model, model_file)

# %%
# --------------------------------------
# 预测部分：加载模型并进行预测
# --------------------------------------

# 加载保存的模型
with open(r'../output/machine_learning/pickle/DT_Model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

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
dt_cv_metrics = cross_val_regression_metrics(loaded_model, X_train, y_train, kf)
metrics_cv_df = {
    'Ten_fold_CV_RMSE': f"{dt_cv_metrics['RMSE'][0]:.2f} ± {dt_cv_metrics['RMSE'][1]:.2f}",
    'Ten_fold_CV_MAE': f"{dt_cv_metrics['MAE'][0]:.2f} ± {dt_cv_metrics['MAE'][1]:.2f}",
    'Ten_fold_CV_MPE(%)': f"{dt_cv_metrics['MPE(%)'][0]:.2f} ± {dt_cv_metrics['MPE(%)'][1]:.2f}",
    'Ten_fold_CV_RMSE(%)': f"{dt_cv_metrics['RMSE(%)'][0]:.2f} ± {dt_cv_metrics['RMSE(%)'][1]:.2f}",
}
metrics_cv_df = pd.DataFrame(metrics_cv_df, index=[0])
metrics_cv_df

# %%
# 训练最终模型并预测
loaded_model.fit(X_train, y_train['Cmin'])
dt_y_pred = loaded_model.predict(X_test)

# 计算测试集上的回归指标
dt_test_metrics = calculate_regression_metrics(y_test, dt_y_pred)

# 合并十折交叉验证和测试集的回归指标
metrics_test = {
    'Test_RMSE': f"{dt_test_metrics['RMSE']:.2f}",
    'Test_MAE': f"{dt_test_metrics['MAE']:.2f}",
    'Test_MPE(%)': f"{dt_test_metrics['MPE(%)']:.2f}",
    'Test_RMSE(%)': f"{dt_test_metrics['RMSE(%)']:.2f}"
}


# 保存到 XLSX 文件
metrics_test_df = pd.DataFrame(metrics_test, index=[0])
metrics_test_df

# %%
metrics_df = pd.concat([metrics_cv_df, metrics_test_df], axis=1)
metrics_df.to_excel(r'../output/machine_learning/result/DT_result.xlsx', index=False)
metrics_df

# %%
# # 保存预测结果
y_test_df = pd.DataFrame(y_test).reset_index(drop=True)
xgb_y_pred_df = pd.DataFrame(dt_y_pred).reset_index(drop=True)

y_test_df.columns = ['True Values Cmin']
xgb_y_pred_df.columns = ['Predicted Values Cmin']

output_df = pd.concat([y_test_df, xgb_y_pred_df], axis=1)
output_df.to_excel(r'../output/machine_learning/pred/DT_predictions.xlsx', index=False)

print("ok!")