# -*- coding: utf-8 -*-
"""
Created on 2026/04/22 11:34:14

@File    :   feature_selection.py
@Author  :   Ronghong Ji
"""

# %%
import re 
import pandas as pd

# 读取统计表
data = pd.read_excel(r'../output/statistic/人口统计学与临床统计描述.xlsx')

# 读取数据
df = pd.read_excel(r"../output/process_data/df.xlsx")

# 处理 p 值
data['p_value'] = data['p_value'].apply(lambda x: 0 if x == '<0.001' else float(x))

# 找出p值显著的变量
significant_variables = data[data['p_value'] <= 0.05]

# 提取对应的变量名称
variable_names = significant_variables['变量名称']

# 取出显著变量的变量名
variables_p_value = [re.sub(r'\s*,? (median.*|n.*)', '', re.sub(r'\s*\(.*\)', '', var)).strip() for var in variable_names]
intersection_variables = set(variables_p_value).intersection(set(df.columns))
print(intersection_variables)

# 找出p值不显著的变量
no_significant_variables = data[data['p_value'] > 0.05]

# 提取对应的变量名称
no_variable_names = no_significant_variables['变量名称']

# 取出不显著变量的变量名
no_variables_p_value = [re.sub(r'\s*,? (median.*|n.*)', '', re.sub(r'\s*\(.*\)', '', var)).strip() for var in no_variable_names]
intersection_no_variables = set(no_variables_p_value).intersection(set(df.drop(columns=['Cmin']).columns))
print(intersection_no_variables)

# %%
len(intersection_variables)

# %%
len(intersection_no_variables)

# %%
# 删除不显著变量
df_select_1 = df.drop(columns=intersection_no_variables, errors='ignore')
df_select_1.to_excel(r'../output/process_data/df_select_1.xlsx',index=False)
df_select_1.columns

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb
from lightgbm import LGBMRegressor

df = df_select_1.copy()

# 设置目标变量和特征
X = df.drop(columns=['Cmin'])
y = df['Cmin']

seed = 292

# 逐个优化参数
def grid_search_cv_single_param(param_grid, X, y, fixed_params):
    model = xgb.XGBRegressor(random_state=seed, **fixed_params)
    grid_search = GridSearchCV(model, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=10, n_jobs=-1)
    grid_search.fit(X, y) 
    return grid_search.best_params_


param_grids = {
    'n_estimators': [100, 150, 200, 250],
    'max_depth': [3,4,5],
    'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1],
}

# 初始化最佳参数字典
best_params = {}

for param, grid in param_grids.items():
    print(f"Optimizing {param}...")
    param_grid = {param: grid}
    best_param = grid_search_cv_single_param(param_grid, X, y, fixed_params={})
    best_params.update(best_param)
    print(f"Best {param}: {best_param}")

# 使用逐个优化后的最佳参数训练模型
xgboost_best_model = xgb.XGBRegressor(**best_params, random_state=seed)

# %%
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import warnings

# 屏蔽 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# 将数据集拆分为训练集和测试集
train_x, X_test, train_y, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# 统计特征数
num_features = train_x.shape[1]

# 初始化存储选择的特征和得分的列表
selected_feat_1r=[]
selected_feat1_1r=[]
scores_1r = []

# 特征选择循环，选择 1 到 train_x 中的所有特征
for i in range(1, num_features + 1):
    print(f"选择 {i} 个特征:")
    
    sfs = SFS(xgboost_best_model,
              k_features=i, 
              forward=True, 
              floating=False, 
              verbose=2,
              scoring='neg_mean_absolute_error',
              cv=2,
              n_jobs=-1)
    
    # 拟合模型
    sfs = sfs.fit(train_x, train_y)

    # 记录选择的特征
    selected_feat_1r = train_x.columns[list(sfs.k_feature_idx_)]
    selected_feat1_1r.append(selected_feat_1r)

    # 记录当前选择的特征组合的得分情况
    scores_1r.append(round(-sfs.k_score_, 2))


# %%
# 找到得分中的最大值及其对应的索引
best_index_1r = scores_1r.index(min(scores_1r))

# 找到最优的特征组合
best_features_1r = selected_feat1_1r[best_index_1r]

# 输出最优特征组合及其对应的分值
print("最优特征组合：", best_features_1r)
print("对应的 MAE 分数：", scores_1r[best_index_1r])

# %%
import matplotlib.pyplot as plt

# 设置字体为新罗马
plt.rc('font', family='Times New Roman')

# 绘制分值随着特征数量变化的曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_x.columns)+1), scores_1r, linestyle='-', color='#4682B4')  # 使用钢蓝色
plt.xlabel('Variable Numbers', fontsize=18)
plt.ylabel(r'MAE', fontsize=18)
plt.xlim(0, 20)
plt.ylim(200, 280)

# 设置坐标轴样式
plt.grid(False)
# 设置 xy 轴的黑线
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')

# 设置刻度字体大小
plt.xticks(range(0, 19, 2), fontsize=16
plt.yticks(fontsize=16)
plt.show()


# %%
# 创建最终数据表
final_features = X[best_features_1r]
df_select_2 = pd.concat([final_features, y], axis=1)
df_select_2

# %%
df_select_2.columns

# %%
df_select_2.to_excel(r'../output/process_data/df_select_2.xlsx',index=False)


