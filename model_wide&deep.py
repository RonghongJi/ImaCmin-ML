# -*- coding: utf-8 -*-
"""
Created on 2026/04/22 11:37:06

@File    :   model_wide&deep.py
@Author  :   Ronghong Ji
"""

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import pickle
import optuna


# 定义Wide & Deep模型
class WideDeepNetwork(nn.Module):
    def __init__(self, in_features, hidden_size=64, device='cpu'):
        super(WideDeepNetwork, self).__init__()
        self.wide = nn.Linear(in_features, 1)
        self.deep = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(hidden_size + 1, 1)
        self.to(device)

    def forward(self, inputs):
        wide_output = self.wide(inputs)
        deep_output = self.deep(inputs)
        combined_output = torch.cat((wide_output, deep_output), dim=1)
        output = self.output_layer(combined_output)
        return output

# %%
# 其他辅助函数
def seed_everything(seed):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

# %%
def train_model(model, train_loader, criterion, optimizer, epochs=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")


def eval_parameter_tuning(eval_model, data_loader, tag="train"):
    y_trues = []
    y_preds = []
    eval_model.eval()
    with torch.no_grad():
        for x, y in data_loader:
            output = eval_model(x)
            y_trues.append(y.cpu().squeeze(-1).numpy())
            y_preds.append(output.cpu().squeeze(-1).numpy())
    y_true = np.concatenate(y_trues)
    y_pred = np.concatenate(y_preds)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return r2


def parameter_tuning(X, y, seed, device):
    criterion = nn.MSELoss()
    
    seed_everything(seed)
    def objective(trial):
        learning_rate = trial.suggest_loguniform('lr', 1e-3, 1e-1)
        batch_size = trial.suggest_int('batch_size', 8, 16)
        hidden_size = trial.suggest_int('hidden_size', 8, 16)
        model = WideDeepNetwork(in_features=X.shape[1], hidden_size=hidden_size, device=device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        kf = KFold(n_splits=10, shuffle=True, random_state=seed)
        val_crition = 0
        for train_idx, val_idx in kf.split(X):
            train_data, val_data = X[train_idx].to(device), X[val_idx].to(device)
            train_labels, val_labels = y[train_idx].to(device), y[val_idx].to(device)
            train_dataset = TensorDataset(train_data, train_labels)
            val_dataset = TensorDataset(val_data, val_labels)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            train_model(model, train_loader, criterion, optimizer)
            r2 = eval_parameter_tuning(model, val_loader, tag="test")
            val_crition += r2 / len(val_loader)
        return val_crition / 10
    
    seed_everything(seed)
    sampler = optuna.samplers.TPESampler(seed=42) # 此处固定随机种子
    study = optuna.create_study(direction='maximize', sampler=sampler)
    # study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    print("Best loss:", study.best_value)
    print("Best parameters:", study.best_params)
    return study.best_params


def train_and_save_model(X, y, seed, device, best_params):
    learning_rate = best_params['lr']
    batch_size = best_params['batch_size']
    hidden_size = best_params['hidden_size']
    model = WideDeepNetwork(in_features=X.shape[1], hidden_size=hidden_size, device=device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_model(model, train_loader, criterion, optimizer)
    with open(r'../output/machine_learning/pickle/Wide&Deep_Model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)


def main_train():
    # 读取数据
    seed = 292
    seed_everything(seed)
    data = pd.read_excel(r'../output/process_data/df_select_3.xlsx')
    features = data.columns[:-1]
    label = data.columns[-1]
    X = data[features].values
    y = data[label].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed_everything(seed)
    # 参数调优
    best_params = parameter_tuning(X, y, seed, device)
    print(best_params)
    # 得到模型并保存
    train_and_save_model(X, y, seed, device, best_params)


if __name__ == '__main__':
    main_train()

# %%
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, KFold


def seed_everything(seed):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def eval_model(eval_model, data_loader, tag="train"):
    y_trues = []
    y_preds = []
    eval_model.eval()
    with torch.no_grad():
        for x, y in data_loader:
            output = eval_model(x)
            y_trues.append(y.cpu().squeeze(-1).numpy())
            y_preds.append(output.cpu().squeeze(-1).numpy())
    y_true = np.concatenate(y_trues)
    y_pred = np.concatenate(y_preds)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mpe_percent = (np.sum((y_pred - y_true) / y_true) / y_true.shape[0]) * 100
    rmse_percent = np.sqrt(np.sum(((y_pred - y_true) / y_true) ** 2) / y_true.shape[0]) * 100
    print(f"{tag}"
          f", mae={mae}"
          f", rmse={np.sqrt(mse)}"
          f", mpe(%)={mpe_percent}"
          f", rmse(%)={rmse_percent}")
    return mae, mse, np.sqrt(mse), mpe_percent, rmse_percent


def main_eval():
    
    # 读取数据
    seed = 292
    seed_everything(seed)

    data = pd.read_excel(r'../output/process_data/df_select_3.xlsx')
    features = data.columns[:-1]
    label = data.columns[-1]
    X = data[features].values
    y = data[label].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    seed_everything(seed)
    # 读取模型
    with open(r'../output/machine_learning/pickle/Wide&Deep_Model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    batch_size = 16  # 可以根据实际情况修改
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32).view(-1, 1)), batch_size=batch_size)
    # 进行十折交叉验证和测试集评估
    n_split = 10
    kf = KFold(n_splits=n_split, shuffle=True, random_state=seed)
    mae_list = []
    mse_list = []
    rmse_list = []
    mpe_percent_list = []
    rmse_percent_list = []
    y_pred = np.array([0] * len(y_test))
    for i, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"Training fold {i + 1}...")
        train_data = torch.tensor(X_train[train_idx], dtype=torch.float32)
        val_data = torch.tensor(X_train[val_idx], dtype=torch.float32)
        train_labels = torch.tensor(y_train[train_idx], dtype=torch.float32).view(-1, 1)
        val_labels = torch.tensor(y_train[val_idx], dtype=torch.float32).view(-1, 1)
        train_dataset = TensorDataset(train_data, train_labels)
        val_dataset = TensorDataset(val_data, val_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        eval_model(model, train_loader, tag="train")
        model.eval()
        y_pred = y_pred + model(X_test.to(device)).squeeze(-1).detach().cpu().numpy() / n_split
        mae, mse, rmse, mpe_percent, rmse_percent = eval_model(model, val_loader, tag="test")
        mae_list.append(mae)
        mse_list.append(mse)
        rmse_list.append(rmse)
        mpe_percent_list.append(mpe_percent)
        rmse_percent_list.append(rmse_percent)
    mae_mean = np.mean(mae_list)
    rmse_mean = np.mean(rmse_list)
    mpe_percent_mean = np.mean(mpe_percent_list)
    rmse_percent_mean = np.mean(rmse_percent_list)
    mae_std = np.std(mae_list)
    rmse_std = np.std(rmse_list)
    mpe_percent_std = np.std(mpe_percent_list)
    rmse_percent_std = np.std(rmse_percent_list)
    test_mae, test_mse, test_rmse, test_mpe_percent, test_rmse_percent = eval_model(model, test_loader, tag="test")
    df_dict = [{
        "Ten_fold_CV_RMSE": f"{rmse_mean:.2f} ± {rmse_std:.2f}",
        "Ten_fold_CV_MAE": f"{mae_mean:.2f} ± {mae_std:.2f}",
        "Ten_fold_CV_MPE(%)": f"{mpe_percent_mean:.2f} ± {mpe_percent_std:.2f}",
        "Ten_fold_CV_RMSE(%)": f"{rmse_percent_mean:.2f} ± {rmse_percent_std:.2f}",
        "Test_RMSE": f"{test_rmse:.2f}",
        "Test_MAE": f"{test_mae:.2f}",
        "Test_MPE(%)": f"{test_mpe_percent:.2f}",
        "Test_RMSE(%)": f"{test_rmse_percent:.2f}",
    }]
    metrics_df = pd.DataFrame(df_dict)
    metrics_df.to_excel(r"../output/machine_learning/result/Wide&Deep_result.xlsx", index=False)
    # 保存测试集的预测结果
    y_trues = []
    y_preds = []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            output = model(x)
            y_trues.append(y.cpu().squeeze(-1).numpy())
            y_preds.append(output.cpu().squeeze(-1).numpy())
    y_true = np.concatenate(y_trues)
    y_pred = np.concatenate(y_preds)
    test_df = pd.DataFrame()
    test_df['True Values Cmin'] = y_true
    test_df['Predicted Values Cmin'] = y_pred
    test_df.to_excel(r"../output/machine_learning/pred/Wide&Deep_predictions.xlsx", index=False)


if __name__ == '__main__':
    main_eval()


