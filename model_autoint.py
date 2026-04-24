# -*- coding: utf-8 -*-
"""
Created on 2026/04/22 11:36:04

@File    :   model_autoint.py
@Author  :   Ronghong Ji
"""

# %%

# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,weichenswc@163.com
Reference:
    [1] Song W, Shi C, Xiao Z, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks[J]. arXiv preprint arXiv:1810.11921, 2018.(https://arxiv.org/abs/1810.11921)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    """The Multi Layer Percetron

      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.

      Arguments
        - **inputs_dim**: input feature dimension.

        - **hidden_units**:list of positive integer, the layer number and units in each layer.

        - **activation**: Activation function to use.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not.

        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 init_std=0.0001, dice_dim=3, seed=1024, device='cpu'):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])
        activation_layer = nn.ReLU(inplace=True)
        self.activation_layers = nn.ModuleList(
            [activation_layer for _ in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.to(device)

    def forward(self, inputs):
        deep_input = inputs

        for i in range(len(self.linears)):

            fc = self.linears[i](deep_input)

            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)

            fc = self.dropout(fc)
            deep_input = fc
        return deep_input

class InteractingLayer(nn.Module):
    """A Layer used in AutoInt that model the correlations between different feature fields by multi-head self-attention mechanism.
      Input shape
            - A 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
            - 3D tensor with shape:``(batch_size,field_size,embedding_size)``.
      Arguments
            - **input_size** : Positive integer, dimensionality of input features.
            - **head_num**: int.The head number in multi-head self-attention network.
            - **use_res**: bool.Whether or not use standard residual connections before output.
            - **seed**: A Python integer to use as random seed.
      References
            - [Song W, Shi C, Xiao Z, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks[J]. arXiv preprint arXiv:1810.11921, 2018.](https://arxiv.org/abs/1810.11921)
    """

    def __init__(self, embedding_size=1, head_num=1, use_res=True, scaling=False, seed=1024, device='cpu'):
        super(InteractingLayer, self).__init__()
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        if embedding_size % head_num != 0:
            raise ValueError('embedding_size is not an integer multiple of head_num!')
        self.att_embedding_size = 1
        self.head_num = head_num
        self.use_res = use_res
        self.scaling = scaling
        self.seed = seed

        self.W_Query = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_key = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_Value = nn.Parameter(torch.Tensor(embedding_size, embedding_size))

        if self.use_res:
            self.W_Res = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)

        self.to(device)

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))

        # None F D
        querys = torch.tensordot(inputs, self.W_Query, dims=([-1], [0]))
        keys = torch.tensordot(inputs, self.W_key, dims=([-1], [0]))
        values = torch.tensordot(inputs, self.W_Value, dims=([-1], [0]))

        # head_num None F D/head_num
        querys = torch.stack(torch.split(querys, self.att_embedding_size, dim=2))
        keys = torch.stack(torch.split(keys, self.att_embedding_size, dim=2))
        values = torch.stack(torch.split(values, self.att_embedding_size, dim=2))

        inner_product = torch.einsum('bnik,bnjk->bnij', querys, keys)  # head_num None F F
        if self.scaling:
            inner_product /= self.att_embedding_size ** 0.5
        self.normalized_att_scores = F.softmax(inner_product, dim=-1)  # head_num None F F
        result = torch.matmul(self.normalized_att_scores, values)  # head_num None F D/head_num

        result = torch.cat(torch.split(result, 1, ), dim=-1)
        result = torch.squeeze(result, dim=0)  # None F D
        if self.use_res:
            result += torch.tensordot(inputs, self.W_Res, dims=([-1], [0]))
        result = F.relu(result)

        return result

class AutoInt(nn.Module):
    """Instantiates the AutoInt Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param att_layer_num: int.The InteractingLayer number to be used.
    :param att_head_num: int.The head number in multi-head  self-attention network.
    :param att_res: bool.Whether or not use standard residual connections before output.
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param dnn_activation: Activation function to use in DNN
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_use_bn:  bool. Whether use BatchNormalization before activation or not in DNN
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self, input_size, att_layer_num=3,
                 att_head_num=2, att_res=True, dnn_hidden_units=(256, 128), dnn_activation='relu',
                 l2_reg_dnn=0,  dnn_use_bn=False, dnn_dropout=0, init_std=0.0001,
                 device='cpu'):

        super(AutoInt, self).__init__()
        if len(dnn_hidden_units) <= 0 and att_layer_num <= 0:
            raise ValueError("Either hidden_layer or att_layer_num must > 0")

        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
        self.dnn_hidden_units = dnn_hidden_units
        self.att_layer_num = att_layer_num
        self.use_dnn = True
        if self.use_dnn:
            self.dnn = DNN(input_size, dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
        self.int_layers = nn.ModuleList(
            [InteractingLayer(1, att_head_num, att_res, device=device) for _ in range(att_layer_num)])

        self.to(device)

    def forward(self, X):
        X = X.unsqueeze(-1)


        att_input = X

        for layer in self.int_layers:
            att_input = layer(att_input)

        att_output = torch.flatten(att_input, start_dim=1)


        if len(self.dnn_hidden_units) > 0 and self.att_layer_num > 0:  # Deep & Interacting Layer
            deep_out = self.dnn(att_output)
            res = self.dnn_linear(deep_out)
        elif len(self.dnn_hidden_units) > 0:  # Only Deep
            deep_out = self.dnn(att_output)
            res = self.dnn_linear(deep_out)
        elif self.att_layer_num > 0:  # Only Interacting Layer
            res = self.dnn_linear(att_output)
        else:  # Error
            pass

        return res

# %%
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import pickle
import optuna
import os

# %%
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
        att_layer_num = trial.suggest_int('att_layer_num', 1, 4)
        # hidden_size = trial.suggest_int('hidden_size', 8, 16)
        model = AutoInt(input_size=X.shape[1], att_layer_num=att_layer_num, att_head_num=1, att_res=True, dnn_hidden_units=(256, 128))
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
    att_layer_num = best_params['att_layer_num']
    learning_rate = best_params['lr']
    batch_size = best_params['batch_size']
    # hidden_size = best_params['hidden_size']
    model = AutoInt(input_size=X.shape[1], att_layer_num=att_layer_num, att_head_num=1, att_res=True, dnn_hidden_units=(256, 128))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_model(model, train_loader, criterion, optimizer)
    with open(r'../output/machine_learning/pickle/AutoInt_Model.pkl', 'wb') as model_file:
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
import pickle
with open(r'../output/machine_learning/pickle/AutoInt_Model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# %%
model

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
    with open(r'../output/machine_learning/pickle/AutoInt_Model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    batch_size = 16
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
    metrics_df.to_excel(r"../output/machine_learning/result/AutoInt_result.xlsx", index=False)
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
    test_df.to_excel(r"../output/machine_learning/pred/AutoInt_predictions.xlsx", index=False)


if __name__ == '__main__':
    main_eval()


