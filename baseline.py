from paddle import optimizer 
import numpy as np
from pprint import pprint
import pickle as pkl
import paddle as pdl
import paddle.nn as nn
import paddle.nn.functional as F
from GEM.baseline_model import ADMET
import random
from GEM.until import get_data_loader
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def trial(model_version, model, batch_size, criterion, scheduler, opt):
    # 创建dataloader
    train_data_loader, valid_data_loader = get_data_loader(mode='train', batch_size=batch_size)   

    current_best_metric = -1e10
    max_bearable_epoch = 50    # 设置早停的轮数为50，若连续50轮内验证集的评价指标没有提升，则停止训练
    current_best_epoch = 0

    train_metric_list = []     # 记录训练过程中各指标的变化情况
    valid_metric_list = []

    for epoch in range(800):   # 设置最多训练800轮
        model.train()
        for (atom_bond_graph, bond_angle_graph, label_true_batch) in train_data_loader:
#             atom_bond_graph = atom_bond_graph.to(device)
#             bond_angle_graph = bond_angle_graph.to(device)
#             label_true_batch = label_true_batch.to(device)
            label_predict_batch = model(atom_bond_graph, bond_angle_graph)
            label_true_batch = pdl.to_tensor(label_true_batch, dtype=pdl.int64, place=pdl.CUDAPlace(0))
            loss = criterion(label_predict_batch, label_true_batch)
            loss.backward()   # 反向传播
            opt.step()   # 更新参数
            opt.clear_grad()
        scheduler.step()   # 更新学习率

        # 评估模型在训练集、验证集的表现
        metric_train = evaluate(model, train_data_loader)
        metric_valid = evaluate(model, valid_data_loader)

        train_metric_list.append(metric_train)
        valid_metric_list.append(metric_valid)

        score = round((metric_valid['ap'] + metric_valid['auc']) / 2, 4)

        if score > current_best_metric:
            # 保存score最大时的模型权重
            current_best_metric = score
            current_best_epoch = epoch
            pdl.save(model.state_dict(), "weight/" + model_version + ".pkl")
        print("=========================================================")
        print("Epoch", epoch)
        pprint(("Train", metric_train))
        pprint(("Validate", metric_valid))
        print('current_best_epoch', current_best_epoch, 'current_best_metric', current_best_metric)
        if epoch > current_best_epoch + max_bearable_epoch:
            break

    return train_metric_list, valid_metric_list


def evaluate(model, data_loader):
    """评估模型"""
    model.eval()
    label_true = pdl.to_tensor([], dtype=pdl.float32, place=pdl.CUDAPlace(0))
    label_predict = pdl.to_tensor([], dtype=pdl.float32, place=pdl.CUDAPlace(0))
    for (atom_bond_graph, bond_angle_graph, label_true_batch) in data_loader:
#         atom_bond_graph = atom_bond_graph.to(device)
#         bond_angle_graph = bond_angle_graph.to(device)
#         label_true_batch = label_true_batch.to(device)
        label_predict_batch = model(atom_bond_graph, bond_angle_graph)
        label_true_batch = pdl.to_tensor(label_true_batch, dtype=pdl.int64, place=pdl.CUDAPlace(0))
        label_predict_batch = F.softmax(label_predict_batch)

        label_true = pdl.concat((label_true, label_true_batch.detach()), axis=0)
        label_predict = pdl.concat((label_predict, label_predict_batch.detach()), axis=0)
    
    y_pred = label_predict[:, 1].cpu().numpy()
    y_true = label_true.cpu().numpy()

    ap = round(average_precision_score(y_true, y_pred), 4)
    auc = round(roc_auc_score(y_true, y_pred), 4)

    y_pred = np.where(y_pred >= 0.5, 1, 0)
    accuracy = round(accuracy_score(y_true, y_pred), 4)
    precision = round(precision_score(y_true, y_pred), 4)
    recall = round(recall_score(y_true, y_pred), 4)
    f1 = round(f1_score(y_true, y_pred), 4)
    confusion_mat = confusion_matrix(y_true, y_pred)

    metric = {'ap': ap, 'auc': auc, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'confusion_mat': confusion_mat}
    return metric

import warnings
warnings.filterwarnings('ignore')

# 固定随机种子
SEED = 1024
pdl.seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
pdl.device.set_device('gpu:0')

model = ADMET()

batch_size = 4096                                                                   # batch size
criterion = nn.CrossEntropyLoss()                                                  # 损失函数
scheduler = optimizer.lr.CosineAnnealingDecay(learning_rate=1e-3, T_max=15)         # 余弦退火学习率
opt = optimizer.Adam(scheduler, parameters=model.parameters(), weight_decay=1e-5)   # 优化器

metric_train_list, metric_valid_list = trial(model_version='1', model=model, batch_size=batch_size, criterion=criterion, scheduler=scheduler, opt=opt)