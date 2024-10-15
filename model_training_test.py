# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from pytorch_lightning.loggers import TensorBoardLogger
import torchmetrics

# %%
train_epochs=512
learning_rate = 1e-5
weight_decay=1e-4
batch_size = 32

# %%
df = pd.read_csv('mixed.csv')

X = df.drop('gesture', axis=1).to_numpy()
y = df['gesture'].to_numpy()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=114514)
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=114514)

# %%
class GestureDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).to("cuda")
        self.y = torch.tensor(y, dtype=torch.long).to("cuda")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 训练集和验证集的 Dataset
train_dataset = GestureDataset(x_train, y_train)
val_dataset = GestureDataset(x_val, y_val)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


# %%
from torch import log_softmax


class GestureClassifier(pl.LightningModule):
    def __init__(self, input_dim, num_classes,learn_rate,weight_decay):
        super(GestureClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.TransformerEncoderLayer(d_model=32,nhead=8),
            nn.Linear(32, 64),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(64, num_classes),  # 输出 logits
            nn.LogSoftmax()
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.learn_rate = learn_rate
        self.weight_decay = weight_decay

        # 评估指标，指定 task 为 'multiclass'，并提供 num_classes
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.precision = torchmetrics.Precision(task='multiclass', num_classes=num_classes, average='weighted')
        self.recall = torchmetrics.Recall(task='multiclass', num_classes=num_classes, average='weighted')
        self.f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='weighted')

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        # 记录训练损失
        self.log('train_loss', loss, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # 计算预测类别
        preds = torch.argmax(logits, dim=1)

        # 更新评估指标
        acc = self.accuracy(preds, y)
        prec = self.precision(preds, y)
        rec = self.recall(preds, y)
        f1 = self.f1(preds, y)

        # 使用相同的 global_step 记录所有评估指标
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_acc', acc, prog_bar=True, logger=True)
        self.log('val_precision', prec, prog_bar=True, logger=True)
        self.log('val_recall', rec, prog_bar=True, logger=True)
        self.log('val_f1', f1, prog_bar=True, logger=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learn_rate,weight_decay=self.weight_decay)
        return optimizer

# %%

# 定义 TensorBoard logger
logger = TensorBoardLogger('tb_logs', name='gesture_classification')

# 获取输入维度和类别数量
input_dim = x_train.shape[1]
num_classes = len(label_encoder.classes_)

# 创建模型
model = GestureClassifier(input_dim=input_dim, num_classes=num_classes,learn_rate=learning_rate,weight_decay=weight_decay)

# 定义 Trainer，添加 TensorBoard logger
trainer = pl.Trainer(max_epochs=train_epochs, logger=logger,accelerator="gpu",devices=1)

# 训练模型
trainer.fit(model, train_loader, val_loader)


# %%
# 验证模型
trainer.validate(model, val_loader)

# %%
torch.save(model.state_dict,"checkpoint.pth")

# %%
dummy_input = torch.randn(1, input_dim)  # 输入维度应与模型的输入匹配
torch.onnx.export(model, dummy_input, "gesture_classifier.onnx", 
                  export_params=True,        # 导出所有权重
                  opset_version=14,          # ONNX 版本
                  do_constant_folding=True,  # 常量折叠优化
                  input_names=['input'],     # 输入名
                  output_names=['output'])   # 输出名

# %%
label_encoder.classes_


