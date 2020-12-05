import torch
import torch.nn as nn
import torch.nn.functional as F


class MoaModel(nn.Module):
    def __init__(self, input_size, hidden_size1=2048, hidden_size2=2048,
                 output_size=206, dropout_ratio=0.4, f_dropout_ratio=0.2):
        super(MoaModel, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(input_size)
        self.dropout1 = nn.Dropout(f_dropout_ratio)
        self.dense1 = nn.Linear(input_size, hidden_size1)
        nn.init.kaiming_normal_(self.dense1.weight, mode='fan_in', nonlinearity='leaky_relu')

        self.batch_norm2 = nn.BatchNorm1d(hidden_size1)
        self.dropout2 = nn.Dropout(dropout_ratio)
        self.dense2 = nn.Linear(hidden_size1, hidden_size2)
        nn.init.kaiming_normal_(self.dense2.weight, mode='fan_in', nonlinearity='leaky_relu')

        self.batch_norm3 = nn.BatchNorm1d(hidden_size2)
        self.dropout3 = nn.Dropout(dropout_ratio)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size2, output_size))

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.leaky_relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = F.sigmoid(self.dense3(x))

        return x


class MoaModelMSDropout(nn.Module):
    def __init__(self, input_size, hidden_size1=2048, hidden_size2=2048, output_size=206,
                 dropout_ratio1=0.05, dropout_ratio2=0.05, dropout_ratio3=0.5):

        super(MoaModelMSDropout, self).__init__()

        self.batch_norm1 = nn.BatchNorm1d(input_size)
        self.dropout1 = nn.Dropout(dropout_ratio1)
        self.dense1 = nn.Linear(input_size, hidden_size1)
        nn.init.kaiming_normal_(self.dense1.weight, mode='fan_in', nonlinearity='leaky_relu')

        self.batch_norm2 = nn.BatchNorm1d(hidden_size1)
        self.dropout2 = nn.Dropout(dropout_ratio2)
        self.dense2 = nn.Linear(hidden_size1, hidden_size2)
        nn.init.kaiming_normal_(self.dense2.weight, mode='fan_in', nonlinearity='leaky_relu')

        self.batch_norm3 = nn.BatchNorm1d(hidden_size2)
        self.dropout3 = nn.Dropout(dropout_ratio3)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size2, output_size))

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.leaky_relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = torch.stack([self.dense3(self.dropout3(x)) for _ in range(5)], dim=0)
        x = torch.mean(x, dim=0)

        x = F.sigmoid(x)

        return x


class MoaDeeperModel(nn.Module):
    def __init__(self, input_size, hidden_size1=2048, hidden_size2=1024, hidden_size3=512, output_size=206,
                 dropout_ratio1=0.05, dropout_ratio2=0.05, dropout_ratio3=0.05, last_dropout_ratio=0.5):

        super(MoaDeeperModel, self).__init__()

        self.batch_norm1 = nn.BatchNorm1d(input_size)
        self.dropout1 = nn.Dropout(dropout_ratio1)
        self.dense1 = nn.Linear(input_size, hidden_size1)
        nn.init.kaiming_normal_(self.dense1.weight, mode='fan_in', nonlinearity='leaky_relu')

        self.batch_norm2 = nn.BatchNorm1d(hidden_size1)
        self.dropout2 = nn.Dropout(dropout_ratio2)
        self.dense2 = nn.Linear(hidden_size1, hidden_size2)
        nn.init.kaiming_normal_(self.dense2.weight, mode='fan_in', nonlinearity='leaky_relu')

        self.batch_norm3 = nn.BatchNorm1d(hidden_size2)
        self.dropout3 = nn.Dropout(dropout_ratio3)
        self.dense3 = nn.Linear(hidden_size2, hidden_size3)
        nn.init.kaiming_normal_(self.dense3.weight, mode='fan_in', nonlinearity='leaky_relu')

        self.batch_norm4 = nn.BatchNorm1d(hidden_size3)
        self.dropout4 = nn.Dropout(last_dropout_ratio)
        self.dense4 = nn.utils.weight_norm(nn.Linear(hidden_size3, output_size))

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.leaky_relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = F.leaky_relu(self.dense3(x))

        x = self.batch_norm4(x)
        x = torch.stack([self.dense4(self.dropout4(x)) for _ in range(5)], dim=0)
        x = torch.mean(x, dim=0)

        x = F.sigmoid(x)

        return x


class MoaDeeperModel2(nn.Module):
    def __init__(self, input_size, hidden_size1=2048, hidden_size2=1024, hidden_size3=512, hidden_size4=256, output_size=206,
                 dropout_ratio1=0.05, dropout_ratio2=0.05, dropout_ratio3=0.05, dropout_ratio4=0.05, last_dropout_ratio=0.5):

        super(MoaDeeperModel2, self).__init__()

        self.batch_norm1 = nn.BatchNorm1d(input_size)
        self.dropout1 = nn.Dropout(dropout_ratio1)
        self.dense1 = nn.Linear(input_size, hidden_size1)
        nn.init.kaiming_normal_(self.dense1.weight, mode='fan_in', nonlinearity='leaky_relu')

        self.batch_norm2 = nn.BatchNorm1d(hidden_size1)
        self.dropout2 = nn.Dropout(dropout_ratio2)
        self.dense2 = nn.Linear(hidden_size1, hidden_size2)
        nn.init.kaiming_normal_(self.dense2.weight, mode='fan_in', nonlinearity='leaky_relu')

        self.batch_norm3 = nn.BatchNorm1d(hidden_size2)
        self.dropout3 = nn.Dropout(dropout_ratio3)
        self.dense3 = nn.Linear(hidden_size2, hidden_size3)
        nn.init.kaiming_normal_(self.dense3.weight, mode='fan_in', nonlinearity='leaky_relu')

        self.batch_norm4 = nn.BatchNorm1d(hidden_size3)
        self.dropout4 = nn.Dropout(dropout_ratio4)
        self.dense4 = nn.Linear(hidden_size3, hidden_size4)
        nn.init.kaiming_normal_(self.dense4.weight, mode='fan_in', nonlinearity='leaky_relu')

        self.batch_norm5 = nn.BatchNorm1d(hidden_size4)
        self.dropout5 = nn.Dropout(last_dropout_ratio)
        self.dense5 = nn.utils.weight_norm(nn.Linear(hidden_size4, output_size))

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.leaky_relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = F.leaky_relu(self.dense3(x))

        x = self.batch_norm4(x)
        x = self.dropout4(x)
        x = F.leaky_relu(self.dense4(x))

        x = self.batch_norm5(x)
        x = torch.stack([self.dense5(self.dropout5(x)) for _ in range(5)], dim=0)
        x = torch.mean(x, dim=0)

        x = F.sigmoid(x)

        return x
