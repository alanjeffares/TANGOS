from functorch import jacrev
from functorch import vmap
from torch import optim
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401,E402
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import json

d = pd.read_csv('path/to/data/dionis', header = None)
TRAINING_RATIO = 0.1  # change this for different ratios of training data

y = d.iloc[:,0]
X = d.iloc[:, 1:]


SEED = 0
BATCH_SIZE = 256
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)


num = int(X_train.shape[0] * TRAINING_RATIO)
X_train = X_train[:num]
y_train = y_train[:num]

scaler_train = StandardScaler()
X_train = scaler_train.fit_transform(X_train)
X_val = scaler_train.transform(X_val)
X_test = scaler_train.transform(X_test)

train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train.to_numpy()))
val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(y_val.to_numpy()))
test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test.to_numpy()))
loaders = {
    'train': DataLoader(train_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=1),

    'val': DataLoader(val_dataset,
                      batch_size=BATCH_SIZE,
                      shuffle=False,
                      num_workers=1),

    'test': DataLoader(test_dataset,
                       batch_size=int(BATCH_SIZE),
                       shuffle=False,
                       num_workers=1)
}


class UCI_MLP(nn.Module):
    def __init__(self, num_features, num_outputs, dropout=0, batch_norm=False):
        super(UCI_MLP, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.batch_norm = batch_norm
        d = num_features + 1
        self.fc1 = nn.Linear(num_features, 400)
        self.bn1 = nn.BatchNorm1d(d)
        self.relu1 = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(400, 100)
        self.bn2 = nn.BatchNorm1d(d)
        self.relu2 = nn.ReLU(inplace=False)
        self.fc3 = nn.Linear(100, 10)
        self.relu3 = nn.ReLU(inplace=False)
        self.fc4 = nn.Linear(10, num_outputs)

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.fc1(x)
        if self.batch_norm and batch_size > 1:
            out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        if self.batch_norm and batch_size > 1:
            out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.fc3(out)
        h_output = self.relu3(out)
        out = self.fc4(h_output)
        return out, h_output


def attr_loss(forward_func, data_input, device='cpu', subsample=-1):
    ########## UPDATE functools ############
    batch_size = data_input.shape[0]

    def test(input_):
        _, h_out = forward_func(input_)
        return h_out

    data_input = data_input.clone().requires_grad_(True)
    jacobian = vmap(jacrev(test))(data_input)
    neuron_attr = jacobian.swapaxes(0, 1)
    h_dim = neuron_attr.shape[0]

    if len(neuron_attr.shape) > 3:
        # h_dim x batch_size x features
        neuron_attr = neuron_attr.flatten(start_dim=2)

    sparsity_loss = torch.norm(neuron_attr, p=1) / (batch_size * h_dim * neuron_attr.shape[2])

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    correlation_loss = torch.tensor(0., requires_grad=True).to(device)
    if subsample > 0 and subsample < h_dim * (h_dim - 1) / 2:
        tensor_pairs = [list(np.random.choice(h_dim, size=(2), replace=False)) for i in range(subsample)]
        for tensor_pair in tensor_pairs:
            pairwise_corr = cos(neuron_attr[tensor_pair[0], :, :], neuron_attr[tensor_pair[1], :, :]).norm(p=1)
            correlation_loss = correlation_loss + pairwise_corr

        correlation_loss = correlation_loss / (batch_size * subsample)

    else:
        for neuron_i in range(1, h_dim):
            for neuron_j in range(0, neuron_i):
                pairwise_corr = cos(neuron_attr[neuron_i, :, :], neuron_attr[neuron_j, :, :]).norm(p=1)
                correlation_loss = correlation_loss + pairwise_corr
        num_pairs = h_dim * (h_dim - 1) / 2
        correlation_loss = correlation_loss / (batch_size * num_pairs)

    return sparsity_loss, correlation_loss


def train_epoch(model, loader, loss_func, optimiser, epoch,
                lambda_1=0, lambda_2=0, device='cpu', subsample=-1):
    running_loss = 0
    for i, (data, label) in enumerate(loader):
        model.train()
        data, label = data.to(device), label.type(torch.LongTensor).to(device)
        optimiser.zero_grad()
        output, _ = model(data)

        pred_loss = loss_func(output.squeeze(), label)

        if lambda_1 + lambda_2 > 0:
            sparsity_loss, correlation_loss = attr_loss(model, data, device=device, subsample=subsample)
        else:
            sparsity_loss, correlation_loss = 0, 0

        loss = pred_loss + lambda_1 * sparsity_loss + lambda_2 * correlation_loss
        running_loss += loss.item()

        loss.backward()
        optimiser.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, i + 1, len(loader), running_loss / (i + 1)))
            print(f"Lambda1: {lambda_1}, Lambda2: {lambda_2}")

    return model


def evaluate(model, loader, loss_func, epoch,
             lambda_1=0, lambda_2=0, device='cpu', subsample=-1, log_set='test'):
    correct, total = 0, 0
    running_loss, running_pred_loss = 0, 0
    running_pred, running_gt = np.array([]), np.array([])

    for i, (data, label) in enumerate(loader):
        model.eval()
        data, label = data.to(device), label.type(torch.LongTensor).to(device)

        output, _ = model(data)
        pred_loss = loss_func(output.squeeze(), label)

        sparsity_loss, correlation_loss = attr_loss(model, data, device=device, subsample=subsample)

        loss = pred_loss + lambda_1 * sparsity_loss + lambda_2 * correlation_loss

        running_loss += loss.item()
        running_pred_loss += pred_loss.item()

        pred_probs = torch.sigmoid(output)
        pred_y = torch.argmax(pred_probs, 1)
        correct += (pred_y == label).sum().item()
        total += float(label.size()[0])


    accuracy = correct / total

    average_loss = running_loss / len(loader)
    averge_pred_loss = running_pred_loss / len(loader)

    print(f'[Test] Epoch: {epoch + 1}, accuracy: {accuracy:.4f}, ' \
          f'average test loss: {average_loss:.4f}, ' \
          f'pred loss: {averge_pred_loss:.4f}, ' \
          f'sparsity loss: {sparsity_loss.item():.4f}, correlation loss: {correlation_loss.item():.4f}')

    return averge_pred_loss, accuracy


def train_full(seed, lambda_1, lambda_2, LR=0):
    EPOCHS = 100
    TRAINING_PATIENCE = 5
    BATCH_SIZE = 256
    DEVICE = 'cuda:0'

    runs = 1

    learning_rate = 0.001
    weight_decay = LR
    num_features = 60
    num_outputs = 355
    subsample = 50
    model_save_path = 'model_weights'
    min_epoch = 1
    best_acc = 0
    accuracy_val_ls = []

    for _ in range(runs):
        torch.random.manual_seed(seed)

        model = UCI_MLP(num_features, num_outputs, dropout=0, batch_norm=False).to(DEVICE)
        print(f'Training on {DEVICE}...')

        loss_func = nn.CrossEntropyLoss()
        optimiser = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        patience = 0

        for epoch in range(EPOCHS):

            model = train_epoch(model, loaders['train'], loss_func, optimiser,
                                epoch=epoch, lambda_1=lambda_1, lambda_2=lambda_2,
                                device=DEVICE, subsample=subsample)

            val_loss, accuracy = evaluate(model, loaders['val'], loss_func, epoch=epoch,
                                          lambda_1=lambda_1, lambda_2=lambda_2, device=DEVICE, subsample=subsample)
            accuracy_val_ls.append(accuracy)

            if epoch >= min_epoch:
                if best_acc < accuracy:
                    print(f'Epoch {epoch + 1} - Validation performance improved, saving model...')
                    best_acc = accuracy
                    torch.save(model.state_dict(), model_save_path)
                    patience = 0
                else:
                    patience += 1

            if patience == TRAINING_PATIENCE:
                print(f'Epoch {epoch + 1} - Early stopping since no improvement after {patience} epochs')
                break

        # evaluate on cutract dataset
        # load best model
        model.load_state_dict(torch.load(model_save_path))
        averge_pred_loss, accuracy = evaluate(model, loaders['test'], loss_func, epoch=0,
                                              lambda_1=lambda_1, lambda_2=lambda_2, device=DEVICE, subsample=subsample,
                                              log_set='target')
        return accuracy



# main logic for training baseline, tangos regularization and l2 regularization
baseline_ls = []
for seed in range(6):
    acc = train_full(seed, 0, 0, LR=0)
    baseline_ls.append(acc)
    with open('src/experiments/larger-data/baseline.json', 'w') as f:
        json.dump({'test_acc': baseline_ls}, f)

TANGOS_ls = []
for seed in range(6):
    acc = train_full(seed, 1, 0.01, LR=0)
    TANGOS_ls.append(acc)
    with open('src/experiments/larger-data/TANGOS.json', 'w') as f:
        json.dump({'test_acc': TANGOS_ls}, f)

l2_ls = []
for seed in range(6):
    acc = train_full(seed, 0, 0, LR=0.0001)
    l2_ls.append(acc)
    with open('src/experiments/larger-data/l2.json', 'w') as f:
        json.dump({'test_acc': l2_ls}, f)

