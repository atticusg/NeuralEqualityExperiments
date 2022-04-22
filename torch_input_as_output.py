import sys
import torch
import torch.nn as nn
from utils import progress_bar


class InputAsOutputModule(nn.Module):
    def __init__(self, input_dim, output_dim=2, activation_class=nn.ReLU):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = int(input_dim  / 2)
        self.output_dim = output_dim
        self.input_layer = nn.Linear(self.input_dim, self.embed_dim)
        self.activation = activation_class()
        self.output_layer = nn.Linear(self.embed_dim, self.output_dim)

    def forward(self, X):
        # This is the full Premack condition:
        if len(X.shape) == 3:
            X_left = X[:, 0, : ]
            X_right = X[:, 1, : ]
            h_left = self.activation(self.input_layer(X_left))
            h_right = self.activation(self.input_layer(X_right))
            h = torch.cat((h_left, h_right), dim=1)
            h = self.activation(self.input_layer(h))
            logits = self.output_layer(h)
            return logits
        # Regular equality:
        else:
            h = self.activation(self.input_layer(X))
            logits = self.output_layer(h)
            return logits


class InputAsOutputClassifier:
    def __init__(self,
            output_dim=2,
            activation_class=nn.ReLU,
            warm_start=True,
            alpha=0.0,
            batch_size=128,
            eta=0.01,
            max_iter=100,
            device=None):
        self.output_dim = output_dim
        self.activation_class = activation_class
        self.batch_size = batch_size
        self.eta = eta
        self.alpha = alpha
        self.max_iter = max_iter
        self.warm_start = warm_start
        self.optimizer_func = torch.optim.Adam
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def fit(self, X, y):
        self.input_dim = X.shape[-1]
        if not self.warm_start or not hasattr(self, "model"):
            self.build_graph()
        X = torch.FloatTensor(X)
        y = torch.tensor(y)
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True)
        loss = nn.CrossEntropyLoss()
        self.model.to(self.device)
        self.model.train()
        for iteration in range(1, self.max_iter+1):
            epoch_error = 0.0
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                preds = self.model(X_batch)
                err = loss(preds, y_batch)
                epoch_error += err.item()
                self.optimizer.zero_grad()
                err.backward()
                self.optimizer.step()
            if len(X.shape) == 3:
                msg_prefix = "Pretraining - finished"
            else:
                msg_prefix = "Finished"
            progress_bar(
                "{} epoch {} of {}; error is {}".format(
                    msg_prefix, iteration, self.max_iter, epoch_error))

    def build_graph(self):
        self.model = InputAsOutputModule(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            activation_class=self.activation_class)
        self.optimizer = self.optimizer_func(
            self.model.parameters(),
            lr=self.eta,
            weight_decay=self.alpha)

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            self.model.to(self.device)
            X = torch.FloatTensor(X).to(self.device)
            preds = self.model(X)
            return torch.softmax(preds, dim=1).cpu().numpy()

    def predict(self, X):
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)
