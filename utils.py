import numpy as np 
import torch
import torch.nn as nn


def MAPE(y_true, y_pred):
    return 100 * np.mean(np.abs(y_true - y_pred) / np.abs(y_true))


def sMAPE(y_true, y_pred):
    return 200 * np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))


def run_gbdt(model, train_X, train_y, val_X, val_y, test_X):
    model.fit(train_X, train_y, eval_set=[(val_X, val_y)], early_stopping_rounds=200)
    return model.predict(test_X)


def run_nbeats(model, train_loader, val_X, val_y, test_X, test_y, fp, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
    criterion = nn.MSELoss().to(device)

    for epoch in range(400):
        model.train()
        for idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            _, forecast = model(batch_X)
            loss = criterion(batch_y, forecast.squeeze())

            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            _, forecast = model(val_X)
            loss = criterion(val_y.to(device), forecast.squeeze())
            print(f'epoch {epoch:03d} loss: {loss.item()}')

    model.eval()
    with torch.no_grad():
        _, test_y_hat = model(test_X)
        test_y, test_y_hat = torch.expm1(test_y), torch.expm1(test_y_hat.squeeze().to(torch.device('cpu')))
        test_y, test_y_hat = test_y.numpy(), test_y_hat.numpy()
        fp.write(f'nbeats >> MAPE: {MAPE(test_y, test_y_hat):.5f}, sMAPE: {sMAPE(test_y, test_y_hat):.5f}\n')
    
    return test_y_hat


def run_rnn(model, train_loader, val_X, val_y, test_X, test_y, fp, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    criterion = nn.MSELoss().to(device)

    for epoch in range(1000):
        model.train()
        for idx, (batch_X, batch_y) in enumerate(train_loader):
            batch_X = batch_X.unsqueeze(-1).permute(1, 0, 2).to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            forecast = model(batch_X)
            loss = criterion(batch_y, forecast.squeeze())

            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            forecast = model(val_X.unsqueeze(-1).permute(1, 0, 2).to(device))
            loss = criterion(val_y.to(device), forecast.squeeze())
            print(f'epoch {epoch:03d} loss: {loss.item()}')

    model.eval()
    with torch.no_grad():
        test_y_hat = model(test_X.unsqueeze(-1).permute(1, 0, 2).to(device))
        test_y, test_y_hat = torch.expm1(test_y), torch.expm1(test_y_hat.squeeze().to(torch.device('cpu')))
        test_y, test_y_hat = test_y.numpy(), test_y_hat.numpy()
        fp.write(f'rnn >> MAPE: {MAPE(test_y, test_y_hat):.5f}, sMAPE: {sMAPE(test_y, test_y_hat):.5f}\n')

    return test_y_hat