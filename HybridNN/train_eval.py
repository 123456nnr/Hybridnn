import torch
import torch.nn as nn
import numpy as np
import copy
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from itertools import chain
from tqdm import tqdm
from models import Trans, Decoder
from data_utils import output_sequence, save_to_excel
from scipy.interpolate import make_interp_spline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def get_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def get_val_loss(args, encoder, decoder, val):
    encoder.eval(); decoder.eval()
    loss_function = nn.MSELoss().to(args.device)
    val_loss = []
    for (seq, label) in val:
        with torch.no_grad():
            seq = seq.to(args.device)
            label = label.to(args.device)
            data_y = output_sequence(seq, args.output_seq).to(args.device)
            encoder_output = encoder(seq)
            y_pred = decoder(encoder_output, data_y)
            loss = loss_function(y_pred, label)
            val_loss.append(loss.item())
    return np.mean(val_loss)


def train(args, Dtr, Val):
    encoder = Trans(args.feature_size, args.nhead, args.num_layer, args.dropout).to(args.device)
    decoder = Decoder(args.hidden_size, args.hidden_size, args.output_seq, args.feature_size).to(args.device)

    loss_function = nn.MSELoss().to(device)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    encoder_scheduler = StepLR(encoder_optimizer, step_size=args.step_size, gamma=args.gamma)
    decoder_scheduler = StepLR(decoder_optimizer, step_size=args.step_size, gamma=args.gamma)

    min_val_loss = float('inf')
    best_encoder, best_decoder = None, None
    train_loss_history, val_loss_history = [], []

    for epoch in range(args.epochs):
        train_loss = []
        encoder.train(); decoder.train()
        for (seq, label) in Dtr:
            seq, label = seq.to(device), label.to(device)
            data_y = output_sequence(seq, args.output_seq).to(args.device)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            encoder_output = encoder(seq)
            y_pred = decoder(encoder_output, data_y)
            loss = loss_function(y_pred, label)
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            train_loss.append(loss.item())
        encoder_scheduler.step(); decoder_scheduler.step()

        val_loss = get_val_loss(args, encoder, decoder, Val)
        train_loss_history.append(np.mean(train_loss))
        val_loss_history.append(val_loss)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_encoder, best_decoder = copy.deepcopy(encoder), copy.deepcopy(decoder)
            print(f"Epoch {epoch} Train Loss {np.mean(train_loss):.6f} Val Loss {val_loss:.6f}")

    torch.save({'encoder': best_encoder.state_dict()}, args.path_encoder)
    torch.save({'decoder': best_decoder.state_dict()}, args.path_decoder)

    plt.plot(train_loss_history, label="Train Loss")
    plt.plot(val_loss_history, label="Validation Loss")
    plt.legend(); plt.show()


def test(args, Dte, path_encoder, path_decoder, mean, std):
    pred = []
    y = []
    print('loading models...')
    encoder = Trans(args.feature_size, args.nhead, args.num_layer, args.dropout).to(args.device)
    decoder = Decoder(code_hidden_size=args.hidden_size, hidden_size=args.hidden_size, time_step=args.output_seq,
                          feature_size=args.feature_size).to(args.device)
    encoder.load_state_dict(torch.load(path_encoder)['encoder'])
    decoder.load_state_dict(torch.load(path_decoder)['decoder'])
    encoder.eval()
    decoder.eval()
    print('predicting...')
    for (seq, target) in tqdm(Dte):
        target = list(chain.from_iterable(target.data.tolist()))
        y.extend(target)
        seq = seq.to(device)
        data_y = output_sequence(seq, args.output_seq).to(args.device)

        with torch.no_grad():
            encoder_output = encoder(seq)
            y_pred = decoder(encoder_output, data_y)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)

    y = np.array(y).reshape(-1, 10)
    pred = np.array(pred).reshape(-1, 10)

    mean = mean.values.reshape(1, -1)
    std = std.values.reshape(1, -1)
    y = y * std + mean
    pred = pred * std + mean
    save_to_excel(y, "data_y_real.xlsx")
    save_to_excel(pred, "data_y_pred.xlsx")
    print('mape:', get_mape(y, pred))
    print('rmse:', get_rmse(y, pred))
    x = [i for i in range(1, 201)]
    x_smooth = np.linspace(np.min(x), np.max(x), 200)


    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 20))
    axes = axes.flatten()
    for i in range(10):
        y_smooth = make_interp_spline(x, y[150:350, i])(x_smooth)
        pred_smooth = make_interp_spline(x, pred[150:350, i])(x_smooth)

        axes[i].plot(x_smooth, y_smooth, c='green', marker='o', ms=5, alpha=1, label='true')
        axes[i].plot(x_smooth, pred_smooth, c='red', marker='o', ms=5, alpha=1, label='pred')
        axes[i].set_title(f'Feature {i + 1}')
        axes[i].grid(axis='y')
        axes[i].legend()


    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 20))
    axes = axes.flatten()
    for i in range(10):
        y_smooth_back = make_interp_spline(x, y[-250:-50, i])(x_smooth)
        pred_smooth_back = make_interp_spline(x, pred[-250:-50, i])(x_smooth)

        axes[i].plot(x_smooth, y_smooth_back, c='green', marker='o', ms=5, alpha=1, label='true')
        axes[i].plot(x_smooth, pred_smooth_back, c='red', marker='o', ms=5, alpha=1, label='pred')
        axes[i].set_title(f'Feature {i + 1}')
        axes[i].grid(axis='y')
        axes[i].legend()

    plt.show()