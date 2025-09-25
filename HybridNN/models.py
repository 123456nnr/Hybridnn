import torch
import torch.nn as nn
import math
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class Trans(nn.Module):
    def __init__(self, feature_size=64, nhead=2, num_layers=6, dropout=0.1):
        super(Trans, self).__init__()
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=nhead,
                                                        dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, src):  # src: [batch, seq, features]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output


class Decoder(nn.Module):
    def __init__(self, code_hidden_size, hidden_size, time_step, feature_size):
        super(Decoder, self).__init__()
        self.code_hidden_size = code_hidden_size
        self.hidden_size = hidden_size
        self.T = time_step

        self.fc0 = nn.Linear(feature_size, hidden_size)
        self.attn1 = nn.Linear(in_features=2 * hidden_size, out_features=code_hidden_size)
        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=self.hidden_size,
                            num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(in_features=code_hidden_size + hidden_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=feature_size)

    def forward(self, h, y_seq):
        h = self.fc0(h)
        h_ = h.transpose(0, 1)
        batch_size = h.size(0)
        d = self.init_variable(batch_size, self.hidden_size, device=device)
        s = self.init_variable(batch_size, self.hidden_size, device=device)
        h_0 = self.init_variable(batch_size, self.hidden_size, device=device)
        h_ = torch.cat((h_0.unsqueeze(0), h_), dim=0)

        outputs = []
        y_seq = y_seq.view(batch_size, self.T, -1)
        for t in range(self.T):
            x = torch.cat((d.unsqueeze(0), h_[t].unsqueeze(0)), 2)
            h1 = self.attn1(x).permute(1, 0, 2).contiguous()
            lstm_input = y_seq[:, t].unsqueeze(1)
            _, (d, s) = self.lstm(lstm_input, (h1.view(1, batch_size, -1), s.view(1, batch_size, -1)))
            d = d.squeeze(0)
            s = s.squeeze(0)
            y_res = self.fc2(self.fc1(torch.cat((d, h_[-1]), dim=1)))
            outputs.append(y_res)
        outputs = torch.stack(outputs, dim=1)
        return outputs

    def init_variable(self, *args, device):
        return Variable(torch.zeros(args).to(device))
