import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def load_data(file_name):
    df = pd.read_excel(file_name)
    return df


def save_to_excel(data, filename):
    df = pd.DataFrame(data)
    df.to_excel(filename, index=False)


def output_sequence(x, output_seq):
    if output_seq == 1:
        x = x[:, -1, :].unsqueeze(1)
    else:
        x = x[:, -output_seq:, :]
    return x


def nn_seq_us(B, file_name, input_seq, output_seq):
    print('data processing...')
    dataset = load_data(file_name)

    global_mean = dataset.mean(axis=1)
    extreme_point = np.argmin(global_mean)
    print(f'Extreme point index: {extreme_point}')

    mean = dataset.mean()
    std = dataset.std()
    dataset = (dataset - mean) / std

    cooling = dataset[:extreme_point + 1]
    rewarming = dataset[extreme_point + 1:]

    cooling_train = cooling[:int(len(cooling) * 0.6)]
    cooling_val = cooling[int(len(cooling) * 0.6):int(len(cooling) * 0.8)]
    cooling_test = cooling[int(len(cooling) * 0.8):len(cooling)]

    rewarming_train = rewarming[:int(len(rewarming) * 0.6)]
    rewarming_val = rewarming[int(len(rewarming) * 0.6):int(len(rewarming) * 0.8)]
    rewarming_test = rewarming[int(len(rewarming) * 0.8):len(rewarming)]

    train = pd.concat([cooling_train, rewarming_train]).sort_index()
    val = pd.concat([cooling_val, rewarming_val]).sort_index()
    test = pd.concat([cooling_test, rewarming_test]).sort_index()

    def process(data, batch_size, shuffle, input_seq, output_seq):
        data = data.values.tolist()
        seq = []
        for i in range(len(data) - input_seq - output_seq):
            train_seq = data[i:i+input_seq]
            train_label = data[i+input_seq:i+input_seq+output_seq]
            train_seq = torch.FloatTensor(train_seq)
            train_label = torch.FloatTensor(train_label).view(output_seq, -1)
            seq.append((train_seq, train_label))
        seq = MyDataset(seq)
        seq = DataLoader(dataset=seq, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)
        return seq

    tra = process(train, B, True, input_seq, output_seq)
    Val = process(val, B, False, input_seq, output_seq)
    tes = process(test, B, False, input_seq, output_seq)

    return tra, Val, tes, mean, std
