from data_utils import nn_seq_us
import torch

class Args:
    def __init__(self):
        self.epochs = 300
        self.feature_size = 10
        self.hidden_size = 256
        self.lr = 0.005
        self.nhead = 2
        self.num_layer = 6
        self.dropout = 0.1
        self.weight_decay = 1e-6
        self.batch_size = 64
        self.input_seq = 10
        self.output_seq = 1
        self.step_size = 50
        self.gamma = 0.8
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.file_name = "data.xlsx"
        self.path_encoder = "best_encoder.pth"
        self.path_decoder = "best_decoder.pth"


if __name__ == "__main__":
    args = Args()

    # data
    tra, Val, tes, mean, std = nn_seq_us(args.batch_size, args.file_name, args.input_seq, args.output_seq)

    # train
    # train(args, tra, Val)

    # test
    # test(args, tes, args.path_encoder, args.path_decoder, mean, std)



