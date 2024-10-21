import torch.nn as nn
import torch
import random
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 1
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(input_seq.device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(input_seq.device)
        output, (h, c) = self.lstm(input_seq, (h_0, c_0))

        return h, c
    
class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=False)
        self.linear = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, input_seq, h, c):
        # input_seq(batch_size, input_size)
        input_seq = input_seq.unsqueeze(1)
        output, (h, c) = self.lstm(input_seq, (h, c))
        # output(batch_size, seq_len, num * hidden_size)
        pred = self.linear(output.squeeze(1))  # pred(batch_size, 1, output_size)

        return pred, h, c
    
class PFM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.Encoder = Encoder(input_size, hidden_size, num_layers, batch_size)
        self.Decoder = Decoder(input_size, hidden_size, num_layers, output_size, batch_size)

    def forward(self, input_seq):
        target_len = self.output_size  # 预测步长
        batch_size, seq_len, _ = input_seq.shape[0], input_seq.shape[1], input_seq.shape[2]
        h, c = self.Encoder(input_seq)
        outputs = torch.zeros(batch_size, self.input_size, self.output_size).to(input_seq.device)
        decoder_input = input_seq[:, -1, :]
        for t in range(target_len):
            decoder_output, h, c = self.Decoder(decoder_input, h, c)
            outputs[:, :, t] = decoder_output
            decoder_input = decoder_output

        return outputs#[:, 0, :]
    
if __name__ == "__main__":
    input = torch.rand(64,336,7)          # [B,L,K]
    pfm = PFM(7,300,32,168,64)
    outputs = pfm(input).permute(0,2,1)    #[B,K,L]->[B,L,K]
    print(1)