import torch.nn as nn
import torch
from torch.autograd import Variable
import argparse
import numpy as np
import torch.nn.functional as F
from util.graph_definition import *

class selfAttn(nn.Module):
    def __init__(self, feature_size, hidden_size, num_desc):
        super(selfAttn, self).__init__()
        self.linear_1 = nn.Linear(feature_size, hidden_size, bias=False)
        self.linear_2 = nn.Linear(hidden_size, num_desc, bias=False)
        self.num_desc = num_desc
        self.bn = nn.BatchNorm1d(feature_size)

    def forward(self, model_input):   # (batch_size, time_step, feature_size)
        reshaped_input = model_input
        s1 = F.tanh(self.linear_1(reshaped_input))  # (batch_size, time_step, hidden_size)
        A = F.softmax(self.linear_2(s1), dim=1)
        M = self.bn(torch.bmm(model_input.permute(0, 2, 1), A)).permute(0, 2, 1).contiguous()  # (batch_size, num_desc, feature_size)
        AAT = torch.bmm(A.permute(0, 2, 1), A)
        I = Variable(torch.eye(self.num_desc)).cuda()
        P = torch.norm(AAT - I, 2)
        penal = P * P / model_input.shape[0]
        return M, penal


class conv_lstm(nn.Module):
    def __init__(self, hidden_size, kernel, stride, nb_filter, input_size):
        super(conv_lstm, self).__init__()
        self.conv = nn.Sequential(nn.Conv1d(input_size, nb_filter, kernel, stride),
                                nn.ReLU(),
                                nn.BatchNorm1d(nb_filter)
        )
        self.lstm = create_model(model='skip_lstm',
                             input_size=nb_filter,
                             hidden_size=hidden_size,
                             num_layers=1)
        #self.lstm = lstm_cell(input_size=nb_filter, hidden_size=hidden_size, batch_first=True, layer_norm=True)
        self.hidden_size = hidden_size

    def forward(self, input):
        input = self.conv(input.permute(0, 2, 1))
        input = input.permute(0, 2, 1)
        output = self.lstm(input)
        output, hx, updated_state = split_rnn_outputs('skip_lstm', output)
        return output[:, -1, :]

class Scoring(nn.Module):
    def __init__(self, feature_size):
        super(Scoring, self).__init__()

        conv_input = 128
        self.conv = nn.Sequential(
                        nn.Conv1d(feature_size, conv_input, 1, 1),
                        nn.ReLU(),
                        nn.BatchNorm1d(conv_input)
        )
        hidden_size = 256
        self.scale1 = conv_lstm(hidden_size, 2, 1, 256, conv_input)
        self.scale2 = conv_lstm(hidden_size, 4, 2, 256, conv_input)
        self.scale3 = conv_lstm(hidden_size, 8, 4, 256, conv_input)
        self.attn = selfAttn(conv_input, 64, 50)
        self.lstm = nn.LSTM(input_size=conv_input, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.linear_skip1 = nn.Linear(hidden_size, 64)
        self.linear_skip2 = nn.Linear(hidden_size, 64)
        self.linear_skip3 = nn.Linear(hidden_size, 64)
        self.linear_attn = nn.Linear(hidden_size, 64)
        self.linear_merge = nn.Linear(64*4, 64)
        self.cls = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(0.7)

    def forward(self, model_input):
        self.h0 = Variable(torch.randn(1, model_input.shape[0], self.hidden_size))
        self.c0 = Variable(torch.randn(1, model_input.shape[0], self.hidden_size))
        if torch.cuda.is_available():
            self.h0 = self.h0.cuda()
            self.c0 = self.c0.cuda()
        model_input = model_input.permute(0, 2, 1)
        model_input = self.conv.forward(model_input).permute(0, 2, 1)
        penal = None
        attn, penal = self.attn.forward(model_input)
        attn, _ = self.lstm(attn)
        attn = attn[:, -1, :]
        M_output = torch.cat([self.relu(self.linear_skip1(self.scale1(model_input))), self.relu(self.linear_skip2(self.scale2(model_input))), self.relu(self.linear_skip3(self.scale3(model_input)))], 1)
        output = torch.cat([M_output, self.relu(self.linear_attn(attn))], 1)
        #output = torch.cat([self.relu(self.linear_skip1(self.scale1(model_input))), self.relu(self.linear_attn(attn))], 1)
        #output = M_output
        output = self.relu(self.linear_merge(output))
        #output = self.relu(self.linear_attn(attn))
        return self.cls(output), penal

    def loss(self, regression, actuals):
        """
        use mean square error for regression and cross entropy for classification
        """
        regr_loss_fn = nn.MSELoss()
        return regr_loss_fn(regression, actuals)

