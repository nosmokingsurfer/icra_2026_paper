import torch
import torch.nn as nn


class SimpleVDRmodel(nn.Module):
    def __init__(self):
        super(SimpleVDRmodel, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(6,64,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        self.downsample1 = nn.Sequential(
            nn.Conv1d(6,64,kernel_size=3, stride=2,padding=1),
            nn.ReLU(inplace=True)
            )

        self.layer2 = nn.Sequential(
            nn.Conv1d(64,64,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        self.downsample2 = nn.Sequential(
            nn.Conv1d(64,64,kernel_size=3, stride=2,padding=1),
            nn.ReLU(inplace=True)
            )

        self.layer3 = nn.Sequential(
            nn.Conv1d(64,64,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )

        self.downsample3 = nn.Sequential(
            nn.Conv1d(64,64,kernel_size=3, stride=2,padding=1),
            nn.ReLU(inplace=True)
            )

        self.lstm = nn.LSTM(64,64,batch_first=True)

        self.regression = nn.Sequential(
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,2),
        )


    def forward(self, X, hidden=None):
        out1 = self.layer1(X)
        out = out1 + self.downsample1(X)

        out2 = self.layer2(out)
        out = out2 + self.downsample2(out)

        out3 = self.layer3(out)
        out = out3 + self.downsample2(out)

        out_lstm, hidden = self.lstm(out.swapaxes(-2,-1), hidden)

        velocity = self.regression(out_lstm)

        return velocity


if __name__ == "__main__":
    model = SimpleVDRmodel()

    # B,C,T
    sample_input = torch.randn((16,6,100))
    output = model(sample_input)

    print(output.shape)

