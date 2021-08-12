from torch import nn
from torchsummary import summary

# todo: this is value from Valerio Tutorial. has to check
# LINEAR_IN_CHANNELS = 128 * 5 * 4
LINEAR_IN_CHANNELS = 4480


class SynthNetwork(nn.Module):
    """
    CNN model to extract parameters from a signal mek-spectrogram

    :return: output_dic, that hold the logits for classification parameters and predictions for regression parameters
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.classification_params = nn.ModuleDict([
            ['osc1_freq', nn.Linear(LINEAR_IN_CHANNELS, 49)],
            ['osc1_wave', nn.Linear(LINEAR_IN_CHANNELS, 4)],
            ['lfo1_wave', nn.Linear(LINEAR_IN_CHANNELS, 4)],
            ['osc2_freq', nn.Linear(LINEAR_IN_CHANNELS, 49)],
            ['osc2_wave', nn.Linear(LINEAR_IN_CHANNELS, 4)],
            ['lfo2_wave', nn.Linear(LINEAR_IN_CHANNELS, 4)],
            ['filter_type', nn.Linear(LINEAR_IN_CHANNELS, 3)],
        ])
        self.regression_params = nn.Linear(LINEAR_IN_CHANNELS, 14)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        output_dic = {}
        for out_name, lin in self.classification_params.items():
            # -----> Shall not use softmax since with use CrossEntropyLoss()
            # predictions_dic[out_name] = self.softmax(lin(x))
            output_dic[out_name] = lin(x)
        output_dic['regression_params'] = self.regression_params(x)

        return output_dic


if __name__ == "__main__":
    synth_net = SynthNetwork()
    summary(synth_net.cuda(), (1, 64, 44))


