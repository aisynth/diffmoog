from torch import nn
from torchsummary import summary
from synth import REGRESSION_PARAM_LIST, CLASSIFICATION_PARAM_LIST, WAVE_TYPE_DIC, FILTER_TYPE_DIC, OSC_FREQ_LIST

# todo: this is value from Valerio Tutorial. has to check
# LINEAR_IN_CHANNELS = 128 * 5 * 4
LINEAR_IN_CHANNELS = 4480
HIDDEN_IN_CHANNELS = 1000


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
            ['osc1_freq', nn.Linear(LINEAR_IN_CHANNELS, len(OSC_FREQ_LIST))],
            ['osc1_wave', nn.Linear(LINEAR_IN_CHANNELS, len(WAVE_TYPE_DIC))],
            ['osc2_freq', nn.Linear(LINEAR_IN_CHANNELS, len(OSC_FREQ_LIST))],
            ['osc2_wave', nn.Linear(LINEAR_IN_CHANNELS, len(WAVE_TYPE_DIC))],
            ['filter_type', nn.Linear(LINEAR_IN_CHANNELS, len(FILTER_TYPE_DIC))],
        ])
        self.regression_params = nn.Sequential(
            nn.Linear(LINEAR_IN_CHANNELS, HIDDEN_IN_CHANNELS),
            nn.Linear(HIDDEN_IN_CHANNELS, len(REGRESSION_PARAM_LIST))
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        output_dic = {}
        for out_name, lin in self.classification_params.items():
            # -----> do not use softmax if using CrossEntropyLoss()
            output_dic[out_name] = self.softmax(lin(x))
            # output_dic[out_name] = lin(x)
        output_dic['regression_params'] = self.regression_params(x)

        return output_dic


if __name__ == "__main__":
    synth_net = SynthNetwork()
    summary(synth_net.cuda(), (1, 64, 44))


