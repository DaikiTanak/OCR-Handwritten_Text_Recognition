
class HTRNet(nn.Module):
    def __init__(self, nclasses, in_channels=3, img_size=(32, 128)):
        super(HTRNet, self).__init__()
        
        self.features = nn.Sequential()

        # img_size is (img_width, img_height)
        self.img_size = img_size

        assert img_size[0] in [32, 64]

        pool_kernel_size = [(2,2), (2,2), (2, 1), (2, 1), (2, 1), (2, 1)]
        filter_size = [in_channels, 32, 64, 128, 128, 256, 256]
        conv_kernel_size = [5, 5, 3, 3, 3, 3]

        self.features.add_module("Conv1", nn.Conv2d(filter_size[0], filter_size[1], kernel_size=conv_kernel_size[0], stride=1, padding=2, bias=True))
        self.features.add_module("NormReLU1", nn.Sequential(nn.BatchNorm2d(filter_size[1], momentum=.5), nn.ReLU()))
        self.features.add_module("Pool1", nn.MaxPool2d(kernel_size=pool_kernel_size[0], stride=pool_kernel_size[0]))

        self.features.add_module("Conv2", nn.Conv2d(filter_size[1], filter_size[2], conv_kernel_size[1], 1, 2, bias=True))
        self.features.add_module("NormReLU2", nn.Sequential(nn.BatchNorm2d(filter_size[2], momentum=.5), nn.ReLU()))
        self.features.add_module("Pool2", nn.MaxPool2d(kernel_size=pool_kernel_size[1], stride=pool_kernel_size[1]))

        self.features.add_module("Conv3", nn.Conv2d(filter_size[2], filter_size[3], conv_kernel_size[2], 1, 1, bias=True))
        self.features.add_module("NormReLU3", nn.Sequential(nn.BatchNorm2d(filter_size[3], momentum=.5), nn.ReLU()))
        self.features.add_module("Pool3", nn.MaxPool2d(kernel_size=pool_kernel_size[2], stride=pool_kernel_size[2]))

        self.features.add_module("Conv4", nn.Conv2d(filter_size[3], filter_size[4], conv_kernel_size[3], 1, 1, bias=True))
        self.features.add_module("NormReLU4", nn.Sequential(nn.BatchNorm2d(filter_size[4], momentum=.5), nn.ReLU()))
        self.features.add_module("Pool4", nn.MaxPool2d(kernel_size=pool_kernel_size[3], stride=pool_kernel_size[3]))

        self.features.add_module("Conv5", nn.Conv2d(filter_size[4], filter_size[5], conv_kernel_size[4], 1, 1, bias=True))
        self.features.add_module("NormReLU5", nn.Sequential(nn.BatchNorm2d(filter_size[5], momentum=.5), nn.ReLU()))
        self.features.add_module("Pool5", nn.MaxPool2d(kernel_size=pool_kernel_size[4], stride=pool_kernel_size[4]))

        if img_size == (64, 256):
            self.features.add_module("Conv6", nn.Conv2d(filter_size[5], filter_size[6], conv_kernel_size[5], 1, 1, bias=True))
            self.features.add_module("NormReLU6", nn.Sequential(nn.BatchNorm2d(filter_size[6], momentum=.5), nn.ReLU()))
            self.features.add_module("Pool6", nn.MaxPool2d(kernel_size=pool_kernel_size[5], stride=pool_kernel_size[5]))

        # rnn_in equals to kernel size of final conv layer.
        rnn_in = 256

        # RNN Params
        hidden = 256
        num_layers = 2

        self.lstm = nn.LSTM(input_size=rnn_in,
                            hidden_size=hidden,
                            num_layers=num_layers,
                            bidirectional=True,
                            batch_first=True)

        self.linear = nn.Sequential(nn.Linear(2*hidden, 512),
                                 nn.ReLU(),
                                 # nn.Dropout(.5),
                                 nn.Linear(512, nclasses))

    def forward(self, img):

        x = self.features(img)
        # x :: (batch, kernel_size(256), 1, feature_num)
        
        assert x.size(2) == 1
        
        x = torch.squeeze(x, dim=2).permute(0, 2, 1)  # batch_size, seq_length, feat

        # output of lstm is (batchsize, seq_len, feat*2)
        x = self.lstm(x)[0]

        y = self.linear(x)
        # output of FC layer is (batch, seq_len, num_classes)

        return y
