from torch import nn
# from torchsummary import summary

class DwConv(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()

        norm_layer = nn.BatchNorm2d
        self.dw1 = nn.Conv2d(
            inputs, inputs,
            kernel_size=3,
            padding='same',
            groups=inputs
        )
        self.bn1 = norm_layer(inputs)
        self.relu = nn.ReLU()
        self.pw = nn.Conv2d(
            inputs, outputs,
            kernel_size=1
        )
        self.bn2 = norm_layer(outputs)


    def forward(self, x):
        out = self.dw1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pw(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out
        
# # Проверка  
# block = DwConv(3, 64)
# summary(block, input_size=(3, 224, 224),  device = 'cpu')