import torch.nn as nn
from torch.nn import functional as F


class SegNet(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()
        C_in, H_in, W_in = params["input_shape"]
        init_f = params["initial_filters"]
        num_outputs = params["num_outputs"]

        self.conv1 = nn.Conv2d(C_in, init_f, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(init_f, 2 * init_f, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(2 * init_f, 4 * init_f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(4 * init_f, 8 * init_f, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(8 * init_f, 16 * init_f, kernel_size=3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv_up1 = nn.Conv2d(16 * init_f, 8 * init_f, kernel_size=3, padding=1)
        self.conv_up2 = nn.Conv2d(8 * init_f, 4 * init_f, kernel_size=3, padding=1)
        self.conv_up3 = nn.Conv2d(4 * init_f, 2 * init_f, kernel_size=3, padding=1)
        self.conv_up4 = nn.Conv2d(2 * init_f, init_f, kernel_size=3, padding=1)

        self.conv_out = nn.Conv2d(init_f, num_outputs, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv5(x))

        x = self.upsample(x)
        x = F.relu(self.conv_up1(x))

        x = self.upsample(x)
        x = F.relu(self.conv_up2(x))

        x = self.upsample(x)
        x = F.relu(self.conv_up3(x))

        x = self.upsample(x)
        x = F.relu(self.conv_up4(x))

        x = self.conv_out(x)

        # NOTE: Observe that we are not using Sigmoid activation in the output mainly because
        # the loss is partly calculate with BCEWITHLOGITSLOSS. This loss function already
        # includes a sigmoid within it. So we avoid using sigmoid here, but we have to
        # manually apply sigmoid function to our prediction when calculating the second part of our loss,
        # which is the dice loss.
        return x


if __name__ == "__main__":
    import torch
    from torchsummary import summary

    h, w = 128, 192

    params_model = {"input_shape": (1, h, w), "initial_filters": 16, "num_outputs": 1}

    model = SegNet(params_model)

    # move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # UNCOMMENT THE FOLLOWING TO ENFORCE CPU-USE ONLY
    # device = torch.device("cpu")
    model = model.to(device)

    print(model)

    summary(model, input_size=params_model["input_shape"], device=device.type)
