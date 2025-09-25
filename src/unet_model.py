import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, n_classes):
        super(UNet, self).__init__()
        self.el1 = nn.Conv2d(3, 64, 3, padding=1)
        self.el2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.el3 = nn.Conv2d(64, 128, 3, padding=1)
        self.el4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.el5 = nn.Conv2d(128, 256, 3, padding=1)
        self.el6 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.el7 = nn.Conv2d(256, 512, 3, padding=1)
        self.el8 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.el9 = nn.Conv2d(512, 1024, 3, padding=1)
        self.el10 = nn.Conv2d(1024, 1024, 3, padding=1)

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dl1 = nn.Conv2d(1024, 512, 3, padding=1)
        self.dl2 = nn.Conv2d(512, 512, 3, padding=1)

        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dl3 = nn.Conv2d(512, 256, 3, padding=1)
        self.dl4 = nn.Conv2d(256, 256, 3, padding=1)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dl5 = nn.Conv2d(256, 128, 3, padding=1)
        self.dl6 = nn.Conv2d(128, 128, 3, padding=1)

        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dl7 = nn.Conv2d(128, 64, 3, padding=1)
        self.dl8 = nn.Conv2d(64, 64, 3, padding=1)

        self.out = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        xe11 = torch.relu(self.el1(x))
        xe12 = torch.relu(self.el2(xe11))
        pool1 = self.pool1(xe12)

        xe21 = torch.relu(self.el3(pool1))
        xe22 = torch.relu(self.el4(xe21))
        pool2 = self.pool2(xe22)

        xe31 = torch.relu(self.el5(pool2))
        xe32 = torch.relu(self.el6(xe31))
        pool3 = self.pool3(xe32)

        xe41 = torch.relu(self.el7(pool3))
        xe42 = torch.relu(self.el8(xe41))
        pool4 = self.pool4(xe42)

        xe51 = torch.relu(self.el9(pool4))
        xe52 = torch.relu(self.el10(xe51))

        xu1 = self.up1(xe52)
        xu1 = torch.cat((xu1, xe42), dim=1)
        xu11 = torch.relu(self.dl1(xu1))
        xu12 = torch.relu(self.dl2(xu11))

        xu2 = self.up2(xu12)
        xu2 = torch.cat((xu2, xe32), dim=1)
        xu21 = torch.relu(self.dl3(xu2))
        xu22 = torch.relu(self.dl4(xu21))

        xu3 = self.up3(xu22)
        xu3 = torch.cat((xu3, xe22), dim=1)
        xu31 = torch.relu(self.dl5(xu3))
        xu32 = torch.relu(self.dl6(xu31))

        xu4 = self.up4(xu32)
        xu4 = torch.cat((xu4, xe12), dim=1)
        xu41 = torch.relu(self.dl7(xu4))
        xu42 = torch.relu(self.dl8(xu41))

        out = self.out(xu42)
        return out
