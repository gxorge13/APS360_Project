from imports import *
from helper import *

class Classifier(nn.Module):
    def __init__(self, name = "classifier"):
        super(Classifier, self).__init__()
        self.name = name

        #512 x 9 x 9
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(512*9*9, 256)
        # self.bn1 = nn.BatchNorm1d(256)
        self.l2 = nn.Linear(256, 128)
        # self.bn2 = nn.BatchNorm1d(128)
        self.l3 = nn.Linear(128, 3)
        self.drop = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.l1(x))
        # x = F.relu(self.bn1(self.l1(x)))
        x = self.drop(x)
        x = F.relu(self.l2(x))
        # x = F.relu(self.bn2(self.l2(x)))
        x = self.l3(x)

        return x


