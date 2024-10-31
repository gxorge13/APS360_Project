from imports import *
from helper import *

class Classifier(nn.Module):
    def __init__(self, name = classifier):
        super(Classifier, self).__init__()
        self.name = name

        #512 x 9 x 9
        self.l1 = nn.Linear(512*9*9, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 3)
        self.drop = nn.dropout(p=0.3)

    def forward(self, x):
        x = F.ReLu(self.l1(x))
        x = self.drop(x)
        x = F.ReLu(self.l2(x))
        x = self.l3(x)

        return x


