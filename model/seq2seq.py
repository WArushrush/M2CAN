from model.config import *
from model.image2image import GRL


class TextFeatureExtractor(nn.Module):
    def __init__(self):
        super(TextFeatureExtractor, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(inplace=True)
        )
        self.g = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64)
        )

    def forward(self, input):
        output = self.hidden(input)
        return F.normalize(output, dim=-1), F.normalize(self.g(output), dim=-1)


class MMDiscriminator(nn.Module):
    def __init__(self):
        super(MMDiscriminator, self).__init__()
        self.dim = 64
        self.linear = nn.Linear(self.dim, 2)
        self.grl = GRL()

    def forward(self, input):
        input = self.grl(input)
        output = self.linear(input)
        output = F.log_softmax(output, dim=1)
        return output


# Task Classifier
class TaskClassifier(nn.Module):
    def __init__(self):
        super(TaskClassifier, self).__init__()
        self.dim = 64
        self.linear = nn.Linear(self.dim, 3)

    def forward(self, input):
        output = self.linear(input)
        output = F.log_softmax(output, dim=1)
        return output


class MLB(nn.Module):
    def __init__(self):
        super(MLB, self).__init__()
        self.dim = 256
        self.U = nn.Linear(self.dim, self.dim // 2)
        self.V = nn.Linear(self.dim, self.dim // 2)
        self.P = nn.Linear(self.dim // 2, self.dim // 4)
        self.h1 = nn.Linear(self.dim, self.dim // 4)
        self.h2 = nn.Linear(self.dim, self.dim // 4)

    def forward(self, input1, input2):
        output1 = self.U(input1)
        output2 = self.V(input2)
        output = F.sigmoid(output1 * output2)
        output = self.P(output) + self.h1(input1) + self.h2(input2)
        return output