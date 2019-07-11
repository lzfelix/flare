from typing import Dict

import torch
import torch.nn as nn
import torch.nn. functional as F
import torch.optim  as optim
import torch.utils.data as data
import torchvision

from flare import trainer

LOG_INTERVAL = 100
N_EPOCHS = 5


def get_dataset(train=True):
    dataset = torchvision.datasets.MNIST('./data', train=train, download=True,
                                         transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.1307, ), (0.1307, ))]
                                        ))
    return data.DataLoader(dataset, batch_size=16, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x: torch.Tensor):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    def metric(self, prediction: torch.Tensor, ground: torch.Tensor) -> Dict[str, float]:
        amount_correct = torch.sum(prediction.argmax(-1) == ground).item()
        return {'accuracy': amount_correct}


if __name__ == '__main__':
    train_loader = get_dataset()
    test_loader  = get_dataset(train=False)

    model = Net()
    opt = optim.SGD(model.parameters(), lr=1e-2, momentum=0.5)

    logs = trainer.train_on_loader(model, train_loader, test_loader, F.nll_loss, opt, n_epochs=2)
