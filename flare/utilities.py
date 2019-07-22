from torch.utils.data import Dataset


class WrapperDataset(Dataset):
    """Creates a Dataset from data in plain tensors / collections?."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item]
