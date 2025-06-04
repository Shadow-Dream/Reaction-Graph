import torch
class RBFLayer(torch.nn.Module):
    def __init__(self, dim):
        super(RBFLayer, self).__init__()
        self.dim = dim
        self.centers = torch.nn.Parameter(torch.Tensor(dim))
        self.beta = torch.nn.Parameter(torch.Tensor(dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.centers, 0, 1)
        torch.nn.init.constant_(self.beta, 10)

    def forward(self, x):
        x = x.view(-1, 1)
        centers = self.centers.view(1, -1)
        diff = x - centers
        dist_sq = torch.square(diff)
        out = torch.exp(-self.beta * dist_sq)
        return out