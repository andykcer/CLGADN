import torch


class BarlowTwinsLoss(torch.nn.Module):
    def __init__(self, lambda_param=5e-3):
        super(BarlowTwinsLoss, self).__init__()
        self.lambda_param = lambda_param

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor):
        # normalize repr. along the batch dimension
        z_a_norm = (z_a - z_a.mean(0)) / z_a.std(0)
        z_b_norm = (z_b - z_b.mean(0)) / z_b.std(0)

        N = z_a.size(0)
        D = z_a.size(1)

        # cross-correlation matrix
        c = torch.mm(z_a_norm.T, z_b_norm) / N  # 256*256 DxD
        # loss
        c_diff = (c - torch.eye(D).cuda()).pow(2)  # DxD
        # multiply off-diagonal elems of c_diff by lambda
        # print(self.lambda_param)
        c_diff[~torch.eye(D, dtype=bool).cuda()] *= self.lambda_param  #
        loss = c_diff.sum() / D / 5

        return loss
