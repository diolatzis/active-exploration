from ext import pytorch_ssim
import torch
import lpips


class LpipsLoss(torch.nn.Module):

    def __init__(self):
        super(LpipsLoss, self).__init__()
        self.loss_map = torch.zeros([0])
        self.loss_network = lpips.LPIPS().to('cuda')

    def forward(self, pred, gt):
        self.loss_map = self.loss_network(pred.permute(0, 3, 2, 1), gt.permute(0, 3, 2, 1))

        return self.loss_map.mean()


class DssimL1Loss(torch.nn.Module):

    def __init__(self):
        super(DssimL1Loss, self).__init__()
        self.loss_map = torch.zeros([0])

    def forward(self, pred, gt):
        self.loss_map = 2 * torch.abs(pred-gt) + (1.0 - pytorch_ssim.ssim(pred, gt, size_average=False))

        return self.loss_map.mean()


class DssimSMAPELoss(torch.nn.Module):

    def __init__(self):
        super(DssimSMAPELoss, self).__init__()
        self.loss_map = torch.zeros([0])

    def forward(self, pred, gt):
        self.loss_map = 2 * torch.abs(pred-gt)/(torch.abs(pred)+torch.abs(gt)+0.01) + (1.0 - pytorch_ssim.ssim(pred, gt, size_average=False))

        return self.loss_map.mean()


class SMAPELoss(torch.nn.Module):

    def __init__(self):
        super(SMAPELoss, self).__init__()
        self.loss_map = torch.zeros([0])

    def forward(self, pred, gt):
        self.loss_map = torch.abs(pred-gt)/(torch.abs(pred)+torch.abs(gt)+0.01)

        return self.loss_map.mean()


class Dssim(torch.nn.Module):

    def __init__(self):
        super(Dssim, self).__init__()

    def forward(self, pred, gt):
        return pytorch_ssim.ssim(pred, gt)


class AllMetrics(torch.nn.Module):

    def __init__(self):
        super(AllMetrics, self).__init__()

        self.metrics = {}

        self.metrics['l1'] = 0
        self.metrics['l2'] = 0
        self.metrics['lpips'] = 0
        self.metrics['dssim'] = 0
        self.metrics['mape'] = 0
        self.metrics['smape'] = 0
        self.metrics['mrse'] = 0

        self.lpips_network = lpips.LPIPS().to('cuda')
        self.loss_map = torch.zeros([0])

    def reset(self):
        self.metrics['l1'] = 0
        self.metrics['l2'] = 0
        self.metrics['lpips'] = 0
        self.metrics['dssim'] = 0
        self.metrics['mape'] = 0
        self.metrics['smape'] = 0
        self.metrics['mrse'] = 0

    def forward(self, pred, gt):

        diff = gt - pred
        eps = 1e-2

        self.metrics['l1'] = (torch.abs(diff))
        self.metrics['l2'] = (diff*diff)
        self.metrics['mrse'] = (diff*diff/(gt*gt+eps))
        self.metrics['mape'] = (torch.abs(diff)/(gt+eps))
        self.metrics['smape'] = (2 * torch.abs(diff)/(gt+pred+eps))
        self.metrics['dssim'] = 1.0 - (pytorch_ssim.ssim(pred, gt))
        self.metrics['lpips'] = self.lpips_network(pred.permute(0, 3, 2, 1), gt.permute(0, 3, 2, 1))

        return self

    def __add__(self, other):
        self.metrics['l1'] = self.metrics['l1'] + other.metrics['l1']
        self.metrics['l2'] = self.metrics['l2'] + other.metrics['l2']
        self.metrics['lpips'] = self.metrics['lpips'] + other.metrics['lpips']
        self.metrics['dssim'] = self.metrics['dssim'] + other.metrics['dssim']
        self.metrics['mape'] = self.metrics['mape'] + other.metrics['mape']
        self.metrics['smape'] = self.metrics['smape'] + other.metrics['smape']
        self.metrics['mrse'] = self.metrics['mrse'] + other.metrics['mrse']
        self.metrics['lpips'] = self.metrics['lpips'] + other.metrics['lpips']
        return self

    def __truediv__(self, other):
        self.metrics['l1'] = self.metrics['l1'] / other.metrics['l1']
        self.metrics['l2'] = self.metrics['l2'] / other.metrics['l2']
        self.metrics['lpips'] = self.metrics['lpips'] / other.metrics['lpips']
        self.metrics['dssim'] = self.metrics['dssim'] / other.metrics['dssim']
        self.metrics['mape'] = self.metrics['mape'] / other.metrics['mape']
        self.metrics['smape'] = self.metrics['smape'] / other.metrics['smape']
        self.metrics['mrse'] = self.metrics['mrse'] / other.metrics['mrse']
        self.metrics['lpips'] = self.metrics['lpips'] / other.metrics['lpips']
        return self
