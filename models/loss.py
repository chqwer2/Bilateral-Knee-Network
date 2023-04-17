import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

# version 1: use torch.autograd
class FocalLossV1(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLossV1()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        probs = torch.sigmoid(logits)
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


##
# version 2: user derived grad computation
class FocalSigmoidLossFuncV2(torch.autograd.Function):
    '''
    compute backward directly for better numeric stability
    '''
    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, logits, label, alpha, gamma):
        #  logits = logits.float()

        probs = torch.sigmoid(logits)   # sigmoid = 5
        coeff = (label - probs).abs_().pow_(gamma).neg_()
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))

        ce_term1 = log_probs.mul_(label).mul_(alpha)
        ce_term2 = log_1_probs.mul_(1. - label).mul_(1. - alpha)

        ce = ce_term1.add_(ce_term2)
        loss = ce * coeff

        ctx.vars = (coeff, probs, ce, label, gamma, alpha)

        return loss

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        '''
        compute gradient of focal loss
        '''
        (coeff, probs, ce, label, gamma, alpha) = ctx.vars

        d_coeff = (label - probs).abs_().pow_(gamma - 1.).mul_(gamma)
        d_coeff.mul_(probs).mul_(1. - probs)
        d_coeff = torch.where(label < probs, d_coeff.neg(), d_coeff)
        term1 = d_coeff.mul_(ce)

        d_ce = label * alpha
        d_ce.sub_(probs.mul_((label * alpha).mul_(2).add_(1).sub_(label).sub_(alpha)))
        term2 = d_ce.mul(coeff)

        grads = term1.add_(term2)
        grads.mul_(grad_output)

        return grads, None, None, None


class FocalLossV2(nn.Module):

    def __init__(self,
                 alpha=0.25,  # 0.75
                 gamma=2,   # or 5?
                 reduction='mean'):

        super(FocalLossV2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, grade, label):
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLossV2()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        loss = FocalSigmoidLossFuncV2.apply(logits, label, self.alpha, self.gamma)
        
        
        if self.reduction == 'mean':
            loss = loss.mean()
            
        if self.reduction == 'sum':
            loss = loss.sum()
            
        return loss


class myBCE(nn.Module):
    def __init__(self):
        super(myBCE, self).__init__()
        self.BCE = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits, grade, label):

        return self.BCE(logits, label)


class Focal_Reg(nn.Module):
    def __init__(self, opt=None):
        super(Focal_Reg, self).__init__()

        self.focal = FocalLossV2()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

        self.eps = 1e-5
        self.opt = opt

    def laydown(self, loss, target):

        return (loss/ ((loss/target) + self.eps).detach() / self.opt['loss']['lmk'])

    def forward(self, logits,  grade, label, lmk=None,  pointwise=False):

        logits, grade_pred, lmk_pred = logits

        focal = self.focal(logits, None, label)

        # mse = self.l1(grade_pred, grade)
        mse = self.mse(grade_pred, grade)

        lmk_mse = self.mse(lmk_pred, lmk.reshape(-1, 32))

        return focal + self. laydown(mse, focal) + self.laydown(lmk_mse, focal)

class LMK_Reg(nn.Module):
    def __init__(self):
        super(LMK_Reg, self).__init__()

        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.weight = 1


    def forward(self, lmk_pred, lmk=None):

        lmk_mse = torch.sqrt(self.mse(lmk_pred, lmk.reshape(-1, 32)))

        return  lmk_mse / self.weight

# loss = loss1 + loss2 / (loss2 / loss1).detach() + loss3 / (loss3 / loss1).detach()