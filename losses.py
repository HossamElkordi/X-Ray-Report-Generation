import torch
import torch.nn as nn
import torch.nn.functional as F


class KLLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.KLLoss = nn.KLDivLoss()

    def forward(self, output, target):
        '''
		Output: (N,*) \n
		Target: (N,*) \n
		'''
        output = torch.log(output)  # Invert softmax
        # target = torch.log(target) # Invert softmax
        # How output distribution differs from target distribution
        return self.KLLoss(output, target)


class CELoss(nn.Module):
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.CELoss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, output, target):
        '''
		Output: (N,*,C) \n
		Target: (N,*) \n
		'''
        output = torch.log(output)  # Invert softmax
        output = output.reshape(-1, output.shape[-1])  # (*,C)
        target = target.reshape(-1).long()  # (*)
        return self.CELoss(output, target)


class CELossSame(nn.Module):
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.CELoss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, outputs, target):
        '''
		Output: (N,*,C) \n
		Target: (N,*) \n
		'''
        output_img = torch.log(outputs[0])  # Invert softmax
        output_txt = torch.log(outputs[1])
        output_sen = torch.log(outputs[2])

        output_img = output_img.reshape(-1, output_img.shape[-1])  # (*,C)
        output_txt = output_txt.reshape(-1, output_txt.shape[-1])  # (*,C)
        output_sen = output_sen.reshape(-1, output_sen.shape[-1])  # (*,C)
        target = target.reshape(-1).long()  # (*)
        return self.CELoss(output_img, target) + self.CELoss(output_txt, target) + self.CELoss(output_sen, target)


class CELossShift(nn.Module):
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.CELoss = CELoss(ignore_index=ignore_index)

    def forward(self, output, target):
        '''
		Output: (N,*,C) \n
		Target: (N,*) \n
		'''
        output = output[:, :-1, :]  # (* - 1,C)
        target = target[:, 1:]  # (* - 1)
        return self.CELoss(output, target)


class ULLoss(nn.Module):
    def __init__(self, min=1e-20):
        self.min = min

    def forward(self, output):
        output = output.unsqueese()
        pred_toks = output.argmax(dim=1, keepdim=True)

        mask = ngram_repeat_mask(pred_toks, 4).type_as(output)
        pred_lprobs = output.view(-1, output.size(2)).gather(1, pred_toks.view(-1, 1))
        one_minus_probs = torch.clamp((1.0 - pred_lprobs.exp()), min=self.min).view(pred_toks.size(0), pred_toks.size(1))
        loss = -torch.log(one_minus_probs) * mask
        return loss.sum()


class CELossTotal(nn.Module):
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.CELoss = CELoss()
        self.CELossShift = CELossShift(ignore_index=ignore_index)

    def forward(self, output, target):
        return self.CELossShift(output[0], target[0]) + self.CELoss(output[1], target[1])


class CELossTotalEval(nn.Module):
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.CELoss = CELoss()
        self.CELossShift = CELossShift(ignore_index=ignore_index)
        self.ulLoss = ULLoss()

    def forward(self, output, target):
        return self.CELossShift(output[0], target[0]) + self.ulLoss(output[0]) + self.CELoss(output[1], target[1]) + self.CELoss(output[2],
                                                                                                        target[1])


class CELossTransfer(nn.Module):
    def __init__(self, ignore_index=-1):
        super().__init__()
        self.CELoss = CELoss()
        self.CELossShift = CELossShift(ignore_index=ignore_index)

    def forward(self, output, target):
        return self.CELossShift(output[0], target[0])  # + self.CELoss(output[1], target[1])


def ngram_repeat_mask(xs, n):
    mask = torch.zeros_like(xs)
    for i, x in enumerate(xs):
        seen = set()
        xl = x.tolist()
        for j in range(len(x)-n):
            ng = tuple(xl[j:j+n])
            if ng in seen:
                mask[i, j:j+n] = 1
            seen.add(ng)
    return mask