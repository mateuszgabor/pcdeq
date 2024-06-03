import math
import torch
from tqdm import tqdm


def initialize_nn_weights(tensor, param=0.01):
    return (
        tensor.uniform_()
        * math.sqrt(param)
        / math.sqrt((tensor.shape[0] + tensor.shape[1]))
    )


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def epoch(loader, model, device, criterion, opt=None):
    top1 = AverageMeter()
    losses = AverageMeter()
    forward_iter = AverageMeter()
    backward_iter = AverageMeter()

    if opt is None:
        model.eval()
    else:
        model.train()
    for inputs, targets in tqdm(loader, leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        if opt:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            model.clamp()

        acc, _ = accuracy(outputs, targets, topk=(1, 5))
        forward_avg = sum([block.deq.iter_forward for block in model.blocks]) / len(
            model.blocks
        )
        backward_avg = sum([block.deq.iter_backward for block in model.blocks]) / len(
            model.blocks
        )
        top1.update(acc[0], inputs.size(0))
        losses.update(loss.item(), inputs.size(0))
        forward_iter.update(forward_avg)
        backward_iter.update(backward_avg)

    return top1.avg, losses.avg, forward_iter.avg, backward_iter.avg
