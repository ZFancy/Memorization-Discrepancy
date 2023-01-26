import numpy as np
import torch
from utils.model import *
from torch.autograd import Variable
import torch.nn.functional as F

def exp_loss(outputs, partialY):
    k = partialY.shape[1]
    can_num = partialY.sum(dim=1).float()  # n
    
    soft_max = nn.Softmax(dim=1)
    sm_outputs = soft_max(outputs)
    final_outputs = sm_outputs * partialY

    average_loss = ((k-1)/(k-can_num) * torch.exp(-final_outputs.sum(dim=1))).mean()
    return average_loss


def cwloss(output, target,confidence=50, num_classes=10):
    target = target.data
    target_onehot = torch.zeros(target.size() + (num_classes,))
    target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = Variable(target_onehot, requires_grad=False)
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = -torch.clamp(real - other + confidence, min=0.)  # equiv to max(..., 0.)
    loss = torch.sum(loss)
    return loss

def GA_PGD(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init):
    model.eval()
    Kappa = torch.ones(len(data))
    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
        nat_output = model(data)
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        x_adv.requires_grad_()
        nat_output = model(data)
        output = model(x_adv)
        predict = output.max(1, keepdim=True)[1]
        # Update Kappa
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                #loss_adv = -exp_loss(output, target)
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
            if loss_fn == "cw":
                loss_adv = cwloss(output,target)
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                loss_adv = criterion_kl(F.log_softmax(output, dim=1),F.softmax(nat_output, dim=1))
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        # Update adversarial data
        x_adv = x_adv.detach() - eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    #x_adv = Variable(x_adv, requires_grad=False)
    return x_adv, Kappa

def GA_PGD_pi(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init,s=0.75):
    model.eval()
    Kappa = torch.ones(len(data))
    perm = torch.randperm(Kappa.size(0))
    idxt = perm[:int((1-s)*len(data))]
    idxp = perm[int((1-s)*len(data)):]
    x_all = data.detach()
    y_all = target.detach()
    xt = x_all[idxt]
    xp = x_all[idxp]
    yt = y_all[idxt]
    yp = y_all[idxp]
    if category == "trades":
        xp_adv = xp.detach() + 0.001 * torch.randn(xp.shape).cuda().detach() if rand_init else xp.detach()
        nat_output = model(xp)
    if category == "Madry":
        xp_adv = xp.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, xp.shape)).float().cuda() if rand_init else xp.detach()
        xp_adv = torch.clamp(xp_adv, 0.0, 1.0)
    for k in range(num_steps):
        xp_adv.requires_grad_()
        xp_output = model(xp_adv)
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(xp_output, yp)
            if loss_fn == "cw":
                loss_adv = cwloss(xp_output, yp)
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                loss_adv = criterion_kl(F.log_softmax(xp_output, dim=1),F.softmax(nat_output, dim=1))
        loss_adv.backward()
        eta = step_size * xp_adv.grad.sign()
        xp_adv = xp_adv.detach() - eta
        xp_adv = torch.min(torch.max(xp_adv, xp - epsilon), xp + epsilon)
        xp_adv = torch.clamp(xp_adv, 0.0, 1.0)
    x_adv = torch.cat([xt, xp], dim=0)
    label = torch.cat([yt, yp], dim=0)
    return x_adv, label


def eval_clean(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def eval_robust(model, test_loader, perturb_steps, epsilon, step_size, loss_fn, category, random):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.enable_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            x_adv, _ = GA_PGD(model,data,target,epsilon,step_size,perturb_steps,loss_fn,category,rand_init=random)
            output = model(x_adv)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

