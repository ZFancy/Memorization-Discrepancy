import numpy as np
import torch
from utils.model import *
from torch.autograd import Variable
import torch.nn.functional as F

def exp_loss(outputs, partialY):
    print(partialY.shape)
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
                #loss_adv = exp_loss(output, target)
                criterion_kl = nn.KLDivLoss().cuda()
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target) + 1.0 * criterion_kl(F.log_softmax(output, dim=1),F.softmax(nat_output, dim=1))
            if loss_fn == "cw":
                loss_adv = cwloss(output,target)
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                loss_adv =  criterion_kl(F.log_softmax(output, dim=1),F.softmax(nat_output, dim=1))
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


def GA_earlystop(model, data, target, step_size, epsilon, perturb_steps, tau=2, type="fat", random=True, omega=0.0):
    
    model.eval()
    K = perturb_steps
    count = 0

    output_target = []
    output_adv = []
    output_natural = []
    output_Kappa = []

    control = torch.zeros(len(target)).cuda()
    control += tau
    Kappa = torch.zeros(len(data)).cuda()

    if random == False:
        iter_adv = data.cuda().detach()
    else:

        if type == "fat_for_trades" :
            iter_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach()
            iter_adv = torch.clamp(iter_adv, 0.0, 1.0)
        if type == "fat" or "fat_for_mart":
            iter_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda()
            iter_adv = torch.clamp(iter_adv, 0.0, 1.0)
    iter_clean_data = data.cuda().detach()
    iter_target = target.cuda().detach()
    output_iter_clean_data = model(data)

    while K>0:
        iter_adv.requires_grad_()
        output = model(iter_adv)
        pred = output.max(1, keepdim=True)[1]
        output_index = []
        iter_index = []

        for idx in range(len(pred)):
            if pred[idx] != target[idx]:
                if control[idx]==0:
                    output_index.append(idx)
                else:
                    control[idx]-=1
                    iter_index.append(idx)
            else:
                # Update Kappa
                Kappa[idx] += 1
                iter_index.append(idx)

        if (len(output_index)!=0):
            if (len(output_target) == 0):
                # incorrect adv data should not keep iterated
                output_adv = iter_adv[output_index].reshape(-1, 3, 32, 32).cuda()
                output_natural = iter_clean_data[output_index].reshape(-1, 3, 32, 32).cuda()
                output_target = iter_target[output_index].reshape(-1).cuda()
                output_Kappa = Kappa[output_index].reshape(-1).cuda()
            else:
                # incorrect adv data should not keep iterated
                output_adv = torch.cat((output_adv, iter_adv[output_index].reshape(-1, 3, 32, 32).cuda()), dim=0)
                output_natural = torch.cat((output_natural, iter_clean_data[output_index].reshape(-1, 3, 32, 32).cuda()), dim=0)
                output_target = torch.cat((output_target, iter_target[output_index].reshape(-1).cuda()), dim=0)
                output_Kappa = torch.cat((output_Kappa, Kappa[output_index].reshape(-1).cuda()), dim=0)

        model.zero_grad()
        with torch.enable_grad():
            if type == "fat" or type == "fat_for_mart":
                loss_adv = nn.CrossEntropyLoss()(output, iter_target)
            if type == "fat_for_trades":
                criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                loss_adv = criterion_kl(F.log_softmax(output, dim=1),F.softmax(output_iter_clean_data, dim=1))
        loss_adv.backward(retain_graph=True)
        grad = iter_adv.grad

        if len(iter_index) != 0:
            Kappa = Kappa[iter_index]
            control = control[iter_index]
            iter_adv = iter_adv[iter_index]
            iter_clean_data = iter_clean_data[iter_index]
            iter_target = iter_target[iter_index]
            output_iter_clean_data = output_iter_clean_data[iter_index]
            grad = grad[iter_index]
            eta = step_size * grad.sign()
            iter_adv = iter_adv.detach() + eta + omega * torch.randn(iter_adv.shape).detach().cuda()
            iter_adv = torch.min(torch.max(iter_adv, iter_clean_data - epsilon), iter_clean_data + epsilon)
            iter_adv = torch.clamp(iter_adv, 0, 1)
            count += len(iter_target)
        else:
            return output_adv, output_target
        K = K-1

    if (len(output_target) == 0):
        output_target = iter_target.reshape(-1).squeeze().cuda()
        output_adv = iter_adv.reshape(-1, 3, 32, 32).cuda()
        output_natural = iter_clean_data.reshape(-1, 3, 32, 32).cuda()
        output_Kappa = Kappa.reshape(-1).cuda()
    else:
        output_adv = torch.cat((output_adv, iter_adv.reshape(-1, 3, 32, 32)), dim=0).cuda()
        output_target = torch.cat((output_target, iter_target.reshape(-1)), dim=0).squeeze().cuda()
        output_natural = torch.cat((output_natural, iter_clean_data.reshape(-1, 3, 32, 32).cuda()),dim=0).cuda()
        output_Kappa = torch.cat((output_Kappa, Kappa.reshape(-1)),dim=0).squeeze().cuda()
    
    return output_adv, output_target

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

def GA_PGD_prune(model, model_eval, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init, ther=0.4):
    model.eval()
    model_eval.eval()
    Kappa = torch.ones(len(data))

    nat_output = model(data)
    predict = nat_output.max(1, keepdim=True)[1]
    # Update Kappa
    for p in range(len(data)):
        if predict[p] != target[p]:
            Kappa[p] = 0

    if category == "trades":
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).cuda().detach() if rand_init else data.detach()
        nat_output = model(data)
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().cuda() if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        x_adv.requires_grad_()
        nat_output = model_eval(x_adv)
        output = model(x_adv)
        model.zero_grad()
        model_eval.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                #loss_adv = exp_loss(output, target)
                criterion_kl = nn.KLDivLoss(reduction='none').cuda()
                loss_adv = 1.0 * nn.CrossEntropyLoss(reduction="mean")(output, target) #+ 0.05 * criterion_kl(F.log_softmax(output, dim=1),F.softmax(nat_output, dim=1))
                #loss_adv = (output - nat_output).sum()
                loss_earlystop = torch.sum(criterion_kl(F.log_softmax(output, dim=1),F.softmax(nat_output, dim=1)),dim=1).mean().data
                if loss_earlystop <= ther:
                    print(loss_earlystop)
                    break
            if loss_fn == "cw":
                loss_adv = cwloss(output,target)
            if loss_fn == "kl":
                criterion_kl = nn.KLDivLoss(size_average=False).cuda()
                loss_adv =  criterion_kl(F.log_softmax(output, dim=1),F.softmax(nat_output, dim=1))
        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        # Update adversarial data
        x_adv = x_adv.detach() - eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    #x_adv = Variable(x_adv, requires_grad=False)
    x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv, Kappa