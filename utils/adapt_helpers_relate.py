from curses import KEY_SAVE
from encodings import normalize_encoding
from multiprocessing import reduction
from pickle import REDUCE
from turtle import color
import torch
import torch.nn as nn
import torch.optim as optim
import attack_generator as attack
from utils.train_helpers import *
import torch.nn.functional as F
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def adapt_tensor(model, inputs, labels, optimizer, criterion, niter, batch_size, mode='train', args=None):
	if mode == 'train':
		model.train()
	elif mode == 'eval':
		model.eval()
	else:
		raise IOError
	for iteration in range(niter):
		optimizer.zero_grad()
		logit = model(inputs)
		loss = criterion(logit, labels)
		loss.backward()
		if args.clip_gradnorm:
			total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipvalue, norm_type='inf')
		optimizer.step()

def adapt_tensor_adv(model, inputs, labels, optimizer, criterion, niter, batch_size, mode='train', args=None):
	if mode == 'train':
		model.train()
	elif mode == 'eval':
		model.eval()
	else:
		raise IOError
	for iteration in range(niter):
		model.eval()
		adv_inputs, _ = attack.GA_PGD(model, inputs, labels, 0.1, 0.1, 1, loss_fn="cent",category="Madry",rand_init=True)
		
		#model.train()
		#optimizer.zero_grad()
		logit = model(adv_inputs)
		pre = logit.max(1, keepdim=True)[1].view_as(labels)
		adv_inputs_again, _ = attack.GA_PGD(model, inputs, pre, 0.2, 0.2, 2, loss_fn="cent",category="Madry",rand_init=True)

		model.train()
		logit_orz = model(adv_inputs_again)
		optimizer.zero_grad()
		#print(pre)
		loss = criterion(logit_orz, labels)
		loss.backward()
		if args.clip_gradnorm:
			total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipvalue, norm_type='inf')
		optimizer.step()

def adapt_tensor_trades(model, inputs, labels, optimizer, criterion, niter, batch_size, mode='train', args=None):
	if mode == 'train':
		model.train()
	elif mode == 'eval':
		model.eval()
	else:
		raise IOError
	for iteration in range(niter):
		model.eval()
		adv_inputs, _ = attack.GA_PGD(model, inputs, labels, 0.1, 0.03, 10, loss_fn="kl",category="Madry",rand_init=True)
		
		model.train()
		#optimizer.zero_grad()
		logit = model(adv_inputs)
		#pre = logit.max(1, keepdim=True)[1].view_as(labels)

		#adv_inputs_again, _ = attack.GA_PGD(model, inputs, pre, 0.1, 0.03, 10, loss_fn="cent",category="Madry",rand_init=True)
		#model.train()
		optimizer.zero_grad()
		#print(pre)
		loss = criterion(logit, labels)
		loss.backward()
		if args.clip_gradnorm:
			total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipvalue, norm_type='inf')
		optimizer.step()

def adapt_tensor_at(model, inputs, labels, optimizer, criterion, niter, batch_size, mode='train', args=None):
	if mode == 'train':
		model.train()
	elif mode == 'eval':
		model.eval()
	else:
		raise IOError
	for iteration in range(niter):
		model.eval()
		adv_inputs, _ = attack.GA_PGD(model, inputs, labels, 0.2, 0.02, 10, loss_fn="cent",category="Madry",rand_init=False)

		model.train()
		optimizer.zero_grad()
		logit = model(adv_inputs)
		loss = criterion(logit, labels)
		loss.backward()
		if args.clip_gradnorm:
			total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipvalue, norm_type='inf')
		optimizer.step()

def TRADES_loss(adv_logits, natural_logits, target, beta):
    batch_size = len(target)
    criterion_kl = nn.KLDivLoss(size_average=False).cuda()
    loss_natural = nn.CrossEntropyLoss(reduction='mean')(natural_logits, target)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_logits, dim=1),
                                                         F.softmax(natural_logits, dim=1))
    loss = loss_natural + beta * loss_robust
    return loss

def MART_loss(adv_logits, natural_logits, target, beta):
    kl = nn.KLDivLoss(reduction='none')
    batch_size = len(target)
    adv_probs = F.softmax(adv_logits, dim=1)
    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]
    new_y = torch.where(tmp1[:, -1] == target, tmp1[:, -2], tmp1[:, -1])
    loss_adv = F.cross_entropy(adv_logits, target) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)
    nat_probs = F.softmax(natural_logits, dim=1)
    true_probs = torch.gather(nat_probs, 1, (target.unsqueeze(1)).long()).squeeze()
    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))
    loss = loss_adv + float(beta) * loss_robust
    return loss

def adapt_tensor_early(model, model_eval, inputs, inputs_eval, labels, optimizer, criterion, niter, batch_size, mode='train', args=None, ther=0.4):
	if mode == 'train':
		model.train()
	elif mode == 'eval':
		model.eval()
	else:
		raise IOError
	clean = []
	poison = []
	cc = 0 
	for iteration in range(niter):
		
		#inputs_original = inputs.clone().cuda()
		
		inputs, Kaa = attack.GA_PGD_prune(model, model_eval, inputs, labels, 0.2, 0.02, 10, loss_fn="cent",category="Madry",rand_init=False, ther=ther)
		
		model_eval.eval()
		Ka = torch.ones(len(inputs)).cuda()
		KL = nn.KLDivLoss(reduction='none').cuda()
		early_output = model_eval(inputs)
		early_output_eval = model_eval(inputs_eval)
		


		model.eval()
		
		logit_nat = model(inputs)
		logit_nat_eval = model(inputs_eval)
		loss_KL_eval = torch.sum(KL(F.log_softmax(logit_nat_eval, dim=1),F.softmax(early_output_eval, dim=1)), dim=1)
		loss_KL = torch.sum(KL(F.log_softmax(logit_nat, dim=1),F.softmax(early_output, dim=1)), dim=1)
		
		print(loss_KL_eval.mean())
		print(loss_KL.mean())

		clean.append(loss_KL_eval.mean().cpu().item())
		poison.append(loss_KL.mean().cpu().item())
		cc = cc+1 		
		print(cc)
		# if cc >= 10:
		# 	print(clean)
		# 	print(poison)
		# 	input()

		

		model.train()
		model.zero_grad()
		optimizer.zero_grad()
		logit_nat = model(inputs)

		loss_KL = torch.sum(KL(F.log_softmax(logit_nat, dim=1),F.softmax(early_output, dim=1)), dim=1)
		
		for i in range(len(inputs)):
			Ka[i] = loss_KL[i].data
		count = 0

		pre = logit_nat.max(1, keepdim=True)[1].view_as(labels)
		for i in range(len(pre)):
			if pre[i]!=labels[i]:
				Ka[i] = 0.0

		Ka = torch.ones(len(inputs)).cuda()
		criterion = nn.CrossEntropyLoss(reduce=False)
		loss = criterion(logit_nat, labels).mul(Ka).mean()
		#loss = MART_loss(logit_nat,inputs_original_logit, labels, beta=0.05)
		loss.backward()
		if args.clip_gradnorm:
			total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipvalue, norm_type='inf')
		optimizer.step()


def adapt_tensor_prune(model, model_eval, inputs, labels, optimizer, criterion, niter, batch_size, mode='train', args=None):
	if mode == 'train':
		model.train()
	elif mode == 'eval':
		model.eval()
	else:
		raise IOError

	for iteration in range(niter):
		model_eval.eval()
		optimizer_eval = optim.SGD(model_eval.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0)

		model.eval()
		inputs_m, Ka = attack.GA_PGD_prune(model, model_eval, inputs, labels, 0.2, 0.02, 10, loss_fn="cent",category="Madry",rand_init=True)

		Ka = torch.ones(len(inputs)).cuda()
		model_eval.eval()
		
		model.train()
		optimizer.zero_grad()
		logit_nat = model(inputs)
		count = 0
		#logit_nat = model_eval(x_inputs)

		pre = logit_nat.max(1, keepdim=True)[1].view_as(labels)
		for i in range(len(pre)):
			if pre[i]!=labels[i]:
				Ka[i] = 0.0
				count += 1
		print(count)
		#logit_adv = model(adv_inputs)
		print(Ka * len(Ka) / Ka.sum() - Ka)
		Ka = Ka * len(Ka) / Ka.sum()
		print(Ka.sum())

		criterion = nn.CrossEntropyLoss(reduce=False)
		loss = criterion(logit_nat, labels).mul(Ka).mean() #+ criterion(logit_adv, labels)
		#loss = criterion(logit, labels).mean()
		loss.backward()
		if args.clip_gradnorm:
			total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipvalue, norm_type='inf')
		optimizer.step()

def adapt_tensor_pi(model, model_eval, inputs, labels, optimizer, criterion, niter, batch_size, mode='train', args=None):
	if mode == 'train':
		model.train()
	elif mode == 'eval':
		model.eval()
	else:
		raise IOError
	for iteration in range(niter):
		adv_inputs, labels = attack.GA_PGD_pi(model, model_eval, inputs, labels, 0.2, 0.02, 10, loss_fn="cent",category="Madry",rand_init=True, s=0.75)
		
		model.train()
		optimizer.zero_grad()
		logit = model(adv_inputs)
		#pre = logit.max(1, keepdim=True)[1].view_as(labels)
		#print(pre)
		loss = criterion(logit, labels)
		loss.backward()
		if args.clip_gradnorm:
			total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipvalue, norm_type='inf')
		optimizer.step()

def adapt_tensor_reverse(model, inputs, labels, optimizer, criterion, niter, batch_size, mode='train', args=None):
	if mode == 'train':
		model.train()
	elif mode == 'eval':
		model.eval()
	else:
		raise IOError
	for iteration in range(niter):
		optimizer.zero_grad()
		logit = model(inputs)
		loss = - args.poisoned_trigger_step * criterion(logit, labels)
		loss.backward()
		if args.clip_gradnorm:
			total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipvalue, norm_type='inf')
		optimizer.step()

def adapt_tensor_PT(model, poisoned_trigger, optimizer, niter=1, mode='train', args=None):
	if mode == 'train':
		model.train()
	elif mode == 'eval':
		model.eval()
	else:
		raise IOError
	for iteration in range(niter):
		optimizer.zero_grad()
		for PT, para in zip(poisoned_trigger, model.parameters()):
			para.grad = PT
		if args.clip_gradnorm:
			total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipvalue, norm_type=2.0)
		optimizer.step()

def test_single(model, image, label):
	model.eval()
	inputs = te_transforms(image).unsqueeze(0)
	with torch.no_grad():
		outputs, outputs_ssh = model(inputs.to(device))
		_, predicted = outputs.max(1)
		confidence = nn.functional.softmax(outputs_ssh, dim=1).squeeze()[0].item()
	correctness = 1 if predicted.item() == label else 0
	return correctness, confidence
