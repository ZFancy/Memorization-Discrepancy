import torch
import torch.nn as nn
import attack_generator as attack
from utils.train_helpers import *

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
		adv_inputs, _ = attack.GA_PGD(model, inputs, labels, 0.2, 0.02, 10, loss_fn="cent",category="Madry",rand_init=True)

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

def adapt_tensor_prune(model, inputs, labels, optimizer, criterion, niter, batch_size, mode='train', args=None):
	if mode == 'train':
		model.train()
	elif mode == 'eval':
		model.eval()
	else:
		raise IOError
	
	

	for iteration in range(niter):
		model.eval()
		adv_inputs, _ = attack.GA_PGD(model, inputs, labels, 0.2, 0.02, 10, loss_fn="cent",category="Madry",rand_init=True)

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

def adapt_tensor_pi(model, inputs, labels, optimizer, criterion, niter, batch_size, mode='train', args=None):
	if mode == 'train':
		model.train()
	elif mode == 'eval':
		model.eval()
	else:
		raise IOError
	for iteration in range(niter):
		adv_inputs, labels = attack.GA_PGD_pi(model, inputs, labels, 0.1, 0.03, 10, loss_fn="cent",category="Madry",rand_init=True, s=0.75)
		
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
