import time

import torch
import torch.optim as optim

from vll.utils.helper import prepare_data, batch_loss
from vll.visualize.dsac import visualize

from vll.data.line_dataset import LineDataset
from vll.model.line_nn import LineNN
from vll.utils.line_loss import LineLoss
from vll.utils.line_dsac import LineDSAC

from vll.model.circle_nn import CircleNN
from vll.utils.circle_loss import ICircleLoss
from vll.utils.circle_dsac import ICircleDSAC


def default_params(id):
	return {
		'receptivefield': 65,
		'capacity': 4,
		'hypotheses': 64,
		'inlierthreshold': 0.05,
		'inlieralpha': 0.5,
		'inlierbeta': 100.0,
		'learningrate': 0.001,
		'lrstep': 2500,
		'lrstepoffset': 30000,
		'batchsize': 32,
		'trainiterations': 100,
		'imagesize': 64,
		'storeinterval': 10,
		'valsize': 9,
		'valthresh': 5,
		'use_gpu': False,
		'id': id
	}


def __train(dataset, loss, dsac, point_nn, direct_nn, params):
	# we train two CNNs in parallel
	# 1) a CNN that predicts points and is trained with DSAC -> PointNN (good idea)
	if params['use_gpu']: point_nn = point_nn.cuda()
	point_nn.train()
	opt_point_nn = optim.Adam(point_nn.parameters(), lr=params['learningrate'])
	lrs_point_nn = optim.lr_scheduler.StepLR(opt_point_nn, params['lrstep'], gamma=0.5)

	# 2) a CNN that predicts the line parameters directly -> DirectNN (bad idea)
	if params['use_gpu']: direct_nn = direct_nn.cuda()
	direct_nn.train()
	opt_direct_nn = optim.Adam(direct_nn.parameters(), lr=params['learningrate'])
	lrs_direct_nn = optim.lr_scheduler.StepLR(opt_direct_nn, params['lrstep'], gamma=0.5)

	# keep track of training progress
	train_log = open('./models/log_dsac_{}.txt'.format(params['id']), 'w', 1)

	# generate validation data (for consistent vizualisation only)
	val_images, val_labels = dataset.samples(params['valsize'])
	val_inputs, val_labels = prepare_data(val_images, val_labels, params['use_gpu'])

	# start training
	for iteration in range(0, params['trainiterations']+1):
		start_time = time.time()

		# generate training data
		inputs, labels = dataset.samples(params['batchsize'])
		inputs, labels = prepare_data(inputs, labels, params['use_gpu'])

		# point nn forward pass
		point_prediction = point_nn(inputs)

		# robust line fitting with DSAC
		exp_loss, top_loss = dsac(point_prediction, labels)
		
		if exp_loss > 0:
			exp_loss.backward()			# calculate gradients (pytorch autograd)
			opt_point_nn.step()			# update parameters 
			opt_point_nn.zero_grad()		# reset gradient buffer
			if iteration >= params['lrstepoffset']:	
				lrs_point_nn.step()		# update learning rate schedule

		# also train direct nn
		direct_prediction = direct_nn(inputs)
		direct_prediction = direct_prediction.cpu()
		direct_loss = batch_loss(direct_prediction, labels, loss).mean()

		if direct_loss > 0:
			direct_loss.backward()			# calculate gradients (pytorch autograd)
			opt_direct_nn.step()			# update parameters 
			opt_direct_nn.zero_grad()		# reset gradient buffer
			if iteration >= params['lrstepoffset']: 
				lrs_direct_nn.step()		# update learning rate schedule

		# wrap up
		end_time = time.time()-start_time
		print('Iteration: %6d, DSAC Expected Loss: %2.2f, DSAC Top Loss: %2.2f, Direct Loss: %2.2f, Time: %.2fs' 
			% (iteration, exp_loss, top_loss, direct_loss, end_time), flush=True)

		train_log.write('%d %f %f %f\n' % (iteration, exp_loss, top_loss, direct_loss))

		del exp_loss, top_loss, direct_loss

		# store prediction vizualization and nn weights (each couple of iterations)
		if iteration % params['storeinterval'] == 0:
			# Visualize prediction
			val_exp, val_loss, direct_val_loss = visualize(dataset, loss, dsac, point_nn, direct_nn, val_images, val_inputs, val_labels, params)
		
			# store model weights
			torch.save(point_nn.state_dict(), './models/weights_pointnn_' + params['id'] + '.net')
			torch.save(direct_nn.state_dict(), './models/weights_directnn_' + params['id'] + '.net')

			print('Storing snapshot. Validation loss: %2.2f'% val_loss, flush=True)

			del val_exp, val_loss, direct_val_loss

			# enable training of models
			point_nn.train()
			direct_nn.train()


	print('Done without errors.')
	train_log.close()


def train_line_dsac(params):
	dataset = LineDataset(params['imagesize'], params['imagesize'])
	loss = LineLoss(params['imagesize'])
	dsac = LineDSAC(params['hypotheses'], params['inlierthreshold'], params['inlierbeta'], params['inlieralpha'], loss)

	point_nn = LineNN(params['capacity'], params['receptivefield'])
	direct_nn = LineNN(params['capacity'], 0, True)

	__train(dataset, loss, dsac, point_nn, direct_nn, params)


def train_circle_dsac(params, dataset, loss=None, dsac=None):
	if not loss:
		loss = ICircleLoss(params['imagesize'])
	if not dsac:
		dsac = ICircleDSAC(params['hypotheses'], params['inlierthreshold'], params['inlierbeta'], params['inlieralpha'], loss)
	
	point_nn = CircleNN(params['capacity'], params['receptivefield'])
	direct_nn = CircleNN(params['capacity'], 0, True)

	__train(dataset, loss, dsac, point_nn, direct_nn, params)
