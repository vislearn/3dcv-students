import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from vll.utils.helper import prepare_data, batch_loss
from vll.model.circle_nn import CircleNN

from vll.utils.circle_loss import ICircleLoss
from vll.utils.circle_dsac import ICircleDSAC


def visualize(dataset, loss, dsac, point_nn, direct_nn, val_images, val_inputs, val_labels, params):
	point_nn.eval()
	direct_nn.eval()

	# DSAC validation prediction
	prediction = point_nn(val_inputs)
	val_exp, val_loss = dsac(prediction, val_labels)
	val_correct = dsac.est_losses < params['valthresh']

	# direct nn validation prediction
	direct_val_est = direct_nn(val_inputs)
	direct_val_loss = batch_loss(direct_val_est, val_labels, loss)
	direct_val_correct = direct_val_loss < params['valthresh']

	direct_val_est = direct_val_est.detach().cpu().numpy()
	dsac_val_est = dsac.est_parameters.detach().cpu().numpy()
	points = prediction.detach().cpu().numpy()

	# draw DSAC estimates
	viz_dsac = dataset.draw_models(val_labels)
	viz_dsac = dataset.draw_points(points, viz_dsac, dsac.batch_inliers)
	viz_dsac = dataset.draw_models(dsac_val_est, viz_dsac, val_correct)

	# draw direct estimates
	viz_direct = dataset.draw_models(val_labels)
	viz_direct = dataset.draw_models(direct_val_est, viz_direct, direct_val_correct)

	def make_grid(batch):
		batch = torch.from_numpy(batch)
		batch.transpose_(1, 3).transpose_(2, 3)		
		return vutils.make_grid(batch, nrow=3,normalize=False)

	viz_inputs = make_grid(val_images)
	viz_dsac = make_grid(viz_dsac)
	viz_direct = make_grid(viz_direct)

	viz = torch.cat((viz_inputs, viz_dsac, viz_direct), 2)
	viz.transpose_(0, 1).transpose_(1, 2)	
	viz = viz.numpy()

    # show image
	plt.imshow(viz)
	plt.show()

	return val_exp, val_loss, direct_val_loss


def visualize_circle_dsac(params, dataset, loss=None, dsac=None):
	if not loss:
		loss = ICircleLoss(params['imagesize'])
	if not dsac:
		dsac = ICircleDSAC(params['hypotheses'], params['inlierthreshold'], params['inlierbeta'], params['inlieralpha'], loss)

	point_nn = CircleNN(params['capacity'], params['receptivefield'])
	direct_nn = CircleNN(params['capacity'], 0, True)

	val_images, val_labels = dataset.samples(params['valsize'])
	val_inputs, val_labels = prepare_data(val_images, val_labels, params['use_gpu'])

	visualize(dataset, loss, dsac, point_nn, direct_nn, val_images, val_inputs, val_labels, params)
