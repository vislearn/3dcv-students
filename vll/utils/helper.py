import torch

def prepare_data(inputs, labels, use_gpu):
	# convert from numpy images to normalized torch arrays
	inputs = torch.from_numpy(inputs)
	labels = torch.from_numpy(labels)	

	if use_gpu: inputs = inputs.cuda()
	inputs.transpose_(1,3).transpose_(2, 3)
	inputs = inputs - 0.5 # normalization

	return inputs, labels


def batch_loss(prediction, labels, loss_fn):
	# caluclate the loss for each image in the batch
	losses = torch.zeros(labels.size(0))
	for b in range(0, labels.size(0)):
		losses[b] = loss_fn(prediction[b], labels[b])
	return losses
