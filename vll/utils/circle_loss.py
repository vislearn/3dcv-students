import torch


class ICircleLoss:
	'''
	Compares two circle by distance of parameter values.
	'''

	def __init__(self, image_size):
		'''
		Constructor.

		image_size -- size of the input images, used to normalize the loss
		'''
		self.image_size = image_size

	def __call__(self, est, gt):
		'''
		Calculate the circle loss.

		est -- estimated circle, form: [cX, cY, r]
		gt -- ground truth circle, form: [cX, cY, r]
		'''
		return 0
