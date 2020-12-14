import torch
import torch.nn.functional as F
import random


class ICircleDSAC:
	'''
	Differentiable RANSAC to robustly fit lines.
	'''

	def __init__(self, hyps, inlier_thresh, inlier_beta, inlier_alpha, loss_function):
		'''
		Constructor.

		hyps -- number of line hypotheses sampled for each image
		inlier_thresh -- threshold used in the soft inlier count, its measured in relative image size (1 = image width)
		inlier_beta -- scaling factor within the sigmoid of the soft inlier count
		inlier_alpha -- scaling factor for the soft inlier scores (controls the peakiness of the hypothesis distribution)
		loss_function -- function to compute the quality of estimated line parameters wrt ground truth
		'''

		self.hyps = hyps
		self.inlier_thresh = inlier_thresh
		self.inlier_beta = inlier_beta
		self.inlier_alpha = inlier_alpha
		self.loss_function = loss_function

	def _sample_hyp(self, x, y):
		'''
		Calculate a circle hypothesis (cX, cY, r) from three random points.

		x -- vector of x values
		y -- vector of y values
		'''
		return 0, 0, 0, False

	def _soft_inlier_count(self, cX, cY, r, x, y):
		'''
		Soft inlier count for a given circle and a given set of points.

		cX -- x of circle center
		cY -- y of circle center
		r -- radius of the line
		x -- vector of x values
		y -- vector of y values
		'''
		return 0, torch.zeros(x.size())

	def _refine_hyp(self, x, y, weights):
		'''
		Refinement by weighted least squares fit.

		x -- vector of x values
		y -- vector of y values
		weights -- vector of weights (1 per point)		
		'''
		return 0, 0, 0
		
	def __call__(self, prediction, labels):
		'''
		Perform robust, differentiable line fitting according to DSAC.

		Returns the expected loss of choosing a good line hypothesis which can be used for backprob.

		prediction -- predicted 2D points for a batch of images, array of shape (Bx2) where
			B is the number of images in the batch
			2 is the number of point dimensions (y, x)
		labels -- ground truth labels for the batch, array of shape (Bx2) where
			B is the number of images in the batch
			2 is the number of parameters (intercept, slope)
		'''

		# working on CPU because of many, small matrices
		prediction = prediction.cpu()

		batch_size = prediction.size(0)

		avg_exp_loss = 0 # expected loss
		avg_top_loss = 0 # loss of best hypothesis

		self.est_parameters = torch.zeros(batch_size, 3) # estimated lines
		self.est_losses = torch.zeros(batch_size) # loss of estimated lines
		self.batch_inliers = torch.zeros(batch_size, prediction.size(2)) # (soft) inliers for estimated lines

		for b in range(0, batch_size):

			hyp_losses = torch.zeros([self.hyps, 1]) # loss of each hypothesis
			hyp_scores = torch.zeros([self.hyps, 1]) # score of each hypothesis

			max_score = 0 	# score of best hypothesis

			y = prediction[b, 0] # all y-values of the prediction
			x = prediction[b, 1] # all x.values of the prediction

			for h in range(0, self.hyps):	

				# === step 1: sample hypothesis ===========================
				cX, cY, r, valid = self._sample_hyp(x, y)
				if not valid: continue

				# === step 2: score hypothesis using soft inlier count ====
				score, inliers = self._soft_inlier_count(cX, cY, r, x, y)

				# === step 3: refine hypothesis ===========================
				cX_ref, cY_ref, r_ref = self._refine_hyp(x, y, inliers)

				if r_ref > 0: # check whether refinement was implemented
					cX, cY, r = cX_ref, cY_ref, r_ref

				hyp = torch.zeros([3])
				hyp[0] = cX
				hyp[1] = cY
				hyp[2] = r

				# === step 4: calculate loss of hypothesis ================
				loss = self.loss_function(hyp, labels[b]) 

				# store results
				hyp_losses[h] = loss
				hyp_scores[h] = score

				# keep track of best hypothesis so far
				if score > max_score:
					max_score = score
					self.est_losses[b] = loss
					self.est_parameters[b] = hyp
					self.batch_inliers[b] = inliers

			# === step 5: calculate the expectation ===========================

			#softmax distribution from hypotheses scores			
			hyp_scores = F.softmax(self.inlier_alpha * hyp_scores, 0)

			# expectation of loss
			exp_loss = torch.sum(hyp_losses * hyp_scores)
			avg_exp_loss = avg_exp_loss + exp_loss

			# loss of best hypothesis (for evaluation)
			avg_top_loss = avg_top_loss + self.est_losses[b]

		return avg_exp_loss / batch_size, avg_top_loss / batch_size
