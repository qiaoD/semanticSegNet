import depth_model
from ioUtils import *
import math
import lossFunction
import tensorflow as tf
import numpy as np
import sys
import os
import scipy.io as sio
import re
import time

import scipy.ndimage.interpolation
import scipy.misc
import skimage.morphology
import skimage.io as skio

THRESHOLD = {"person":1, "rider":1, "motorcycle":1, "bicycle":1,
             "car":2, "truck":2, "bus":2, "train":2}
MIN_SIZE = {"person":20, "rider":20, "motorcycle":20, "bicycle":20,
            "car":25, "truck":45, "bus":45, "train":45}
SELEM = {1: (np.ones((3,3))).astype(np.bool),
         2: (np.ones((5,5))).astype(np.bool)}
VGG_MEAN = [103.939, 116.779, 123.68]

tf.set_random_seed(0)

def initialize_model(outputChannels, wd=None, modelWeightPaths=None):
	params = {"depth/conv1_1": {"name": "depth/conv1_1", "shape": [5,5,2,64], "std": None, "act": "relu", "reuse": False},
			  "depth/conv1_2": {"name": "depth/conv1_2", "shape": [5,5,64,128], "std": None, "act": "relu", "reuse": False},
			  "depth/conv2_1": {"name": "depth/conv2_1", "shape": [5,5,128,128], "std": None, "act": "relu", "reuse": False},
			  "depth/conv2_2": {"name": "depth/conv2_2", "shape": [5,5,128,128], "std": None, "act": "relu", "reuse": False},
			  "depth/conv2_3": {"name": "depth/conv2_3", "shape": [5,5,128,128], "std": None, "act": "relu", "reuse": False},
			  "depth/conv2_4": {"name": "depth/conv2_4", "shape": [5,5,128,128], "std": None, "act": "relu", "reuse": False},
			  "depth/fcn1": {"name": "depth/fcn1", "shape": [1,1,128,128], "std": None, "act": "relu", "reuse": False},
			  "depth/fcn2": {"name": "depth/fcn2", "shape": [1,1,128,outputChannels], "std": None, "act": "relu", "reuse": False},
			  "depth/upscore": {"name": "depth/upscore", "ksize": 8, "stride": 4, "outputChannels": outputChannels},
			  }

	return depth_model.Network(params, wd=wd, modelWeightPaths=modelWeightPaths)

def forward_model(model, feeder, outputSavePath):
	with tf.Session() as sess:
		tfBatchDirs = tf.placeholder("float")
		# QD tfBatchSS = tf.placeholder("float")
		keepProb = tf.placeholder("float")

		with tf.name_scope("model_builder"):
			print "attempting to build model"
			# QD model.build(tfBatchDirs, tfBatchSS, keepProb=keepProb)
			model.build(tfBatchDirs, keepProb=keepProb)
			print "built the model"

		init = tf.initialize_all_variables()

		sess.run(init)

		if not os.path.exists(outputSavePath):
			os.makedirs(outputSavePath)

		for i in range(int(math.floor(feeder.total_samples() / batchSize))):
			'''QD
			dirBatch, ssBatch, idBatch = feeder.next_batch()
			outputBatch = sess.run(model.outputDataArgMax, feed_dict={tfBatchDirs: dirBatch,
																	  tfBatchSS: ssBatch,
																	  keepProb: 1.0})
			'''
			dirBatch, idBatch = feeder.next_batch()
			outputBatch = sess.run(model.outputDataArgMax, feed_dict={tfBatchDirs: dirBatch,
																	  keepProb: 1.0})
			outputBatch = outputBatch.astype(np.uint8)

			for j in range(len(idBatch)):
				outputFilePath = os.path.join(outputSavePath, idBatch[j].lstrip('/')+'.mat')
				outputFileDir = os.path.dirname(outputFilePath)
				print outputFileDir
				print outputFilePath
				# raw_input("pause")

				if not os.path.exists(outputFileDir):
					os.makedirs(outputFileDir)

				sio.savemat(outputFilePath, {"depth_map": outputBatch[j]})

				print "processed image %d out of %d"%(j+batchSize*i, feeder.total_samples())
				
				# use watershed_cut
				outputFilePath = os.path.join(outputSavePath, idBatch[j].lstrip('/')+'.png')
				outputImage = watershed_cut(outputBatch[j])
				skio.imsave(outputFilePath, scipy.ndimage.interpolation.zoom(outputImage, 2.0, mode='nearest', order=0))
				

def watershed_cut(depthImage):
	resultImage = np.zeros(shape=(512,1024), dtype=np.float32)
	# QD ssMask = ssMask.astype(np.int32)
	# QD resultImage = np.zeros(shape=ssMask.shape, dtype=np.float32)

	#for semClass in CLASS_TO_CITYSCAPES.keys():
	#	csCode = CLASS_TO_CITYSCAPES[semClass]
	#	ssCode = CLASS_TO_SS[semClass]
	#	ssMaskClass = (ssMask == ssCode)

	#	ccImage = (depthImage > THRESHOLD[semClass]) * ssMaskClass
	#	ccImage = skimage.morphology.remove_small_objects(ccImage, min_size=MIN_SIZE[semClass])
	#	ccImage = skimage.morphology.remove_small_holes(ccImage)
	#	ccLabels = skimage.morphology.label(ccImage)
	ccImage = (depthImage > 1)
	ccImage = skimage.morphology.remove_small_objects(ccImage, min_size=MIN_SIZE['person'])
	ccImage = skimage.morphology.remove_small_holes(ccImage)
	return ccImage
	ccLabels = skimage.morphology.label(ccImage)
	
	ccIDs = np.unique(ccLabels)[1:]
	for ccID in ccIDs:
		ccIDMask = (ccLabels == ccID)
		ccIDMask = skimage.morphology.binary_dilation(ccIDMask, SELEM[1])
		instanceID = ccID
		resultImage[ccIDMask] = instanceID

	resultImage = resultImage.astype(np.uint16)
	return resultImage


def train_model(model, outputChannels, learningRate, trainFeeder, valFeeder, modelSavePath=None, savePrefix=None, initialIteration=1):
	with tf.Session() as sess:
		tfBatchDirs = tf.placeholder("float")
		tfBatchGT = tf.placeholder("float")
		tfBatchWeight = tf.placeholder("float")
		# QD tfBatchSS = tf.placeholder("float")
		keepProb = tf.placeholder("float")

		with tf.name_scope("model_builder"):
			print "attempting to build model"
			# QD model.build(tfBatchDirs, tfBatchSS, keepProb=keepProb)
			model.build(tfBatchDirs,  keepProb=keepProb)
			print "built the model"
		sys.stdout.flush()
		''' QD
		loss = lossFunction.modelTotalLoss(pred=model.outputData, gt=tfBatchGT, weight=tfBatchWeight, ss=tfBatchSS, outputChannels=outputChannels)
		numPredictedWeighted = lossFunction.countTotalWeighted(ss=tfBatchSS, weight=tfBatchWeight)
		numPredicted = lossFunction.countTotal(ss=tfBatchSS)
		numCorrect = lossFunction.countCorrect(pred=model.outputData, gt=tfBatchGT, ss=tfBatchSS, k=1, outputChannels=outputChannels)
		'''
		loss = lossFunction.modelTotalLoss(pred=model.outputData, gt=tfBatchGT, weight=tfBatchWeight, outputChannels=outputChannels)
		numPredictedWeighted = lossFunction.countTotalWeighted(weight=tfBatchWeight)
		numPredicted = lossFunction.countTotal()
		numCorrect = lossFunction.countCorrect(pred=model.outputData, gt=tfBatchGT, k=1, outputChannels=outputChannels)
		
		train_op = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(loss=loss)

		init = tf.initialize_all_variables()

		sess.run(init)
		iteration = initialIteration

		while iteration < 1000:
			batchLosses = []
			totalPredicted = 0
			totalCorrect = 0
			totalPredictedWeighted = 0
			print "This is iteration:%d" % (iteration)
			for k in range(int(math.floor(valFeeder.total_samples() / batchSize))):
				'''QD
				dirBatch, gtBatch, weightBatch, ssBatch, _ = valFeeder.next_batch()
				# batchLoss, batchDepthError, batchPredicted, batchExceed3, batchExceed5 = \
				#     sess.run([loss, depthError, numPredicted, exceed3, exceed5],
				#                             feed_dict={tfBatchDirs: dirBatch,
				#                                         tfBatchGT: gtBatch,
				#                                         tfBatchSS: ssBatch})

				batchLoss, batchPredicted, batchPredictedWeighted, batchCorrect = sess.run([loss, numPredicted, numPredictedWeighted, numCorrect],
					feed_dict={tfBatchDirs: dirBatch,
							   tfBatchGT: gtBatch,
							   tfBatchWeight: weightBatch,
							   tfBatchSS: ssBatch,
							   keepProb: 1.0})
				'''
				print "This is val iteration:%d" % (k)
				dirBatch, gtBatch, weightBatch, _ = valFeeder.next_batch()
				batchLoss, batchPredicted, batchPredictedWeighted, batchCorrect = sess.run([loss, numPredicted, numPredictedWeighted, numCorrect],
					feed_dict={tfBatchDirs: dirBatch,
							   tfBatchGT: gtBatch,
							   tfBatchWeight: weightBatch,
							   keepProb: 1.0})
				
				batchLosses.append(batchLoss)
				totalPredicted += batchPredicted
				totalPredictedWeighted += batchPredictedWeighted
				totalCorrect += batchCorrect

			if np.isnan(np.mean(batchLosses)):
				print "LOSS RETURNED NaN"
				sys.stdout.flush()
				return 1

			# print "Itr: %d - b %d - val loss: %.3f, depth MSE: %.3f, exceed3: %.3f, exceed5: %.3f"%(iteration,j,
			#     float(np.mean(batchLosses)), totalDepthError/totalPredicted,
			#     totalExceed3/totalPredicted, totalExceed5/totalPredicted)
			print "%s Itr: %d - val loss: %.6f, correct: %.6f" % (time.strftime("%H:%M:%S"),
			iteration, float(np.mean(batchLosses)), totalCorrect / totalPredicted)
			
			print "model saving..."
			if (iteration > 0 and iteration % 5 == 0) or checkSaveFlag(modelSavePath):
				modelSaver(sess, modelSavePath, savePrefix, iteration)

				# print "Processed iteration %d, batch %d" % (i,j)
				# sys.stdout.flush()
			print "model saved"
			sys.stdout.flush()
			# raw_input("paused")
			#for j in range(10):
			for j in range(int(math.floor(trainFeeder.total_samples() / batchSize))):
				'''QD
				dirBatch, gtBatch, weightBatch, ssBatch, _ = trainFeeder.next_batch()
				sess.run(train_op, feed_dict={tfBatchDirs: dirBatch,
											  tfBatchGT: gtBatch,
											  tfBatchWeight: weightBatch,
											  tfBatchSS: ssBatch,
											  keepProb: 0.7})
				'''
				print "This is train iteration:%d" % (j)
				dirBatch, gtBatch, weightBatch, _ = trainFeeder.next_batch()
				sess.run(train_op, feed_dict={tfBatchDirs: dirBatch,
											  tfBatchGT: gtBatch,
											  tfBatchWeight: weightBatch,
											  keepProb: 0.7})
			iteration += 1


def modelSaver(sess, modelSavePath, savePrefix, iteration, maxToKeep=5):
	allWeights = {}

	for name in [n.name for n in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]:
		param = sess.run(name)
		nameParts = re.split('[:/]', name)
		saveName = nameParts[-4]+'/'+nameParts[-3]+'/'+nameParts[-2]
		allWeights[saveName] = param

		# print "Name: %s Mean: %.3f Max: %.3f Min: %.3f std: %.3f" % (name,
		#                                                              param.mean(),
		#                                                              param.max(),
		#                                                              param.min(),
		#                                                              param.std())
		# if name == "depth/fcn2/weights:0":
		#     for j in range(outputChannels):
		#         print "ch: %d, max %e, min %e, std %e" % (
		#             j, param[:, :, :, j].max(), param[:, :, :, j].min(), param[:, :, :, j].std())

	# raw_input("done")

	sio.savemat(os.path.join(modelSavePath, savePrefix+'_%03d'%iteration), allWeights)


def checkSaveFlag(modelSavePath):
	flagPath = os.path.join(modelSavePath, 'saveme.flag')

	if os.path.exists(flagPath):
		return True
	else:
		return False

if __name__ == "__main__":
	outputChannels = 16
	classType = 'unified_CR'
	indices = [0,1,2,3,4,5,6,7]
	# 0=car, 1=person, 2=rider, 3=motorcycle, 4=bicycle, 5=truck, 6=bus, 7=train
	savePrefix = "depth_" + classType + "_CR_pretrain"

	train = True
	train = False
	if train:
		batchSize = 6
		learningRate = 5e-4
		wd = 1e-6

		modelWeightPaths = ["../model/dwt_cityscapes_pspnet.mat"]
		initialIteration = 1

		model = initialize_model(outputChannels=outputChannels, wd=wd, modelWeightPaths=modelWeightPaths)
		trainFeeder = Batch_Feeder(dataset="cityscapes",
										   indices=indices,
										   train=train,
										   batchSize=batchSize,
										   padWidth=None,
										   padHeight=None, flip=True, keepEmpty=False)
		'''QD
		trainFeeder.set_paths(idList=read_ids('../cityscapes/splits/train/list.txt'),
						 gtDir="../cityscapes/unified/iGTFull/train",
						 ssDir="../cityscapes/unified/ssMaskFinePSP/train")
		'''
		trainFeeder.set_paths(idList=read_ids('../cityscapes/splits/train/list.txt'),
						 gtDir="../cityscapes/unified/LabelGT/train")
						 
		valFeeder = Batch_Feeder(dataset="cityscapes", indices=indices, train=train, batchSize=batchSize, padWidth=None, padHeight=None)
		'''QD
		valFeeder.set_paths(idList=read_ids('../cityscapes/splits/val/list.txt'),
						 gtDir="../cityscapes/unified/iGTFull/val",
						 ssDir="../cityscapes/unified/ssMaskFinePSP/val")
		'''
		valFeeder.set_paths(idList=read_ids('../cityscapes/splits/val/list.txt'),
						 gtDir="../cityscapes/unified/LabelGT/val")
						 
		train_model(model=model, outputChannels=outputChannels,
					learningRate=learningRate,
					trainFeeder=trainFeeder,
					valFeeder=valFeeder,
					modelSavePath="../model/depth",
					savePrefix=savePrefix,
					initialIteration=initialIteration)

	else:
		batchSize = 5
		modelWeightPaths = ["../model/depth/depth_unified_CR_CR_pretrain_002.mat"]
		model = initialize_model(outputChannels=outputChannels, wd=5e-5, modelWeightPaths=modelWeightPaths)

		feeder = Batch_Feeder(dataset="cityscapes", train=train, indices=indices, batchSize=batchSize, padWidth=None, padHeight=None)
		feeder.set_paths(idList=read_ids('../cityscapes/splits_sample/val/list.txt'),
							ssDir="../output/dn")
							#ssDir="../cityscapes/unified/LabelGT/val")

		forward_model(model, feeder=feeder,
					  outputSavePath="../output/wtn/1113")
