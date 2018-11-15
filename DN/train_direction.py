import direction_model
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

VGG_MEAN = [103.939, 116.779, 123.68]

tf.set_random_seed(0)

def initialize_model(outputChannels, wd=None, modelWeightPaths=None):
	fuseChannels=256
	params = {"direction/conv1_1": {"name": "direction/conv1_1", "shape": [3,3,3,64], "std": None, "act": "relu"},
			"direction/conv1_2": {"name": "direction/conv1_2", "shape": [3,3,64,64], "std": None, "act": "relu"},
			"direction/conv2_1": {"name": "direction/conv2_1", "shape": [3,3,64,128], "std": None, "act": "relu"},
			"direction/conv2_2": {"name": "direction/conv2_2", "shape": [3,3,128,128], "std": None, "act": "relu"},
			"direction/conv3_1": {"name": "direction/conv3_1", "shape": [3,3,128,256], "std": None, "act": "relu"},
			"direction/conv3_2": {"name": "direction/conv3_2", "shape": [3,3,256,256], "std": None, "act": "relu"},
			"direction/conv3_3": {"name": "direction/conv3_3", "shape": [3,3,256,256], "std": None, "act": "relu"},
			"direction/conv4_1": {"name": "direction/conv4_1", "shape": [3,3,256,512], "std": None, "act": "relu"},
			"direction/conv4_2": {"name": "direction/conv4_2", "shape": [3,3,512,512], "std": None, "act": "relu"},
			"direction/conv4_3": {"name": "direction/conv4_3", "shape": [3,3,512,512], "std": None, "act": "relu"},
			"direction/conv5_1": {"name": "direction/conv5_1", "shape": [3,3,512,512], "std": None, "act": "relu"},
			"direction/conv5_2": {"name": "direction/conv5_2", "shape": [3,3,512,512], "std": None, "act": "relu"},
			"direction/conv5_3": {"name": "direction/conv5_3", "shape": [3,3,512,512], "std": None, "act": "relu"},
			"direction/fcn5_1": {"name": "direction/fcn5_1", "shape": [5,5,512,512], "std": None, "act": "relu"},
			"direction/fcn5_2": {"name": "direction/fcn5_2", "shape": [1,1,512,512], "std": None, "act": "relu"},
			"direction/fcn5_3": {"name": "direction/fcn5_3", "shape": [1,1,512,fuseChannels], "std": 1e-2, "act": "relu"},
			"direction/upscore5_3": {"name": "direction/upscore5_3", "ksize": 8, "stride": 4, "outputChannels": fuseChannels},
			"direction/fcn4_1": {"name": "direction/fcn4_1", "shape": [5,5,512,512], "std": None, "act": "relu"},
			"direction/fcn4_2": {"name": "direction/fcn4_2", "shape": [1,1,512,512], "std": None, "act": "relu"},
			"direction/fcn4_3": {"name": "direction/fcn4_3", "shape": [1,1,512,fuseChannels], "std": 1e-3, "act": "relu"},
			"direction/upscore4_3": {"name": "direction/upscore4_3", "ksize": 4, "stride": 2, "outputChannels": fuseChannels},
			"direction/fcn3_1": {"name": "direction/fcn3_1", "shape": [5,5,256,256], "std": None, "act": "relu"},
			"direction/fcn3_2": {"name": "direction/fcn3_2", "shape": [1,1,256,256], "std": None, "act": "relu"},
			"direction/fcn3_3": {"name": "direction/fcn3_3", "shape": [1,1,256,fuseChannels], "std": 1e-4, "act": "relu"},
			"direction/fuse3_1": {"name": "direction/fuse_1", "shape": [1,1,fuseChannels*3, 512], "std": None, "act": "relu"},
			"direction/fuse3_2": {"name": "direction/fuse_2", "shape": [1,1,512,512], "std": None, "act": "relu"},
			"direction/fuse3_3": {"name": "direction/fuse_3", "shape": [1,1,512,outputChannels], "std": None, "act": "lin"},
			"direction/upscore3_1": {"name": "direction/upscore3_1", "ksize": 8, "stride": 4, "outputChannels":outputChannels}}

	return direction_model.Network(params, wd=wd, modelWeightPaths=modelWeightPaths)

def forward_model(model, feeder, outputSavePath):
	with tf.Session() as sess:
		tfBatchImages = tf.placeholder("float", shape=[None, 512, 1024, 3])
		'''QD
		tfBatchSS = tf.placeholder("float", shape=[None, 512, 1024])
		tfBatchSSMask = tf.placeholder("float", shape=[None, 512, 1024])
		'''

		with tf.name_scope("model_builder"):
			print "attempting to build model"
			'''QD
			model.build(tfBatchImages, tfBatchSS, tfBatchSSMask)
			'''
			model.build(tfBatchImages)
			print "built the model"
		sys.stdout.flush()

		init = tf.initialize_all_variables()
		sess.run(init)

		for i in range(int(math.floor(feeder.total_samples() / batchSize))):
			# QD imageBatch, ssBatch, ssMaskBatch, idBatch = feeder.next_batch()
			imageBatch, idBatch = feeder.next_batch()

			outputBatch = sess.run(model.output, feed_dict={tfBatchImages: imageBatch})

			for j in range(len(idBatch)):
				print "outputSavePath :%s" % (outputSavePath)
				outputFilePath = os.path.join(outputSavePath, idBatch[j].lstrip('/')+'.mat')
				outputFileDir = os.path.dirname(outputFilePath)
				print "into batch outputFilePath :%s" % (outputFilePath)
				if not os.path.exists(outputFileDir):
					os.makedirs(outputFileDir)

				sio.savemat(outputFilePath, {"dir_map": outputBatch[j]}, do_compression=True)

				print "processed image %d out of %d"%(j+batchSize*i, feeder.total_samples())

def train_model(model, outputChannels, learningRate, trainFeeder, valFeeder, modelSavePath=None, savePrefix=None, initialIteration=1):
	
	with tf.Session() as sess:
		
		# training input 
		# 
		
		''' QD
		tfBatchImages = tf.placeholder("float", shape=[None, 512, 1024, 3])
		tfBatchGT = tf.placeholder("float", shape=[None, 512, 1024, 2])
		tfBatchWeight = tf.placeholder("float", shape=[None, 512, 1024])
		tfBatchSS = tf.placeholder("float", shape=[None, 512, 1024])
		tfBatchSSMask = tf.placeholder("float", shape=[None, 512, 1024])
		'''
		tfBatchImages = tf.placeholder("float", shape=[None, 512, 1024, 3])
		tfBatchGT = tf.placeholder("float", shape=[None, 512, 1024, 2])
		tfBatchWeight = tf.placeholder("float", shape=[None, 512, 1024])
		
		# build model
		with tf.name_scope("model_builder"):
			print "attempting to build model"
			'''QD
			model.build(tfBatchImages, tfBatchSS, tfBatchSSMask)
			'''
			model.build(tfBatchImages)
			print "built the model"

		sys.stdout.flush()
		
		''' QD
		loss = lossFunction.angularErrorLoss(pred=model.output, 
											   gt=tfBatchGT,
											   weight=tfBatchWeight,
											   ss=tfBatchSS,
											   outputChannels=outputChannels)
		angleError = lossFunction.angularErrorTotal(pred=model.output,
													  gt=tfBatchGT,
													  weight=tfBatchWeight,
													  ss=tfBatchSS,
													  outputChannels=outputChannels)
		numPredicted = lossFunction.countTotal(ss=tfBatchSS)
		numPredictedWeighted = lossFunction.countTotalWeighted(ss=tfBatchSS, weight=tfBatchWeight)
		exceed45 = lossFunction.exceedingAngleThreshold(pred=model.output,
														gt=tfBatchGT,
														ss=tfBatchSS,
														threshold=45.0,
														outputChannels=outputChannels)
		exceed225 = lossFunction.exceedingAngleThreshold(pred=model.output,
														gt=tfBatchGT,
														ss=tfBatchSS,
														threshold=22.5,
														outputChannels=outputChannels)
		'''
		loss = lossFunction.angularErrorLoss(pred=model.output, 
											   gt=tfBatchGT,
											   weight=tfBatchWeight,
											   outputChannels=outputChannels)
		angleError = lossFunction.angularErrorTotal(pred=model.output,
													  gt=tfBatchGT,
													  weight=tfBatchWeight,
													  outputChannels=outputChannels)
		
		train_op = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(loss=loss)

		init = tf.initialize_all_variables()

		sess.run(init)
		iteration = initialIteration

		while iteration < 10000:
			batchLosses = []
			totalAngleError = 0
			totalExceed45 = 0
			totalExceed225 = 0
			totalPredicted = 0
			totalPredictedWeighted = 0
			print "this is iteration:%d" % (iteration)
			
			print "begining val..."
			
			for k in range(int(math.floor(valFeeder.total_samples() / batchSize))):
				
				'''QD 
				imageBatch, gtBatch, weightBatch, ssBatch, ssMaskBatch, _ = valFeeder.next_batch()
				'''
				
				imageBatch, gtBatch, weightBatch = valFeeder.next_batch()
				
				print"this is batch:%d" % (k)
				
				'''QD
				batchLoss, batchAngleError, batchPredicted, batchPredictedWeighted, batchExceed45, batchExceed225 = sess.run([loss,
																															angleError,
																															numPredicted,
																															numPredictedWeighted,
																															exceed45,
																															exceed225],
																															feed_dict={tfBatchImages: imageBatch,
																																	   tfBatchGT: gtBatch,
																																	   tfBatchWeight: weightBatch,
																																	   tfBatchSSMask: ssMaskBatch,
																																	   tfBatchSS: ssBatch,})
				'''
				batchLoss, batchAngleError,  = sess.run([loss,
														angleError],
														feed_dict={tfBatchImages: imageBatch,
																   tfBatchGT: gtBatch,
																   tfBatchWeight: weightBatch})
				
				
				
				batchLosses.append(batchLoss)
				totalAngleError += batchAngleError
				'''QD
				totalPredicted += batchPredicted
				totalPredictedWeighted += batchPredictedWeighted
				totalExceed45 += batchExceed45
				totalExceed225 += batchExceed225
				'''
			if np.isnan(np.mean(batchLosses)):
				print "LOSS RETURNED NaN"
				sys.stdout.flush()
				return 1
			'''QD
			print "%s Itr: %d - val loss: %.3f, angle MSE: %.3f, exceed45: %.3f, exceed22.5: %.3f" % (
				  time.strftime("%H:%M:%S"),
				  iteration,
				  float(np.mean(batchLosses)),
				  totalAngleError / totalPredictedWeighted,
				  totalExceed45 / totalPredicted,
				  totalExceed225 / totalPredicted)
			'''
			print "%s Itr: %d - val loss: %.3f, angle MSE: %.3f" % (
				  time.strftime("%H:%M:%S"),
				  iteration,
				  float(np.mean(batchLosses)),
				  totalAngleError)
			print "end val..."
			sys.stdout.flush()

			if (iteration > 0 and iteration % 5 == 0) or checkSaveFlag(modelSavePath):
				modelSaver(sess, modelSavePath, savePrefix, iteration)
			
			print "begining train..."
			
			for j in range(int(math.floor(trainFeeder.total_samples() / batchSize))):
				
				print "running batch %d"%(j)
				
				'''QD
				imageBatch,gtBatch,weightBatch,ssBatch,ssMaskBatch, _ = trainFeeder.next_batch()
				
				sess.run(train_op, feed_dict={tfBatchImages: imageBatch,
											  tfBatchGT: gtBatch,
											  tfBatchWeight: weightBatch,
											  tfBatchSS: ssBatch,
											  tfBatchSSMask: ssMaskBatch})
				'''
				imageBatch,gtBatch,weightBatch = trainFeeder.next_batch()
				
				sess.run(train_op, feed_dict={tfBatchImages: imageBatch,
											  tfBatchGT: gtBatch,
											  tfBatchWeight: weightBatch})
			iteration += 1

def modelSaver(sess, modelSavePath, savePrefix, iteration, maxToKeep=5):
	
	allWeights = {}
	
	for name in [n.name for n in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]:
		param = sess.run(name)
		nameParts = re.split('[:/]', name)
		saveName = nameParts[-4]+'/'+nameParts[-3]+'/'+nameParts[-2]
		allWeights[saveName] = param

	weightsFileName = os.path.join(modelSavePath, savePrefix+'_%03d'%iteration)

	sio.savemat(weightsFileName, allWeights)


def checkSaveFlag(modelSavePath):
	flagPath = os.path.join(modelSavePath, 'saveme.flag')

	if os.path.exists(flagPath):
		return True
	else:
		return False


if __name__ == "__main__":
	
	outputChannels = 2
	classType = 'unified_CR'
	indices = [0,1,2,3,4,5,6,7]
	# 0=car, 1=person, 2=rider, 3=motorcycle, 4=bicycle, 5=truck, 6=bus, 7=train
	savePrefix = "direction_" + classType + "_unified_CR_pretrain"
	train = False
	#train = True
	
	if train:
		
		batchSize = 2
		learningRate = 1e-5
		wd = 1e-5
		modelWeightPaths = ["../model/direction/32s_1110.mat"]
		initialIteration = 1
		model = initialize_model(outputChannels=outputChannels, wd=wd, modelWeightPaths=modelWeightPaths)

		trainFeeder = Batch_Feeder(dataset="cityscapes",
									indices=indices,
									train=train,
									batchSize=batchSize,
									padWidth=None,
									padHeight=None,
									flip=True,
									keepEmpty=False)
		
		'''	QD
		trainFeeder.set_paths(idList=read_ids('../cityscapes/splits/train/list.txt'),
							  imageDir="../cityscapes/leftImg8bit/train",
							  gtDir="../cityscapes/unified/iGTFull/train",
							  ssDir="../cityscapes/unified/ssMaskFinePSP/train")
		'''
		trainFeeder.set_paths(idList=read_ids('../cityscapes/splits/train/list.txt'),
							  imageDir="../cityscapes/leftImg8bit/train",
							  gtDir="../cityscapes/unified/LabelGT/train")
		valFeeder = Batch_Feeder(dataset="cityscapes",
								indices=indices,
								train=train,
								batchSize=batchSize,
								padWidth=None,
								padHeight=None)
		
		''' QD
		valFeeder.set_paths(idList=read_ids('../cityscapes/splits/val/list.txt'),
							imageDir="../cityscapes/leftImg8bit/val",
							gtDir="../cityscapes/unified/iGTFull/val",
							ssDir="../cityscapes/unified/ssMaskFinePSP/val")
		'''
		valFeeder.set_paths(idList=read_ids('../cityscapes/splits/val/list.txt'),
							imageDir="../cityscapes/leftImg8bit/val",
							gtDir="../cityscapes/unified/LabelGT/val")
							
		train_model(model=model,
					outputChannels=outputChannels,
					learningRate=learningRate,
					trainFeeder=trainFeeder,
					valFeeder=valFeeder,
					modelSavePath="../model/direction",
					savePrefix=savePrefix,
					initialIteration=initialIteration)
	else:
		batchSize = 2
		modelWeightPaths = ["../model/direction/direction_unified_CR_unified_CR_pretrain_019.mat"]

		model = initialize_model(outputChannels=outputChannels, wd=0, modelWeightPaths=modelWeightPaths)

		feeder = Batch_Feeder(dataset="cityscapes",
								indices=indices,
								train=train,
								batchSize=batchSize,
								padWidth=None,
								padHeight=None)
								
		feeder.set_paths(idList=read_ids("../cityscapes/splits_sample/val/list.txt"),
						 imageDir="../cityscapes/leftImg8bit/val")

		forward_model(model, feeder=feeder, outputSavePath="/usr/qd/pdeep/output/dn")
