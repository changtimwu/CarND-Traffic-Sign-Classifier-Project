# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./pics/p2pic1.png "P2 PIC"
[image2]: ./pics/p2pic2.png "P2 PIC"
[image3]: ./pics/p2pic3.png "P2 PIC"
[websign1]: ./test_images/pedestrian.png "websign"
[websign2]: ./test_images/children.png "websign"
[websign3]: ./test_images/stop.png "websign"
[websign4]: ./test_images/speed_limit_30.png "websign"
[websign5]: ./test_images/construction.png "websign"

## Rubric Points

### Files Submitted

You're reading it! and here is a link to my [project code](https://github.com/changtimwu/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Exploration

#### Dataset Summary 

Firstly, I show number of each dataset.  The code for this step is contained in the 3rd code cell of the IPython notebook.  I used the pandas library to calculate summary statistics of the traffic signs data set:

![summary statistics][image1]

#### Exploratory Visualization

At the 4th code cell of the IPython notebook, I counts the number of each kind of traffic signs and show it in a table with `pandas`.  As you can see that the table is sorted,  the top three traffic signs in the dataset are `Speed limit(50kmh)`, `Speed limit(30km/h)`, and `Yield`.

![alt text][image2]

At the 6th code code, I pick an image of each kind of traffic sign and pack them together in a figure.

![alt text][image3]

### Design and Test a Model Architecture

#### Preprocessing

The code for this step is contained in the 5th & 7th code cell of the IPython notebook.

5th code cell contains various converters I've tried for preprocessing filters.  Unfortunately, all of them do not gain signficantly on accurancy over the grayscale filter.  So on 7th code cell, I still use grayscale.

### Model architecture

No data augumentation is used due to lack of time.  I'll add them later.
I use the classic LeNet model, which is taught in class without modification.  The code is located in the 8th cell of the ipython notebook.  It's consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6					|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16					|
| Flatten               | outputs 400                                   |
| Fully connected		| outputs 120 									|
| RELU					|												|
| Fully connected		| outputs 84  									|
| RELU					|												|
| Fully connected		| outputs 43   									|
| RELU					|												|
| Softmax				| outputs probabilities of 43 classes			|
 

The code for training the model is located in the 40th cell of the ipython notebook.  Here is summary of hyperparameters.
* EPOCHS = 200
* BATCH_SIZE = 256
* optimizer = `AdamOptimizer`
* learning_rate = 0.001

initial weights
* sigma = 0.05
* mean = 0

The code for calculating the accuracy of the model is located in the 36th & 41th cell of the Ipython notebook.

My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.949206349477 
* test set accuracy of 0.932462391312

I choose `LeNet` because it works well on hand written recognition application and most importantly it's the only architecture I'm currently familiar with.  I've tried adding some dropout layers but the test accuracy gets worse so I soon take them away.

Increasing EPOCH is a simple way to increase accuracy.  Eventhough accuracy grows very slowly after several epochs.  Batch size 256 is obtained via a try-and-error process.  Increasing it or decreasing it all get worse result.

### Test a Model on New Images

#### Acquiring New Images
Here are five German traffic signs that I found on the web:

![alt text][websign1] ![alt text][websign2] ![alt text][websign3] 
![alt text][websign4] ![alt text][websign5]

Their sizees are vary and some of them are protected with watermark.  I chose these pictures with purpose to test the capability of my model..

#### Performance on New Images

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Pedestrian       		| 30 km/h   									| 
| Children crossing		| Children crossing								|
| Stop					| Stop											|
| 30 km/h	      		| 30 km/h					 					|
| Road work				| Road work		      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%.  This compares favorably to the accuracy on the test set of 93.24%.

#### Model Certainty - Softmax Probabilities
The code for making predictions on my final model is located in the 47th cell of the Ipython notebook.

For the 1th image, this is a wrong prediction.  The model is 100% sure that this is 30km/h limit sign (probability of 1) but it's actually an pedestrian sign.  The probability of all other traffic signs are almost 0.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Speed limit (30km/h) 							| 
| 7.9278523e-28 		| Speed limit (70km/h)							|
| 2.9994467e-28			| Roundabout mandatory							|
| 5.5545305e-32	 		| Speed limit (120km/h)			 				|
| 1.0700292e-34		    | Speed limit (50km/h)							|



For the 2th image, the model is 100% sure that this is a children crossing sign (probability of 1) and it's indeed a children crossing sign.  The probability of 2nd possible traffic sign - 'Dangerous curve to the right' is too low.  We just treat it as 0.  The probability of all other traffic signs are all 0.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| children crossing 							| 
| 7.9278523e-28 		| Dangerous curve to the right					|
| .0					| N/A											|
| .0	      			| N/A							 				|
| .0				    | N/A			      							|



For the 3th image, the model is 100% sure that this is a stop sign (probability of 1) and the image surely is a stop sign. The probability of other traffic signs are almost 0.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Stop sign										| 
| .0     				| N/A	 										|
| .0					| N/A											|
| .0	      			| N/A							 				|
| .0				    | N/A			      							|



For the 4th image, the model is 100% sure that this is a Speed limit (30km/h) sign (probability of 1) and the image surely is a 30km/h speed limit. The probability of other traffic signs are all 0.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Speed limit (30km/h)							| 
| .0     				| N/A	 										|
| .0					| N/A											|
| .0	      			| N/A							 				|
| .0				    | N/A			      							|



For the 5th image, the model is 100% sure that this is a road work sign (probability of 1), and the image surely is a road work. The probability of other traffic signs are all 0.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Road work   									| 
| .0     				| N/A	 										|
| .0					| N/A											|
| .0	      			| N/A							 				|
| .0				    | N/A			      							|



