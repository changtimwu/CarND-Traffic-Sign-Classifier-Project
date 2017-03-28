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

The code for this step is contained in the 5th code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by ...

My final training set had X number of images. My validation set and test set had Y and Z number of images.

The sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because ... To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an ....

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

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



