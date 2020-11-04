# **Traffic Sign Recognition** 

## Aim

### Use deep neural networks and convolutional neural networks to classify traffic signs.

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

[image4]: ./examples/1_double_curve.jpg "Traffic Sign 1"
[image5]: ./examples/2_no entry.jpg "Traffic Sign 2"
[image6]: ./examples/3_slippery_road.jpg "Traffic Sign 3"
[image7]: ./examples/4_speed limit 20.jpg "Traffic Sign 4"
[image8]: ./examples/5_stop.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set. 

I used the pandas library to calculate summary statistics of the traffic
signs data set:
* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Shape of traffic sign image = (32, 32, 3)
* Number of unique classes/labels in the data set = 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed accross various classes. 

<img src="examples\bar_graph.jpg" width="350">

On the Y axis, we have number of images from each class in our training set. On the X axis, we have the class-codes which range from 0 to 42. The class name corresponding to each class code can be seen [here](https://github.com/animesh-singhal/Traffic-Sign-Classifier/blob/master/signnames.csv).

Here's an example image for each class:

<img src="examples\all_classes.jpg" width="350">


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Here are the steps that I took for Pre-processing the images: 

1) Conversion of images to grayscale
>Reason:
>* If you don’t have more data, you are more prone to overfitting color images than grayscale.
>* It is more compute and memory intensive.
>
>If you have a color image (say 3 channels), it will have kernels of size k x k x 3 whereas grayscale image will have a kernel of size k x k x 1. Depending on the number of output kernels you specify, the number of parameters increases proportionally. For example, if you set k=3 and number of outputs to 64, an RGB image has 1728 parameters and 576 parameters for a grayscale image in the first convolution layer. If there are too many parameters, amount of data required to prevent overfitting also increases. 

I picked every image and took the intensity from each channel. Further I obtained the weighted average of those intensities using the weights [0.2989, 0.5870, 0.1140] so as to obtain a greyscale image. 

2) Normalization of images:
>Reason:
>Normalizing the data generally speeds up learning and leads to faster convergence. Hence it is a standard practice to best normalize your data to obtain a mean close to 0. It also brings the values of different features close to each other.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| 2D Convolution     	| 6 kernels, 5x5: kernel dimensions, 1x1 stride, valid padding. Output: 28x28x6 	|
| RELU					|Applied on the output of previous layer												|
| Max pooling	      	| 1 kernel (default), 2x2: kernel dimensions, 2x2 stride, valid padding, Output: 14x14x6  				|
| 2D Convolution     	| 16 kernels, 5x5: kernel dimensions, 1x1 stride, valid padding. Output: 10x10x16 	|
| RELU					|Applied on the output of previous layer												|
| Max pooling	      	| 1 kernel (default), 2x2: kernel dimensions, 2x2 stride, valid padding, Output: 5x5x16  				|
| Flatten		| Input: 5x5x16, Output: 400        									|
| Fully connected		| Input:400, Output: 120        									|
| RELU		| Applied on the output of previous layer        									|
| Dropout		| Applied on the output of previous layer        									|
| Fully connected		| Input:120, Output: 84        									|
| RELU		| Applied on the output of previous layer        									|
| Dropout		| Applied on the output of previous layer        									|
| Fully connected		| Input:84, Output: 43 (number of classes)        									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Here's what I did to train the network:

1) The above metioned model can give us logits which we can use to train the model. The model contains several parameters (weights) which are initialized randomly (range: [-1,1]). Using these, logits are calculated.
2) We begin by defining the cost function (which applies softmax function itself while calculating cost converting the logits to probabilities first, and then calculates the cost. )
3) Then we use Adam Optimizer to optimize this cost function and take the cost to it's minimum value possible. This operation basically updates the weights in our model, giving us new values of logits. The update is done such that there is a decrease in the cost function. 
4) Finally after training the entire model by breaking data set into various batch size and performing the optimization multiple times (in an attempt to reach global minima), we compare the logits to the labels. For each sample, if the activiation in the logit and label matches, our job is done!

Hyper-parameters used: 
| Hyperparameter |   Value	| Description
|:-----------:|:-----------:| :--------:|
| Epoch  | 40 | Number of times entire sample set was optimized (can be understood as iterations)   							| 
|Batch size|64| For each epoch, when the training set is huge, it is broken into different batches. Then optimization is performed on those batches one by one. Each batch updates the weights in within each epoch. Finally going through these different batches in each epoch, our model trains|
|Learning rate	|0.001| The _learning rate_ controls how quickly the model is adapted to the problem|
|Dropout|0.5| In order to avoid overfitting, we randomly hide some nodes in our network during training. This reduces the number of features which otherwise could lead to overfitting. Here, 0.5 would mean that we'll drop 50% nodes in our layer

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Here's the iterative approach that I choose to get the job done:
* I started out by quickly **setting up a LeNet architecture** and used the basic images without processing. I achieved a validation accuracy between 80-85%.
	> I used Lenet because it is a simple architecture which can be used to classfiy images. Plus, it works pretty quickly and hence can do the job at hand very efficiently.
* Next, I tried to **tune the hyper parameters**: epocs, batch_size, learning rate, dropouts. I tried creating graphs by varying one parameter and fixing others. Using those graphs, I picked out the best values of those parameters. This gave me a validation accuracy close to 90%. 
	>Different hyper parameters had different impacts as discussed below:
	> - If __Epocs__ are low, model would be under-fit.  If they are high, lot of computing resources could get wasted and the training time will increase.
	> - The use of small __batch sizes__ has been shown to improve generalization performance and optimization convergence. It requires a significantly smaller memory footprint, but needs a different type of processor to sustain full speed training. 
	> - **Learning rate** should not be very high otherwise the model could by-pass the minima even if it finds one. It shoudn't be too low because it would increase the time taken to convergence (and would require more epochs)
	> - **Dropouts** is a good technique to prevent overfitting. Dropping lots of nodes could result in an underfit model and dropping too less could lead to an overfit model 
* But, **preprocessing** the data was the dealbreaker (which I should have done initially but I was too excited to try the architecture)! After normalizing the data and making the images greyscale, it saw a surge in the validation accuracy and it went close to 96%. 
	 > This reduced the number of parameters and ensured faster convergence

**My final model results were:**
* training set accuracy of 99.8%
* validation set accuracy of 95.6%
* test set accuracy of 93.5%


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Double curve      		| Double curve   									| 
| No entry     			| No entry	 										|
| Slippery road					| Slippery road											|
| Speed limit (20km/h)	      		| Speed limit (20km/h)					 				|
| Stop			| General caution      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 93.5%. (Due to 5 number of images, any accuracy between (80%,100%) was not possible. We could have obtained an accuracy closer to our test accuracy if I would have taken test 6-8 images to classify.)

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located under the heading "Output Top 5 Softmax Probabilities For Each Image Found on the Web" in my Ipython notebook.

Below, I've provided two examples, one for each: correctly and incorrectly classified images. 

For the first image, the model is very confident that this is a Double curve sign (probability of 0.922), and the image does contain a Double curve sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .922         			| Double curve   									| 
| .045     				| Dangerous curve to the left 										|
| .009					| Slippery road											|
| .008	      			| Beware of ice/snow					 				|
| .006				    | Road narrows on the right      							|


For the 5th image the model is relatively less confident stating that this is a General caution sign (probability of 0.683). But, the image actually contain a Stop sign. The top five soft max probabilities were:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .683         			| General caution   									| 
| .023  				| Road work 										|
| .004					| Traffic signals										|
| .003	      			| Right-of-way at the next intersection					 				|
| .0002				    | Priority road      							|
It is surprising that the stop sign doesn't even appear in the top 5 probabilities. It could be because of the relatively lesser number of samples of that sign in our training set data. 


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


dit.io/).