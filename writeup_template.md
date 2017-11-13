#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./random-sign.png  "Visualization"
[image20]: ./distribution.png  "Histogram"
[image21]: ./grayscale_converted.png  "Grayscale"
[image22]: ./augmented_image.png  "Augmented"
[image23]: ./top5_softmax.png  "Softmax"
[image24]: ./top5_softmax1.png  "Softmax"
[image25]: ./top5_softmax2.png  "Softmax"
[image26]: ./top5_softmax3.png  "Softmax"
[image27]: ./top5_softmax4.png  "Softmax"
[image28]: ./top5_softmax5.png  "Softmax"

[image2]: ./mysigns/1.png "Traffic Sign 1"
[image3]: ./mysigns/2.png "Traffic Sign 2"
[image4]: ./mysigns/3.png "Traffic Sign 3"
[image5]: ./mysigns/4.png "Traffic Sign 4"
[image6]: ./mysigns/5.png "Traffic Sign 5"
[image7]: ./mysigns/6.png "Traffic Sign 6"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
* The size of training set is 34799
* The size of the validation set is ?
* The size of validation examples = 4410
* The size of test set is ?
* The size of test set is 12630
* The shape of a traffic sign image is ?
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is ?
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

Here is an exploratory visualization of the data set. It pulls in a random set of eight images and labels them with the correct names in reference with the csv file to their respective id's.

![alt text][image1]

I also created a histogram of each image class and their count 

![alt text][image20]


I did some preprocessing of the images by converting to grayscale because it is better for machine than human  After the grayscale. Very Deep nets can be trained faster and generalize better when the distribution of activations is kept normalized.

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale

Here is an example of a traffic sign image after grayscaling.

![alt text][image21]

As a last step, I normalized the image data because ...
It help to speed the training and performance because of things like resources. Also added additional images to the datasets through randomized modifications.

I decided to generate additional data because ... 
The more diverse the data the performance of the network increases.

To add more data to the the data set, I used the following techniques because ... 
I used opencv Affine transformation which consist 
* Rotations (linear transformation)
* Translations (vector addition)
* Scale operations (linear transformation)

Here is an example of an original image and an augmented image:

![alt text][image22]

The difference between the original data set and the augmented data set is the following ... 
I increased the train dataset size to 89860 from 34799.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 16x16x64 				|
| Convolution 5x5		    | 1x1 stride, valid padding, outputs 14x14x6|
| Relu		| 			|
| Fully connected				| Input 14x14x6 = 400 output 120      									|
|	Relu				|												|
| droupout					|												|
| Fully connected					|			Input 120 output 84									|
| Relu					|									|
| droupout					|									|
| Fully connected						|			Input 84 output 43						|
| Softmax						|								|

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I trained the model using an Adam optimizer , learning rate of 0.00097 , batch size of 27.The epochs used was 27
To train the model, I started from a a well known architecture (LeNet) because of simplicity of implementation. 


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.


My final model results were:
* training set accuracy of ?
* Train Accuracy = 0.994
* validation set accuracy of ? 
* Valid Accuracy = 0.962  
* test set accuracy of ?
* Test Accuracy = 0.933

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* I used the Lenet to train and it gave very good result.
* What were some problems with the initial architecture?
* Since some of the classes had very few dataset it was really important to augment the data so that each class had at least * 1000 images. And after adding an extra convolution layer the model performed really good on Test images with accuracy of  * 99%
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

* A dropout rate of 30% and a learning rate of 0.00097 was selected after a few trial and errors. Training the model overall 
* takes around 15 mins.
* Which parameters were tuned? How were they adjusted and why?
* Batch size, learning rate, epoch were all parameters tuned along with the number of random 
* modifications to generate more image data was tuned.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
* The most important thing was to augment the dataset along with enough convolutions to capture 
* features will greatly improve speed of training and accuracy.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6] ![alt text][image7]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection  		| 1.00   									| 
| Bumpy road		| 1.00 										|
| Ahead Only					| 1.00											|
| General Caution      		| 1.00					 				|
| Go Straight or Left			| 1.00     							|
| Road Work			| 1.00     							|

The model was able to correctly guess all traffic signs, which gives an accuracy of 100%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

![alt text][image23]
![alt text][image24]
![alt text][image25]
![alt text][image26]
![alt text][image27]
![alt text][image28]

For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


