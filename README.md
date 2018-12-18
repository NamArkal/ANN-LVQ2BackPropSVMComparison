# ANN-LVQ2BackPropSVMComparison
Comparing performance of LVQ2, Back propagation and SVM on a classification problem

1.	Data set used is the diabetic retinopathy dataset from UCI. Diabetic retinopathy dataset has 19 attributes and is a binary classification problem. This dataset contains a higher rate of type 1 and type 2 errors.

2.	Keras by default uses backpropagation in its neural network. Support Vector Machines (SVM) and Learning Vector Quantization (LVQ) algorithms are implemented using the Scikit-Learn and Neupy libraries respectively.

3.	The algorithm parameters that will be changed to check sensitivity are learning rate, number of subclasses, input data (normalized and unnormalized data), epochs.

4.	The following is the implementation for LVQ2 with its parameters. The verbose from Neupy on calling fit() prints the training error which is (1 - n_correct_predictions / n_samples). Accuracy is calculated from this as (1 – training error).

```
	lvqnet = algorithms.LVQ2(
        		# number of features
        		n_inputs=19,

        		# number of data points that we want
        		# to have at the end
       		n_subclasses=None,

       	 	# number of classes
        		n_classes=2,

        		verbose=True,
        		show_epoch=20,
		
		# learning rate
        		step=0.001,
        		n_updates_to_stepdrop=150 * 100,
    	)
 ```

5.	The following is the implementation for back propagation with its parameters.

```
	network = models.Sequential()
	network.add(layers.Dense(units=19, activation = 'sigmoid', input_shape=(x_train.shape[1],)))
	network.add(layers.Dense(units=1))
	sgd = optimizers.SGD(lr = 0.001)

	network.compile(loss='mse',
              optimizer='sgd',
              metrics=['accuracy'])
```

6.	The following is the implementation for SVM which uses linear kernel.

```
	svclassifier = SVC(kernel='linear')
	svclassifier.fit(x_train, y_train)
```

7.	For the above parameters and normalized input, the following is the quality or accuracy of the model at the end of 50 epochs.

	LVQ2: 0.61
	Backprop: 0.696 
	SVM: 0.72

8.	When I changed the learning rate from 0.001 to 0.01, the following was the effect on the accuracy to Backprop and LVQ.

	LVQ2: 0.605
	Backprop: 0.70

9.	For LVQ2, if the number of subclasses for the classes is increased to 16 from 2, the accuracy improves from 0.61 to 0.64, the time taken to fit remains pretty much the same.

10.	So far, all observations have been for normalized data. For unnormalized data, the following are the accuracy observations. However, the time taken is lesser for the network to fit.

	LVQ2: 0.61
	Backprop: 0.72
	SVM: 0.73

11.	If the number of epochs is changed from 50 to 75 for LVQ and backprop, I observed that the accuracy and time taken does not vary a lot.

12.	The false negative rates (FNR) and false positive rates (FPR) for unnormalized input is given below.

	LVQ2: 0.45, 0.45
	Backprop: 0.56, 0.08
	SVM: 0.28, 0.21

13.	The same for normalized data is given below.

	LVQ2: 0.55, 0.16
	Backprop: 0.32, 0.16
	SVM: 0.42, 0.05

14.	To summarize the observations, LVQ2 works best for normalized data as it reduces false positive rate. Time taken for prediction is a little longer, but accuracy remains the same at 0.61. Backpropagation works best for unnormalized input with an accuracy of 0.72 and better FPR and FNR than with normalized data. SVM’s accuracy improves with non-normalized data. However, depending on the requirement, one can decide if they want to have really low FPR or have low enough FPR and FNR. Overall, SVM performs best for the diabetic retinopathy dataset.
