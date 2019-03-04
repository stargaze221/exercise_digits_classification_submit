##  Handwritten Digits Classification.
The digits classification is formulated as a supervised learning problem. The handwritten data (images and labels) are downloaded from [Yann LeCun's homepage](http://yann.lecun.com/exdb/mnist "yann.lecun.com"). A convolutional neural network (CNN) maps the images to probability distributions on the labels (0, 1,..., 9). The stochastic gradient descent method is employed to minimize the cross-entropy between the label and the probability distributions. A sampled based optimization tool ([Hyperopt](https://hyperopt.github.io/hyperopt/), a Bayesian optimizer) was employed for hyper-parameter tunning. Further details on the implementation are listed as follows:

1. The stochastic gradient descent algorithm is implemented using Python3 and Pytorch. The list of apps which can be installed using pip is as follows:
    <p>python3, wget, gzip, idx2numpy, numpy, pytorch, hyperopt</p>

2. Despite the automized hyper-parameter tunning using Hyperopt, we need to choose a few hyper-parameter since it is not feasible to search the entire hyper-parameter space. I have selected the number of layers, l2 regularization weight, and the number of epochs in the hyper-parameter optimization based on the following list of heuristics:
    <p> a. What makes neural networks deep is having multiple layers. There are many theoretical efforts to show the deep-layered structure is useful. For example, [1] shows that each layer in a deep neural network (DNN) act as information bottleneck to decrease the effect of the nuisance in classification tasks. Also more layers increase its capability to approximate arbitrary functions (model complexity).  
    </p>
    <p> b. l2 regularization on the parameters is a well-known method to prevent overfitting; in turn, the model has a better generalization.
    </p>
    <p> c. Early stopping is also popularly used to enhance generalization.
    </p>
    The choice of the hyper-parameter can consider the trade-off between model complexity and generalization. To run the script for hyperparameter tunning, run the Python script in the root directory of the repository as follows:

    > $python main.py optimize hyperparam

    The hyper-parameter optimization took about 6 hours to run 100 iterations with Nvidia GeForce GTX 1060. The result of the optimization is saved in 'hyper_param.json'. The exploration in hyper-parameter space during the 100 iterations is shown in the charts located in the bottom.

3. With determined hyper-parameters in 'hyper_param.json,' you can run the training with the command below.

    > $python main.py optimize classifier

    The determined parameter after the training is saved in 'Saved_CNN_Mdl.pt.' 
    
4. Testing the trained model with the test dataset can be done with the command below.

    > $python main.py test

    The classification error with the files: 'hyper_param.json' and 'Saved_CNN_Mdl.pt' is
    <p> Valication error:  1.09%
    </p>

    <p> Test error:  1.32% 
    </p>

5. Compared to the state of the art result in [2] which has only 0.21% test error, the implementation in this repository falls behind. In [2], drop-out is introduced as a way to improve generalization (a way to impose regularization). Furthermore, in [3], the equivalence of the deep neural network (DNN) with drop-out to the Gaussian process (GP). So the DNN with drop-out behaves similarly to the Bayesian model, GP. This suggests we might be able to improve the performance by employing the drop-out approach.

5. However, there is another significant difference. The difference is on generating training data. As described in Section 6 in [2], the original training data are manipulated with scaling, rotation and cropping to create more training data sample. The manipulation seems to be crucial for the incredibly good performance with an error rate at 0.21%. We might be able to argue that those manipulations in [2] were done based on human's prior knowledge or domain knowledge such that the classification should be robust to scaling, rotation, and translation. 



### References
[1] Achille, Alessandro, and Stefano Soatto. "Emergence of invariance and disentanglement in deep representations." The Journal of Machine Learning Research 19.1 (2018): 1947-1980.

[2] Wan, Li, et al. "Regularization of neural networks using dropconnect." International conference on machine learning. 2013.

[3] Gal, Yarin, and Zoubin Ghahramani. "Dropout as a bayesian approximation: Representing model uncertainty in deep learning." international conference on machine learning. 2016.

### Charts on the exploration by Hyperopt to minimize the validation error.

#### Chart 1. Effect of L2 regularization on the validation error.
![Chart 1](/charts/chart1.png)

#### Chart 2. Effect of number of layers on the validation error.
![Chart 2](/charts/chart2.png)

#### Chart 3. Effect of total epochs on the validation error.
![Chart 3](/charts/chart3.png)