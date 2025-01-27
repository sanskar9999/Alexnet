In this project, i've tried implementing the AlexNet Architecture, using tensorflow Keras.

At first I tried to train the model on the same image dataset as the original AlexNet (ImageNet dataset)

But since the ImageNet API endpoints I was trying to use are no longer publicly accessible, I couldn't access URLs from ImageNet.

So I resorted to using the TensorFlow Datasets (CIFAR-10 dataset) instead of manually downloading ImageNet images. This is because this dataset was readily available and well-maintained.

I made sure to include proper data preprocessing and batching, and made sure to Implement a modified AlexNet architecture suitable for smaller images (277x277).

After training for 10 epoches, the model achieves a Test Accuracy of 77.34%

You can look at the sample images and as well as the classification matrix below to get an idea of how well the model performs.

We can see from the trainig accuracy plot that the model was still learning, without getting to diminishing returns, so we can say that there is still potential for improvement.

Perhaps in the future, if I get access to better hardware resources, I can traing a better model with a larger and higher quality dataset with more training time, but as of now this implementation stands as a great exercise for me to understand the basics of Deep Learning systems.
