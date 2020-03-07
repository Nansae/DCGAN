# Deep Convolutional Generative Adversarial Network(DCGAN)
The repository has implemented the **Deep Convolutional Generative Adversarial Network(DCGAN)** and has been applied to the **CelebA dataset.**
   
## Concept
**Architecture**

![Architecture](https://user-images.githubusercontent.com/11286586/76143166-761bbe00-60b8-11ea-8317-dd601bec5c09.png)

**Architecture guidelines for stable Deep Convolutional GANs**
* Replace any pooling layers with strided convolutions and fractional-strided convolutions (generator).   
* Use batchnorm in both the generator and the discriminator.   
* Remove fully connected hidden layres for deeper architectures.   
* Use ReLU activation in generator for all layres except for the output, which uses Tanh.   
* Use LeakyReLU activation in the discriminator for all layers.   
 

## Files and Directories
* config.py : A file that stores various parameters and path settings.
* model.py : DCGAN's network implementation file consists of a DCGAN class.
* train.py : A file that loads train data and starts learning.
* utils.py : Various functions such as visualization and loading data
* checkpoints : A folder that stores learning results with a ckpt extension.
* sample_data :  Save sample image to verify the result of creation with DCGAN for every epoch.
* tensorboard : The trend of loss of generator and detector has been saved.
   
## Train
1. Download the CelebA dataset. ->https://www.kaggle.com/jessicali9530/celeba-dataset
2. The **read_images** function on **utils.py** has a subfolder that has an image corresponding to each class in the root folder.   
   ```
   ROOT_FOLDER
      |   
      |--------SUBFOLDER (Class 0)   
      |          |------image1.jpg   
      |          |------image2.jpg   
      |          |------etc..   
      |--------SUBFOLDER (Class 1)   
      |          |------image1.jpg   
      |          |------image2.jpg   
      |          |------etc..
   ```
      
   Please create a folder to store learning images and insert learning images to this standard. I used 10,856 images from 000001.jpg to 010856.jpg.
   The path i used is as follows.
   ```
   example: Place the training data in D:/data/celeba-dataset and Write D:/data on config.py
   ```
3. Run train.py
4. The result images are stored per epoch in the **sample_data folder**, and the result images are created with random vectors that you created before the learning begins.
 
## Result
![sample](https://user-images.githubusercontent.com/11286586/76104053-0d7c0500-6016-11ea-8d0d-2d68339e56b7.gif)

## Future work
* Walking in the latent sapce
* Extract features from images to create new images

## Reference
* Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).
* [Machine Learning Mastery](https://machinelearningmastery.com/how-to-interpolate-and-perform-vector-arithmetic-with-faces-using-a-generative-adversarial-network/)
