# [CS 230 - Deep Learning](https://cs230.stanford.edu/) (Stanford University)
## A Deep Learning Approach to Continuous ASL-English Translation
Collaborators:
- Sunny Shah (smshah94@stanford.edu; [Institute for Computational and Mathematical Engineering](https://icme.stanford.edu/))
- Teresa Noyola (tnoyola@stanford.edu; [Computer Science](https://cs.stanford.edu/))
- Georgia Sampaio (gsamp@stanford.edu; Computer Science)

This is a project on American Sign Language (ASL)-to-English translation using a combination of Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs). We try to improve on the architecture discussed in Bantupalli and Xie (2018). 

Baseline Model:
- We set up the [InceptionV3](https://arxiv.org/abs/1512.00567) network to process the frames of each video. For each video, we obtain a set of features obtained at the end of the final pool layer. These features will then be used as inputs in the LSTM network next.
- We also set up 3 LSTMs and a Dense layer. The loss function used in Bantupalli and Xie (2018) is the [Categorical Cross Entropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy), while the optimizer is the [Adam](https://arxiv.org/pdf/1412.6980.pdf) optimizer.

We used the ASL dataset from the (RWTH-Boston-104)[https://www-i6.informatik.rwth-aachen.de/web/Software/Databases/Signlanguage/details/rwth-boston-104/index.php], which has a total of 201 signs/sentences. We plan to use data augmentation to increase the size of our dataset, probably using the [Augmentor](https://augmentor.readthedocs.io/en/master/) library.



Sources:
- P. Dreuw, D. Rybach, T. Deselaers, M. Zahedi, and H. Ney. Speech Recognition Techniques for a Sign Language Recognition System. In Interspeech, pages 2513-2516, Antwerp, Belgium, August 2007. ISCA best student paper award Interspeech 2007.