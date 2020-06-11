# going to contain the utils for setting up, training, evaluating and saving the losses/accuracies for a given model
from cnn import *
from crnn import *

def vary_adamLR(train_path, test_path):
    x_train, y_train = np.load(train_path[0]), np.load(train_path[1])
    x_test, y_test = np.load(test_path[0]), np.load(test_path[1])

    #3DCNN
    #shuffled
    #img_size = (208, 50, 60, 3)
    #num_classes = 535
    #non-shuffled
    img_size = (195, 50, 60, 3)
    num_classes = 133

    #CRNN
    #img_size = (50, 60, 3)
    #num_classes = 535
                        
    rates = np.linspace(10e-8, 10e-5, 5)
    accuracies = []
    losses = []

    for i in range(len(rates)):
        print(f"Learning rate = {rates[i]:.3e}")
        cnn_model = ConvNet(img_size, num_classes, model_num = 1) # create a 3D-CNN
        model = cnn_model.get_model()

        #crnn_model = CRNN(img_size, num_classes, model_num = 1) # create a CRNN
        #model = crnn_model.get_model()
        
        opt = optimizers.Adam(learning_rate = rates[i])
        model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = model.fit(x_train, y_train.T, epochs = 7, batch_size = 9) #transpose y_train for 3D CNN only

        accuracies.append(hist.history['accuracy'])
        losses.append(hist.history['loss'])
        print('---------------------------------------------------')

    plot('Adam', rates, losses, accuracies)


def vary_sgdLR(train_path, test_path):
    x_train, y_train = np.load(train_path[0]), np.load(train_path[1])
    x_test, y_test = np.load(test_path[0]), np.load(test_path[1])

    img_size = (187, 50, 60, 3)
    num_classes = 194

    rates = np.linspace(10e-6, 10e-2, 3)
    accuracies = []
    losses = []

    for i in range(len(rates)):
        print(f"Learning rate = {rates[i]:.3e}")
        cnn_model = ConvNet(img_size, num_classes, model_num = 1) # create a 3D-CNN
        model = cnn_model.get_model()

        opt = optimizers.SGD(learning_rate = rates[i])
        model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = model.fit(x_train, y_train.T, epochs = 7, batch_size = 9)

        accuracies.append(hist.history['accuracy'])
        losses.append(hist.history['loss'])
        print('---------------------------------------------------')

    plot('SGD', rates, losses, accuracies)
