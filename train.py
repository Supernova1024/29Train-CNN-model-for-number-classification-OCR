
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
from warnings import filterwarnings
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.image as mpimg
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
filterwarnings('ignore')


def new_build_model():
    classifier = Sequential()
    classifier.add(Conv2D(16,(7,7),input_shape=(128,128,3),activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2),strides=2)) #if stride not given it equal to pool filter size
    classifier.add(Conv2D(32,(3,3),activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2),strides=2))
    classifier.add(Flatten())
    # classifier.add(Dense(units=256,activation='relu'))
    # classifier.add(Dense(units=1,activation='sigmoid'))
    # classifier.add(Dense(10))
    # classifier.add(activation = 'softmax')
    classifier.add(Dense(units=10,activation='softmax'))
    classifier.add(Dense(units=10,activation='softmax'))
    adam = tensorflow.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # classifier.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy'])
    classifier.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
    return classifier

def prepare_dataset(train, validation, test):
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.1,
                                       zoom_range=0.1,
                                       horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    #Training Set
    train_set = train_datagen.flow_from_directory(train,
                                                 target_size=(128,128),
                                                 batch_size=5,
                                                 class_mode='categorical')
    #Validation Set
    validation_set = test_datagen.flow_from_directory(validation,
                                               target_size=(128,128),
                                               batch_size = 5,
                                               class_mode='categorical',
                                               shuffle=False)
    #Test Set /no output available
    test_set = test_datagen.flow_from_directory(test,
                                                target_size=(128,128),
                                                batch_size=5,
                                                shuffle=False)
    return train_set, validation_set, test_set

def model_fit(classifier, train_set, validation_set):
    classifier.fit_generator(train_set,
                            steps_per_epoch=10, 
                            epochs = 10,
                            validation_data = validation_set,
                            validation_steps = 20, 
                            #callbacks=[tensorboard]
                            );
    return classifier

def calculate_misclassified(classifier, validation_set):
    validation_set.reset
    ytesthat = classifier.predict_generator(validation_set)
    df = pd.DataFrame({
        'filename':validation_set.filenames,
        'predict':ytesthat[:,0],
        'y':validation_set.classes
    })

    pd.set_option('display.float_format', lambda x: '%.5f' % x)
    df['y_pred'] = df['predict']>0.5
    df.y_pred = df.y_pred.astype(int)
    df.head(10)
    print(df)

    misclassified = df[df['y']!=df['y_pred']]
    print('Total misclassified image from 10 Validation images : %d'%misclassified['y'].count())

def model_evaluate(classifier, train_set, validation_set):
    # Model Accuracy
    x1 = classifier.evaluate_generator(train_set)
    x2 = classifier.evaluate_generator(validation_set)

    print('Training Accuracy  : %1.2f%%     Training loss  : %1.6f'%(x1[1]*100,x1[0]))
    print('Validation Accuracy: %1.2f%%     Validation loss: %1.6f'%(x2[1]*100,x2[0]))

def training_new_model(model_path, train_set, validation_set, test_set):
    classifier = new_build_model()
    classifier = model_fit(classifier, train_set, validation_set)
    classifier.save(model_path)
    classifier = load_model(model_path)
    return classifier

def traing_old_model(model_path, train_set, validation_set, test_set):
    classifier = load_model(model_path)
    classifier = model_fit(classifier, train_set, validation_set)
    classifier.save(model_path)
    classifier = load_model(model_path)
    return classifier

def main(model_path, train_folder, validation_folder, test_folder):
    train_set, validation_set, test_set = prepare_dataset(train_folder, validation_folder, test_folder)

    classifier = training_new_model(model_path, train_set, validation_set, test_set)
    # classifier = traing_old_model(model_path, train_set, validation_set, test_set)
    
    calculate_misclassified(classifier, validation_set)
    classifier.summary()

    # ### Model Performance on Unseen Data
    print(train_set.class_indices)
    fig=plt.figure(figsize=(15, 6))
    columns = 3
    rows = 3
    for i in range(columns*rows):
        fig.add_subplot(rows, columns, i+1)
        file = 'test/'+test_set.filenames[np.random.choice(range(5))]
        img1 = image.load_img(file, target_size=(128,128))
        img = image.img_to_array(img1)
        img = img/255
        img = np.expand_dims(img, axis=0)
        prediction = classifier.predict(img, batch_size=None,steps=1) #gives all class prob.
        print("==prediction",file,  prediction[:,:])
        # if(prediction[:,:]>0.5):
        #     value ='machine :%1.2f'%(prediction[0,0])
        #     plt.text(20, 58,value,color='red',fontsize=10,bbox=dict(facecolor='white',alpha=0.8))
        # else:
        #     value ='human :%1.2f'%(1.0-prediction[0,0])
        #     plt.text(20, 58,value,color='red',fontsize=10,bbox=dict(facecolor='white',alpha=0.8))

        plt.imshow(img1)
    plt.show()

    model_evaluate(classifier, train_set, validation_set)

if __name__ == '__main__':
    model_path = 'resources/ocr_model.h5'
    train_folder = 'train'
    validation_folder = 'validation'
    test_folder = 'test'
    main(model_path, train_folder, validation_folder, test_folder)
