from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras import applications
import cv2, numpy as np
from sklearn.model_selection import train_test_split
import argparse
import pickle
from keras.models import load_model
from keras.models import Sequential, Model, load_model
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense


def save_model(model,training,path="./models/",model_name="run22"):
    print("saving model...")
    model.save(path+str(model_name)+'-model.h5')  # creates a HDF5 file 'my_model.h5'

    print("saving model weights...")
    model.save_weights(path+str(model_name)+'-weights.h5')

    filename = str(model_name)+'_history.pkl'
    print("saving history...")
    with open(filename, 'wb') as handle:
        pickle.dump(training.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

def VGG_16():
    img_rows, img_cols, img_channel = 64, 64, 3
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, img_channel))
    add_model = Sequential()
    add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    add_model.add(Dense(256, activation='relu'))
    add_model.add(Dropout(args.dropout_rate))
    add_model.add(Dense(5, activation='softmax'))

    return base_model, add_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="./data/", type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--dropout_rate', default=0.8, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--model_name', default=1e-4, type=str)
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    print("loading data matrices...")
    filename = DATA_DIR+'X_128.pickle'
    with open(filename, 'rb') as handle:
        X = pickle.load(handle)
    filename = DATA_DIR+'y_128.pickle'
    with open(filename, 'rb') as handle:
        hot_y = pickle.load(handle)
    print("splitting train and test...")
    X = (X-127.5)/255 # Normalize pixels 

    # split train/val/test
    train_X,test_X,train_y,test_y = train_test_split(X, hot_y, test_size=0.1,random_state=42)
    train_X,val_X,train_y,val_y = train_test_split(train_X, train_y, test_size=0.1,random_state=42)

    base_model, add_model = VGG_16()
    model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                     metrics=['accuracy'])
    _train = model.fit(train_X, train_y,batch_size=args.batch_size, nb_epoch=args.epochs, verbose=1,validation_data=(val_X, val_y))
    save_model(model,_train,path="my_models/",model_name=args.model_name)
