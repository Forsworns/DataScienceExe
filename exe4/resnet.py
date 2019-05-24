from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.backend import squeeze
import numpy as np
import tensorflow as tf 
import pandas as pd
import os
import sys
import numpy as np
from load_data import load_data


RESNET = 'Resnet50'

class Resnet(object):
    def __init__(self,configs):
        self.name = configs.Resnet
        self.epochs = configs.epochs
        self.batch_size = configs.batch_size
        if not os.path.exists('model'):
            os.mkdir('model')
        if not os.path.exists('history'):
            os.mkdir('history')
        self.weight_name = 'model/{}-{}-{}.hdf5'.format(self.name,self.epochs,self.batch_size)
        self.history_name = 'history/{}-{}-{}.csv'.format(self.name,self.epochs,self.batch_size)
        self.model = None
        self.prebuilt(self.name)

    def prebuilt(self,name):
        self.model = Sequential()
        self.model.add(Dense(65,input_shape=(2048,),activation='softmax'))

    def train(self,data,label):
        self.model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
        checkpoint = ModelCheckpoint(self.weight_name, monitor='val_acc', save_best_only=True, save_weights_only=True, verbose=1)
        print(data.shape)
        print(label.shape)
        history = self.model.fit(data,label,epochs=self.epochs,batch_size=self.batch_size,shuffle=True,callbacks=[checkpoint])
        pd.DataFrame.from_dict(history.history).to_csv(self.history_name, float_format="%.5f", index=False)

    def test(self,data,label):
        self.model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
        loss_and_metric = self.model.evaluate(data,label,batch_size=32)
        print("{} on test set".format(self.name),loss_and_metric)

    def load(self):
        self.model.load_weights(self.weight_name)

    def print(self):
        self.model.summary()

flags = tf.app.flags
flags.DEFINE_string('Resnet',RESNET,"decide the Resnet version")
flags.DEFINE_integer('epochs',5,"epochs to train")
flags.DEFINE_integer('batch_size',32,"batch size to train")
flags.DEFINE_boolean('is_train',True,"train or test")
configs = flags.FLAGS

def main(_):
    for data in range(3):
        # 本身就是从resnet50种抽取的2048维deep feature
        X_src, y_src, X_tgt, y_tgt = load_data(data) 
        y_src = to_categorical(y_src)
        y_tgt = to_categorical(y_tgt)
        ## model
        resnet = Resnet(configs)
        resnet.print()
        if configs.is_train:
            resnet.train(X_src,y_src)
            resnet.test(X_tgt,y_tgt)
        else:
            resnet.load()
            resnet.test(X_tgt,y_tgt)

if __name__ == "__main__":
    tf.app.run()