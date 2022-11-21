import warnings, os
warnings.filterwarnings('ignore') 
project_folder = "/content/drive/MyDrive/Colab Notebooks/AE/SE-CNN"
os.chdir(project_folder)

import tensorflow as tf
from tf2crf import CRF, ModelWithCRFLoss
import numpy as np
from metrics import report
from data_generator import DataGenerator
from config import *
import random as python_random

np.random.seed(2022)
python_random.seed(2022)
tf.random.set_seed(2022)

class Monitor(tf.keras.callbacks.Callback):
  def __init__(self, dataset, 
               save_best, 
               save_path, 
               patience):
      super(Monitor, self).__init__()
      self.dataset = dataset
      self.save_best = save_best
      self.f1 = 0.9
      self.path = save_path
      self.patience = patience    

  def on_epoch_end(self, epoch, logs = {}):
    y_true, y_pred = [], [] 
    for x, y in self.dataset:
      temp_p = self.model.predict(x, verbose = 0)
      if temp_p.ndim != 2:
        temp_p = tf.math.argmax(temp_p, axis = -1).numpy()
      y_pred.append(temp_p)
      if temp_p.shape[-1] != y.shape[-1]:
        y = tf.math.argmax(y, axis = -1).numpy()
      y_true.append(y)

    f1 = report(np.concatenate(y_true), np.concatenate(y_pred))
    """
    if (epoch + 1) == 50:
      self.model.optimizer.lr.assign(0.0001)
      print('Current LR:', self.model.optimizer.lr.read_value())
    """
    if f1 > self.f1:
      self.f1 = f1
      if self.patience: self.patience = 0
      if self.save_best:
        self.model.save(self.path)
        print('The model has been saved at epoch #{}'.format(epoch))
    elif self.patience:
      self.patience -= 1
      if self.patience == 0: self.model.stop_training = True


class SECNN(tf.keras.Model):
  def __init__(
        self,
        dropout_rate,
        crf_flag,
        n_tags,
  ):
      super(SECNN, self).__init__()

      self.mask = tf.keras.layers.Masking(mask_value = 0.0)

      self.dropout1 = tf.keras.layers.Dropout(0.1)
      self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
      
      self.conv1_1 = tf.keras.layers.Conv1D(256, 5,padding='same',strides=1,
                        activation= None,name='conv1_1')
      self.conv1_2 = tf.keras.layers.Conv1D(256, 3,padding='same',strides=1,
                        activation= None,name='conv1_2')
      self.conv1_3 = tf.keras.layers.Conv1D(512, 3,padding='same',strides=1,
                        activation= 'gelu',name='conv1_3')
      self.conv1_4 = tf.keras.layers.Conv1D(512, 3,padding='same',strides=1,
                        activation= 'gelu',name='conv1_4')
      self.conv1_5 = tf.keras.layers.Conv1D(512, 3,padding='same',strides=1,
                        activation= 'gelu',name='conv1_5')
      self.conv1_6 = tf.keras.layers.Conv1D(512, 3,padding='same',strides=1,
                        activation= 'gelu',name='conv1_6')

      self.linear_ae = tf.keras.layers.Dense(n_tags)

      self.crf_flag = crf_flag
      if crf_flag:
         self.crf = CRF(units = n_tags, dtype='float32')
  
  def call(self, inputs, training = False):
      x_emb = self.mask(inputs)
      x_emb = self.dropout1(x_emb, training = training)
      x_conv = tf.keras.layers.concatenate([self.conv1_1(x_emb), self.conv1_2(x_emb)], axis=-1)
      x_conv = tf.keras.activations.gelu(x_conv)
      x_conv = self.dropout2(x_conv, training = training)

      x_conv = self.conv1_3(x_conv)
      x_conv = self.dropout2(x_conv, training = training)
      
      x_conv = self.conv1_4(x_conv)
      x_conv = self.dropout2(x_conv, training = training)

      x_conv = self.conv1_5(x_conv)
      x_conv = self.dropout2(x_conv, training = training)

      x_conv = self.conv1_6(x_conv)

      x_logit = self.linear_ae(x_conv)

      if self.crf_flag:
         out = self.crf(x_conv)
      else:
         out = tf.keras.activations.softmax(x_logit, axis=-1)
      return out

def build_and_train(
        embedding_path,
        dropout_rate = 0.5,
        crf_flag = False,
        n_tags = 4,
        learning_rate = 0.0005,
        clipnorm = 0.001,
        shuffle = False,
        one_hot_labels = False,
        use_tpu = False,
        save_best = True, 
        save_path = None, 
        patience = None,
        epochs = 90,
        val_rate = 0.1
):  
    n_train_batch = len([name for name in os.listdir(project_folder + '/temp/train')])
    n_val_batch = int(val_rate * n_train_batch) # 9
    n_test_batch = len([name for name in os.listdir(project_folder + '/temp/test')])
    batch_train_list = [i for i in range(n_train_batch - 0)]
    #batch_val_list = [i for i in range(len(batch_train_list), n_train_batch)]
    batch_test_list = [i for i in range(n_test_batch)]
    train_gen = DataGenerator(embedding_path, 'train', batch_train_list, shuffle = shuffle,
                              n_tags = n_tags, one_hot_labels = one_hot_labels
                              )
    #val_gen = DataGenerator(embedding_path, 'train', batch_val_list, shuffle = shuffle,
    #                          n_tags = n_tags, one_hot_labels = one_hot_labels
    #                          )
    test_gen = DataGenerator(embedding_path, 'test', batch_test_list, shuffle = shuffle,
                              n_tags = n_tags, one_hot_labels = one_hot_labels
                              )

    callbacks = [
      Monitor(
        dataset = test_gen,
        save_best = save_best, 
        save_path = save_path, 
        patience = patience
      )
    ]

    if use_tpu:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
        strategy = tf.distribute.TPUStrategy(tpu)
        with strategy.scope():
            model = SECNN(dropout_rate = dropout_rate,
                          crf_flag = crf_flag,
                          n_tags = n_tags)
    else:
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        model = SECNN(dropout_rate = dropout_rate,
                      crf_flag = crf_flag,
                      n_tags = n_tags)

    optimizer = tf.keras.optimizers.Adam(
                      learning_rate = learning_rate,
                      epsilon = 1e-08,
                      clipnorm = clipnorm)
    if crf_flag:
        model = ModelWithCRFLoss(model, sparse_target = True)
        model.compile(optimizer = optimizer)
    else:
        model.compile(optimizer = optimizer,
                loss =  tf.keras.losses.SparseCategoricalCrossentropy())
    
    history = model.fit(x = train_gen, epochs = epochs, callbacks = callbacks)
    
    return model, history
  

if __name__=="__main__":
    build_and_train(
            embedding_path = EMBEDING_PATH,
            dropout_rate = DROPOUT_RATE,
            crf_flag = CRF_FLAG,
            n_tags = NUM_TAGS,
            learning_rate = LR,
            clipnorm = CLIP_NORM,
            shuffle = SHUFFLE,
            one_hot_labels = ONE_HOT_LABELS,
            use_tpu = USE_TPU,
            save_best = SAVE_BEST, 
            save_path = MODEL_PATH_SE16, 
            patience = PATIENCE,
            epochs = EPOCHS,
            val_rate = VAL_RATE
    ) 
