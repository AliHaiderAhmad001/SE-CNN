import numpy, pickle
import tensorflow as tf
            
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, embedding_path, data_partition, batch_list, n_tags = 4,
                 shuffle = False, one_hot_labels = False
    ):
      self.path = embedding_path + data_partition
      self.batch_list = batch_list
      self.n_tags = n_tags
      self.shuffle = shuffle
      self.one_hot_labels = one_hot_labels
      self.rnd = numpy.random.RandomState(0)
      self.on_epoch_end()

    def __len__(self):
        return len(self.batch_list)

    def __getitem__(self, index):
        with open (self.path + '/batch{}'.format(self.batch_list[index]), 'rb') as fp:
           batch = pickle.load(fp)
        X = batch[0]
        y = numpy.array(batch[1])
        if self.one_hot_labels:
          y = tf.keras.utils.to_categorical(numpy.reshape(y, (X.shape[0], X.shape[1], 1)),\
              num_classes = self.n_tags)
        return X, y

    def end_of_epoch(self):
        if self.shuffle:
          self.rnd.shuffle(self.batch_list)






