import warnings, os
warnings.filterwarnings('ignore') 
project_folder = "/content/drive/MyDrive/Colab Notebooks/AE/SE-CNN"
os.chdir(project_folder)

import json, numpy, pickle, flair
import tensorflow as tf, tensorflow_hub as hub
from tqdm.notebook import tqdm_notebook as tq
from dataset import DataStreamer
from config import *

class Embeddings():
  'Features extract'
  def __init__(self,
               models_list,
               doc_embedding,
               batch_size,
               max_len,
               token_features,
               use_elmo
               ):
    super().__init__()
    self.elmo_model = hub.load('https://tfhub.dev/google/elmo/3')
    self.stack = flair.embeddings.StackedEmbeddings(models_list)
    self.doc_embedding = flair.embeddings.TransformerDocumentEmbeddings(doc_embedding)
    self.use_elmo = use_elmo
    self.batch = numpy.zeros((batch_size, max_len, token_features))

  def elmo_encoder(self, sentence):
    emb = tf.squeeze(self.elmo_model.signatures["default"](tf.constant(sentence))["elmo"]).numpy()
    return  emb.reshape(1,-1) if emb.ndim == 1 else emb

  def tr_encoder(self, sentence):
    s = flair.data.Sentence(sentence)
    sd = flair.data.Sentence(sentence)
    self.doc_embedding.embed(sd)
    self.stack.embed(s)
    return  numpy.array([ (token.embedding.cpu().numpy() + sd.get_embedding().cpu().numpy()) for token in s])

  def forward(self, sentences):
    self.batch *= 0.0
    for i, sent in enumerate(sentences):
      embs = self.tr_encoder(sent)
      if self.use_elmo:
        embs = numpy.concatenate((self.elmo_encoder([sent]), embs), axis = -1)
      self.batch[i,:embs.shape[0]] = embs  
    return self.batch[:len(sentences)]

class FeatureExtractor(Embeddings):
  'Extract and store features for later feeding the model'
  def __init__(self, models_list, doc_embedding, batch_size, max_len, token_features, taging_scheme, use_elmo):
    super().__init__(models_list, doc_embedding, batch_size, max_len, token_features, use_elmo)
    self.max_len = max_len   
    self.taging_scheme = taging_scheme

  def label_encoder(self, sentence, max_len, taging_scheme):
    seq = sentence.split()
    return [taging_scheme[tag] for tag in seq] + [0]*(max_len - len(seq))

  def extract_train_features(self, ldr, save_path, data_partition):
      i = 0
      for batch in tq(ldr, total = len(ldr)):
        x = self.forward(batch[0])
        y = [self.label_encoder(sent, self.max_len, self.taging_scheme) for sent in batch[1]]
        batch = [x ,y]
        with open(save_path + data_partition + '/batch{}'.format(i) , 'wb') as fp:
            pickle.dump(batch, fp)
        i += 1
      ldr.file.close()


def extraxt_features(train_file, test_file, models_list, doc_embedding, taging_scheme, embedding_path, buffer_size = 512, batch_size = 32, lower_case = True, max_len = 100, token_features = 1792, use_elmo = True):
  train_ldr = DataStreamer(train_file,
                            buffer_size = buffer_size,
                            batch_size = batch_size,
                            lower_case = lower_case)
  test_ldr = DataStreamer(test_file,
                            buffer_size = buffer_size,
                            batch_size = batch_size,
                            lower_case = lower_case)
  extractor = FeatureExtractor(models_list = pretrained_models,
                               doc_embedding = doc_embedding,
                                batch_size = batch_size,
                                max_len = max_len,
                                token_features = token_features,
                                taging_scheme = taging_scheme,
                                use_elmo = use_elmo
                                )
  extractor.extract_train_features(train_ldr, embedding_path, data_partition = 'train')
  extractor.extract_train_features(test_ldr, embedding_path, data_partition = 'test')


if __name__=="__main__":
  froberta = flair.embeddings.TransformerWordEmbeddings('AliAhmad001/absa-restaurant-froberta-base')
  doc_embedding = 'AliAhmad001/absa-restaurant-froberta-base'
  pretrained_models = [froberta]
  extraxt_features(train_file = TRAIN_FILE_se16,
       test_file = TEST_FILE_se16, 
       models_list = pretrained_models,
       doc_embedding = doc_embedding,
       taging_scheme = TAG2IDX, 
       embedding_path = EMBEDING_PATH, 
       buffer_size = BUFSIZE, 
       batch_size = BATCH_SIZE, 
       lower_case = LOWER_CASE, 
       max_len = MAX_LEN, 
       token_features = TOKEN_FEATURES, 
       use_elmo = USE_ELMO)
  