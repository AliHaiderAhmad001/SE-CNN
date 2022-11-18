VAL_RATE = 0.1
TOKEN_FEATURES = 1792 # ELMo: 1024 FRoBERTa: 768
BATCH_SIZE = 32
BUFSIZE = 512
EPOCHS = 85
MAX_LEN = 100
NUM_TAGS = 4 # 3 + PAD
LR = 0.0005
PATIENCE = None
CLIP_NORM = 0.001
DROPOUT_RATE = 0.5
USE_TPU = False
CRF_FLAG = False
USE_ELMO = True
LOWER_CASE = True
SHUFFLE = False
ONE_HOT_LABELS = False
SAVE_BEST = True

TAG2IDX = {'O':1, 'B-A':2, 'I-A':3, 'PAD':0}

ROOT = '/content/drive/MyDrive/Colab Notebooks/AE/SE-CNN/'

EMBEDING_PATH = ROOT + 'temp/'

MODEL_PATH_SE14 = ROOT + 'models/se14_model'
MODEL_PATH_SE15 = ROOT + 'models/se15_model'
MODEL_PATH_SE16 = ROOT + 'models/se16_model'
MODEL_PATH_LA14 = ROOT + 'models/la14_model'

TRAIN_FILE_SE14 = ROOT + 'datasets/Re14/train/seq.in'
TEST_FILE_SE14 = ROOT + 'datasets/Re14/test/seq.in'
TRAIN_FILE_SE15 = ROOT + 'datasets/Re15/train/seq.in'
TEST_FILE_SE15 = ROOT + 'datasets/Re15/test/seq.in'
TRAIN_FILE_SE16 = ROOT + 'datasets/Re16/train/seq.in'
TEST_FILE_SE16 = ROOT + 'datasets/Re16/test/seq.in'
TRAIN_FILE_LA14 = ROOT + 'datasets/La14/train/seq.in'
TEST_FILE_LA14 = ROOT + 'datasets/LA14/test/seq.in'









