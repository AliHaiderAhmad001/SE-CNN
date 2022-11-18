import numpy as np
class DataStreamer():
    ' Read data from files '
    def __init__(self, file, buffer_size = 512, batch_size = 32, shuffle = False,
                 lower_case = True, seed = 0):
        self.file = open(file, "r")
        for count, _ in enumerate(self.file):
            pass
        self.file.seek(0)
        self.data_size = (count + 1)//2
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lower_case = lower_case
        self.rnd = np.random.RandomState(seed)
        self.the_buffer = []  
        self.ptr = 0
        self.flag = False
        self.fetch_to_buffer()

    def __len__(self):
        return int(np.ceil(self.data_size/self.batch_size))
        
    def fetch_to_buffer(self):
        self.the_buffer = []
        self.ptr = 0
        ct = 0  
        while ct < self.buffer_size:
          sent = self.file.readline().strip()
          if sent == "":
            self.reset()
            break
          else:
            tags = self.file.readline().strip()
            if self.lower_case:
              sent = sent.lower()
            self.the_buffer.append([sent, tags]) # list<str, str>, e.g. ['bla bla','bla bla']
            ct += 1
        
        if self.shuffle:
          self.rnd.shuffle(self.the_buffer)  # in-place

    def __iter__(self):
       return self

    def reset(self):
        self.file.seek(0)

    def __next__(self):
        if self.flag:
          self.flag = False
          raise StopIteration

        if self.ptr + self.batch_size > self.buffer_size:
           self.fetch_to_buffer()

        x, y = [], []
        for example in self.the_buffer[self.ptr:self.ptr + self.batch_size]:
          x.append(example[0])
          y.append(example[1])
        self.ptr += self.batch_size
        if len(y) != self.batch_size: 
          self.fetch_to_buffer()  
          if len(y) == 0:
            raise StopIteration
          else:
            self.flag = True
        return x,y




