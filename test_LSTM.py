#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

from UDLC import *
from BatchGenerator import BatchGenerator

# LSTM
udlc = UDLC()
filename = udlc.maybe_download('text8.zip', 31344016)
udlc.read_data(filename)
udlc.format_lstm_file(filename)

print(udlc.char2id('a'), udlc.char2id('z'), udlc.char2id(' '), udlc.char2id('ï'))
print(udlc.id2char(1), udlc.id2char(26), udlc.id2char(0))


train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, 1)

print( train_batches.next_as_string())
print( train_batches.next_as_string())
print( valid_batches.next_as_string())
print( valid_batches.next_as_string())

print(9)