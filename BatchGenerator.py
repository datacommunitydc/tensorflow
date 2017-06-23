
import string
import numpy as np

class BatchGenerator(object):

  def __init__(self, text, batch_size=64, num_unrollings=10):

    self.vocabulary_size = len(string.ascii_lowercase) + 1 # [a-z] + ' '
    self.first_letter = ord(string.ascii_lowercase[0])

    self._text = text
    self._text_size = len(text)
    self._batch_size = batch_size
    self._num_unrollings = num_unrollings
    segment = self._text_size // batch_size
    self._cursor = [ offset * segment for offset in range(batch_size)]
    self._last_batch = self._next_batch()

  def char2id(self,char):
    if char in string.ascii_lowercase:
      return ord(char) - self.first_letter + 1
    elif char == ' ':
      return 0
    else:
      print('Unexpected character: %s' % char)
      return 0

  def id2char(self,dictid):
    if dictid > 0:
      return chr(dictid + self.first_letter - 1)
    else:
      return ' '

  def characters(probabilities):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return [self.id2char(c) for c in np.argmax(probabilities, 1)]

  def batches2string(batches):
    """Convert a sequence of batches back into their (most likely) string
    representation."""
    s = [''] * batches[0].shape[0]
    for b in batches:
      s = [''.join(x) for x in zip(s, characters(b))]
    return s

  def _next_batch(self):
    """Generate a single batch from the current cursor position in the data."""
    batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
    for b in range(self._batch_size):
      batch[b, char2id(self._text[self._cursor[b]])] = 1.0
      self._cursor[b] = (self._cursor[b] + 1) % self._text_size
    return batch

  def next(self):
    """Generate the next array of batches from the data. The array consists of
    the last batch of the previous array, followed by num_unrollings new ones.
    """
    batches = [self._last_batch]
    for step in range(self._num_unrollings):
      batches.append(self._next_batch())
    self._last_batch = batches[-1]
    return batches

  def next_as_string(self):
    return self.batches2string(self.next())

#
# train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
# valid_batches = BatchGenerator(valid_text, 1, 1)
#
# print(batches2string(train_batches.next()))
# print(batches2string(train_batches.next()))
# print(batches2string(valid_batches.next()))
# print(batches2string(valid_batches.next()))