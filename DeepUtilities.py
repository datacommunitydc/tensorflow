
import string

class DeepUtilities():

  def __init__(self):
    self.vocabulary_size = len(string.ascii_lowercase) + 1 # [a-z] + ' '
    self.first_letter = ord(string.ascii_lowercase[0])

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