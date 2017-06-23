from UDLC import *

udlc = UDLC()
udlc.load_mnist('./notMNIST.pickle')
udlc.format_datasets()
# L2 regularization
udlc.create_graph(l2_regularization=False)
udlc.train_graph(batch_processing=False)

blah = 9
