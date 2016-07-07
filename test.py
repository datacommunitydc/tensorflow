from UDLC import *

udlc = UDLC()
udlc.load_mnist('./notMNIST.pickle')
udlc.format_datasets()
udlc.create_graph()
udlc.train_graph()