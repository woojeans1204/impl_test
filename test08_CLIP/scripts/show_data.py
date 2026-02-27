import pickle
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data_batch = unpickle('../data/cifar-10-batches-py/data_batch_1')

images = data_batch[b'data']
labels = data_batch[b'labels']

image = images[0].reshape(3, 32, 32).transpose(1, 2, 0)

plt.imshow(image)
plt.title(f"Label: {labels[0]}")
# plt.show()
plt.savefig('sample.png')