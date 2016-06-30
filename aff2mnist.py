#!/usr/bin/python

"""
Convert affNIST training/validation and testing files to MNIST binary data files
download from here:
http://www.cs.toronto.edu/~tijmen/affNIST/
http://www.cs.toronto.edu/~tijmen/affNIST/32x/transformed/training_and_validation_batches.zip
http://www.cs.toronto.edu/~tijmen/affNIST/32x/transformed/test_batches.zip
"""

import os
import scipy.io.matlab as sim
import numpy as np

def tohex(x):
	str = ''
	while (x != 0):
		str += chr(x%256)
		x /= 256
	r = 4-len(str)
	while r > 0:
		str += '\x00'
		r -= 1
	str = str[::-1]
	return str

def convert(in_dir, size, out_imagef, out_labelf):
	out_i = open(out_imagef, 'w')
	out_l = open(out_labelf, 'w')

	#write the headers of image and label files
	str_img = '\x00\x00\x08\x03' + tohex(size) + tohex(40) + tohex(40)
	str_lab = '\x00\x00\x08\x01' + tohex(size)
	for i in range(1,33):
		with open(in_dir+str(i)+'.mat') as f:
			d = sim.loadmat(f)
			data = d['affNISTdata']
			for x in np.nditer(data[0][0][2]):
				str_img += chr(x)
			for x in np.nditer(data[0][0][5]):
				str_lab += chr(x)
			f.close()
	out_i.write(str_img)
	out_l.write(str_lab)
	out_i.close()
	out_l.close()

convert('training_and_validation_batches/', 60000*32, 'train_images_ubyte', 'train_labels_ubyte')
convert('test_batches/', 10000*32, 'test_images_ubyte', 'test_labels_ubyte')
