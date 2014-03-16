#!/usr/local/bin/python2.7
# encoding: utf-8

import dicom
import matplotlib.pylab as plot
import matplotlib.cm as cm

# to find current working directory:
# os.getcwd()

ds = dicom.read_file("../data/000000.dcm")
#ds #print all data

plot.imshow(ds.pixel_array, cmap=plot.gray())
plot.show()