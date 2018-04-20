import os
import numpy as np

from struct import unpack


class mnist(object):
    """ Provides images and labels from the MNIST dataset in a variety of formats """

    datasets = { "test"    :   { "images" : "t10k-images-idx3-ubyte", "labels" : "t10k-labels.idx1-ubyte" },
                 "train"     :   { "images" : "train-images-idx3-ubyte", "labels" : "train-labels-idx1-ubyte" }};

    def __init__(self, dataset):
        """ Creates an instance using the specified dataset. Current values are 'train' and 'test' """
        # Check that the parameters passed are correct
        if not dataset in self.datasets:
            raise Exception("Specified dataset does not exist")
        # Configure the base data
        base_directory = "data"
        image_path = "%s/%s" % (base_directory, self.datasets[dataset]['images'])
        label_path = "%s/%s" % (base_directory, self.datasets[dataset]['labels'])
        # Ensure that the files exist
        if not (os.path.exists(image_path) and (os.path.exists(label_path))):
            raise Exception("Specified input files do not exist")
        # Open the files
        self.image_file = open(image_path,"rb")
        self.label_file = open(label_path, "rb")
        # Read and check the magic numbers
        if not (unpack(">i", self.label_file.read(4)) == 2051) and (unpack(">i", self.image_file.read(4)) == 2049):
            raise Exception("Magic numbers in input files are not correct")
        # Read in the number of items
        num_images = unpack(">i", self.image_file.read(4))[0]
        num_labels = unpack(">i", self.label_file.read(4))[0]
        if (num_labels != num_images):
            raise Exception("Number of images and number of labels do not agree")
        self._num_items = num_images
     
    @property
    def NumberOfItems(self):
        """ Returns the number of items in the current data set """
        return self._num_items
    
    def GetImage(self, number):
        """ Returns a tuple containing a numpy array for the specified image, and an integer representing it's label """
        if (number > self._num_items) or (number < 0):
            raise Exception("Specified image in not in the dataset")
        # Attempt to seek to the appropriate spot in the files
        self.label_file.seek(8 + number)
        self.image_file.seek(8 + (number * (28*28)))
        # Read the data out of the file
        label = unpack("B", self.label_file.read(1))[0]     
        # Read in the rows and columnns
        rows = unpack(">i", self.image_file.read(4))[0]
        cols = unpack(">i", self.image_file.read(4))[0]
        # Read in the actual pixel values (unsigned bytes)
        pixels = np.zeros([28*28])
        for i in range(0, 28*28):
            pixels[i] =  unpack("B", self.image_file.read(1))[0]
        return label, pixels

    def GetRandomImage(self):
        """ Returns a random image from the dataset, complete with label """
        number = np.random.random_integers(self.NumberOfItems)
        return self.GetImage(number)
    
    def __del__(self):
        # Close the files
        self.image_file.close()
        self.label_file.close()





