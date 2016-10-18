# Introduction
This repository contains a python implementation of the Region Based Easy Path Wavelet Transform [1]. 

# Dependencies
The following python packages are used:

	numpy
	scipy
	pywt
	skimage


# External Files
In order to compute the VSI and HaarPSI indexes the files VSI.m [2] and HaarPSI.m [3] respectively are needed, as long as a working octave installation and the python package `oct2py`.


# Example Usage

Create an Image instance and load an image file:

	import rbepwt
    rbimg = rbepwt.Image()
    rbimg.read('/path/to/cameraman.png')
	
Segment it using the Felzenszwalb-Huttenlocher method and view the segmentation:

	rbimg.segment(method='felzenszwalb',scale=200,sigma=2,min_size=10)
	rbimg.show_segmentation()

Encode it using the RBEPWT method with the `easypath` path-finding procedure, 16 levels and the CDF 9/7 wavelet (`bior4.4` in `pywt`):

	rbimg.encode_rbepwt(16,'bior4.4',path_type='easypath',euclidean_distance=True)
	rbimg.threshold_coefs(512)
	rbimg.decode_rbepwt()
	
View the result:

	rbimg.show_decoded()
    


[1] to appear in....
[2] http://sse.tongji.edu.cn/linzhang/IQA/VSI/Files/VSI.m
[3] http://www.math.uni-bremen.de/~reisenho/software/HaarPSI.m
