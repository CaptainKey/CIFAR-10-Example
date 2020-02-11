import random 
import numpy as np
import logging as log

log.basicConfig(filename='debug.log',format='%(levelname)s : %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',level=log.INFO,filemode='w')

"""
    convolution
"""
class convolution:
    
    def __init__(self,name,img_size,kernel,stride,bias=False):
        self.name = name
        self.img_size = img_size
        self.kernel = kernel 
        self.stride = stride 
        self.output_size = int(((img_size[2] - (kernel[3] - 1) - 1) / stride))+ 1
        self.weight = np.array( [ [ [ [random.random() for i in range(kernel[3])] for i in range(kernel[2]) ] for j in range(kernel[1])] for k in range(kernel[0])] )

        if bias:
            self.bias = np.array([random.random() for i in range(kernel[0])])
        else:
            self.bias = bias

        log.info('init - {}'.format(self.name))
    
    def __call__(self,img):
        log.debug('call - {}'.format(self.name))
        img = np.array(img)
        output = np.zeros((self.kernel[0],self.output_size,self.output_size))
        for f in range(self.kernel[0]):
            for c in range(self.img_size[0]):
                for i in range(0,self.output_size):
                    for j in range(0,self.output_size):
                        for m in range(self.kernel[3]):
                            for n in range(self.kernel[3]):
                                output[f][i][j] += img[c][i+m][j+n]*self.weight[f][c][m][n]
                        if self.bias.all():     output[f][i][j] += self.bias[f]
        return output

    def load_weight(self,weight):
        log.debug('load_weight - {}'.format(self.name))
        assert self.weight.shape == weight.shape, log.error('load_weight {}'.format(self.name))
        self.weight = weight

    def load_bias(self,bias):
        log.debug('load_bias - {}'.format(self.name))
        assert self.bias.shape == bias.shape, log.error('load_bias {}'.format(self.name))
        self.bias = bias

    def get_name(self):
        log.debug('get_name - {}'.format(self.name))
        return self.name

"""
    linear
"""
class linear:
    def __init__(self,name,dim_in,dim_out,bias=False):
        self.name    = name
        self.dim_in  = dim_in
        self.dim_out = dim_out 
        self.weight  = np.random.rand(dim_in,dim_out)
        self.bias = np.random.random(dim_out) if bias else False 

        log.info('init - {}'.format(self.name))

    def __call__(self,vecteur):
        log.debug('call - {}'.format(self.name))
        output = np.zeros(self.dim_out)
        for i in range(self.dim_in):
            for j in range(self.dim_out):
                output[j] += vecteur[i]*self.weight[i][j]
            if self.bias.all():    output[j] += self.bias[j]

        return output

    def load_weight(self,weight):
        log.debug('load_weight - {}'.format(self.name))
        assert self.weight.shape == weight.shape, log.error('load_weight {}'.format(self.name))
        self.weight = weight

    def load_bias(self,bias):
        log.debug('load_bias - {}'.format(self.name))
        assert self.bias.shape == bias.shape, log.error('load_bias {}'.format(self.name))
        self.bias = bias

    def get_name(self):
        log.debug('get_name - {}'.format(self.name))
        return self.name
"""
    Maxpooling
"""
class maxpooling:
    def __init__(self,name,kernel_size):
        self.name = name
        self.kernel_size = kernel_size
        log.info('init - {}'.format(self.name))

    def __call__(self,img):
        log.debug('call - {}'.format(self.name))
        shape = img.shape
        output_size = int(shape[1]/self.kernel_size)
        output = np.array([])
        for channels in range(shape[0]):
            for height in range(0,shape[1],self.kernel_size):
                for width in range(0,shape[2],self.kernel_size):
                    maximum = img[channels][height][width]
                    for m in range(0,self.kernel_size):
                        for n in range(0,self.kernel_size):
                            if img[channels][height+m][width+n] > maximum : maximum = img[channels][height+m][width+n]

                    output = np.append(output,maximum)

        output = output.reshape((shape[0],output_size,output_size))
        return output

    def get_name(self):
        log.debug('get_name - {}'.format(self.name))
        return self.name
"""
    relu
"""
class relu:
    def __init__(self,name):
        self.name = name
        log.info('init - {}'.format(self.name))

    def __call__(self,x):
        log.debug('call - {}'.format(self.name))
        x[x < 0] = 0

        return x

    def get_name(self):
        log.debug('get_name - {}'.format(self.name))
        return self.name