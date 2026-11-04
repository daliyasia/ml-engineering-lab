import torch
from itertools import product as product
import numpy as np
from math import ceil


class PriorBox(object):
    def __init__(self, min_sizes = [[16, 32], [64, 128], [256, 512]], 
                 steps = [8, 16, 32], clip = False, image_size = [360,360]):
        super(PriorBox, self).__init__()
        self.min_sizes = min_sizes
        self.steps = steps
        self.clip = clip
        self.image_size = (image_size[0],image_size[1])
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        #print("self.feature_maps",self.feature_maps)
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            #print("k",k)
            #print("f",f)
            min_sizes = self.min_sizes[k]
            #print('min_sizes',min_sizes)
            for i, j in product(range(f[0]), range(f[1])):
                #print("i,j",i,j)
                for min_size in min_sizes:
                    #print("min_size",min_size)
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    #print("s_kx",s_kx)
                    #print("s_ky",s_ky)
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    #print("dense_cx",dense_cx)
                    #print("dense_cy",dense_cy)
                    for cy, cx in product(dense_cy, dense_cx):
                        #print("anchors",anchors)
                        anchors += [cx, cy, s_kx, s_ky]


        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        #print("output",output)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
