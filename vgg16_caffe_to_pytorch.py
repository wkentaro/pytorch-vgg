#!/usr/bin/env python

import os.path as osp

import caffe
import torch
import torchvision


here = osp.dirname(osp.abspath(__file__))

caffe_prototxt = osp.join(here, 'caffe_model_zoo/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers_deploy.prototxt')  # NOQA
caffe_model_path = osp.expanduser('~/data/models/caffe/VGG_ILSVRC_16_layers.caffemodel')  # NOQA
url = 'http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel'  # NOQA
if not osp.exists(caffe_model_path):
    import gdown
    gdown.download(url, caffe_model_path, quiet=False)
caffe_model = caffe.Net(caffe_prototxt, caffe_model_path, caffe.TEST)

torch_model = torchvision.models.vgg16()
torch_model_params = torch_model.parameters()

for name, p1 in caffe_model.params.iteritems():
    p2 = torch_model_params.next()
    print('%s: %s -> %s' % (name, p1[0].data.shape, p2.data.size()))
    p2.data = torch.from_numpy(p1[0].data)
    if len(p1) == 2:
        p2 = torch_model_params.next()
        print('%s: %s -> %s' % (name, p1[1].data.shape, p2.data.size()))
        p2.data = torch.from_numpy(p1[1].data)

torch_model_path = osp.expanduser('~/data/models/torch/vgg16-from-caffe.pth')
torch.save(torch_model.state_dict(), torch_model_path)
