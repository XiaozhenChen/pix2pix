import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models_zk import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
import time
from memory_profiler import profile
import numpy as np
import cv2
def dataViz(img,label,out):
    print(img.max(),img.min())
    imgNumpy = (np.transpose(img.cpu().numpy()[0],(1,2,0))+1.0)*0.5
    labelNumpy = (np.transpose(label.cpu().numpy()[0], (1, 2, 0))+1.0)*0.5
    outNumpy = (np.transpose(out.detach().cpu().numpy()[0], (1, 2, 0))+1.0)*0.5
    cv2.imshow('img',imgNumpy)
    cv2.imshow('label', labelNumpy)
    cv2.imshow('out', outNumpy)
    cv2.waitKey(0)

def encode_input(label_map, link_map=None, real_image=None, feat_map=None, infer=False):
    input_label = label_map.data.cuda()

    input_label = Variable(input_label, volatile=infer)

    link_map = Variable(link_map.data.cuda())

    # real images for training
    if real_image is not None:
        real_image = Variable(real_image.data.cuda())

    return input_label, link_map, real_image, feat_map
def prePost(label, link=None, image=None):
    image = Variable(image) if image is not None else None
    input_label, link_map, real_image, _ = encode_input(Variable(label), Variable(link), image, infer=True)
    input_concat = torch.cat((input_label, link_map), dim=1)
    return input_concat
@profile
def main():
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    # test
    if not opt.engine and not opt.onnx:
        model = create_model(opt)
        if opt.data_type == 16:
            model.half()
        elif opt.data_type == 8:
            model.type(torch.uint8)
                
        if opt.verbose:
            print(model)
    else:
        from run_engine import run_trt_engine, run_onnx
        
    for i, data in enumerate(dataset):
        print('image', data['image'].shape)
        print('label', data['label'].shape)
        print('link', data['link'].shape)
        input('check')
        if i >= opt.how_many:
            break
        if opt.data_type == 16:
            data['label'] = data['label'].half()
            data['inst']  = data['inst'].half()
        elif opt.data_type == 8:
            data['label'] = data['label'].uint8()
            data['inst']  = data['inst'].uint8()
        if opt.export_onnx and False:
            print ("Exporting to ONNX: ", opt.export_onnx)
            assert opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
            torch.onnx.export(model, [data['label'], data['inst']],
                            opt.export_onnx, verbose=True)
            exit(0)
        minibatch = 1 
        if opt.engine:
            generated = run_trt_engine(opt.engine, minibatch, [data['label'], data['inst']])
        elif opt.onnx:
            generated = run_onnx(opt.onnx, opt.data_type, minibatch, [data['label'], data['inst']])
        else:        
            generated = model.inference(prePost(data['label'], data['link'], data['image']) )
        print('generated',generated.shape)
        dataViz(data['label'], data['link'],generated)
        input('check')
        visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                            ('synthesized_image', util.tensor2im(generated.data[0]))])
        img_path = data['path']
        print('process image... %s' % img_path)
        visualizer.save_images(webpage, visuals, img_path)

    webpage.save()

if __name__ == '__main__':
    main()