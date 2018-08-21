'''
Created on 2016/03/05

@author: zhouhui, UESTC
'''
from __future__ import print_function
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import caffe

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def conv_factory(bottom, ks, nout, stride=1, pad=0):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, bias_term=False, weight_filler=dict(type='msra'))
    batch_norm = L.BatchNorm(conv, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    return scale

def conv_factory_relu(bottom, ks, nout, stride=1, pad=0):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, bias_term=False, weight_filler=dict(type='msra'))
    batch_norm = L.BatchNorm(conv, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    relu = L.ReLU(scale, in_place=True)
    return relu

def conv_factory_relu_h_w(bottom, ks_h, ks_w, nout, stride=1, pad_h=0, pad_w=0):
    conv = L.Convolution(bottom, kernel_h=ks_h, kernel_w=ks_w, stride=stride,
                                num_output=nout, pad_h=pad_h, pad_w=pad_w, bias_term=False, weight_filler=dict(type='msra'))
    batch_norm = L.BatchNorm(conv, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    relu = L.ReLU(scale, in_place=True)
    return relu

def Inception_ResNet_C(bottom, bottom_size=2048, num1x1=192, num1x3=224, num3x1=256):
    conv1x1 = conv_factory_relu(bottom, 1, num1x1, 1, 0)
    conv1x3_1x1 = conv_factory_relu(bottom, 1, num1x1, 1, 0)
    conv3x1_1x3= conv_factory_relu_h_w(conv1x3_1x1, 1, 3, num1x3, 1, 0, 1)
    conv3x1 = conv_factory_relu_h_w(conv3x1_1x3, 3, 1, num3x1, 1, 1, 0)
    
    concat = L.Concat(conv1x1, conv3x1)
    proj = conv_factory(concat, 1, bottom_size)
    residual = L.Eltwise(bottom, proj, operation=P.Eltwise.SUM)
    return residual

def ReductionB(bottom, num1x1=256, num3x3=384, num3x3double=288, num3x3three=320):
    pool = max_pool(bottom, 3, 2)
    
    conv1x1_1 = conv_factory_relu(bottom, 1, num1x1, 1, 0)
    conv3x3_1x1_1 = conv_factory_relu(conv1x1_1, 3, num3x3, 2, 0)
    
    conv1x1_2 = conv_factory_relu(bottom, 1, num1x1, 1, 0)
    conv3x3_1x1_2 = conv_factory_relu(conv1x1_2, 3, num3x3double, 2, 0)
    
    conv1x1_3 = conv_factory_relu(bottom, 1, num1x1, 1, 0)
    conv3x3_1x1_3 = conv_factory_relu(conv1x1_3, 3, num3x3double, 1, 1)
    conv3x3_3x3 = conv_factory_relu(conv3x3_1x1_3, 3, num3x3three, 2, 0)
    
    concat = L.Concat(pool, conv3x3_1x1_1, conv3x3_1x1_2, conv3x3_3x3)
    return concat

def Inception_ResNet_B(bottom, bottom_size=1154, num1x1=192, num1x1double=128, num7x1=160, num1x7=192):
    conv1x1 = conv_factory_relu(bottom, 1, num1x1, 1, 0)
    conv1x7_1x1 = conv_factory_relu(bottom, 1, num1x1double, 1, 0)
    conv7x1_1x7= conv_factory_relu_h_w(conv1x7_1x1, 1, 7, num1x7, 1, 0, 3)
    conv7x1 = conv_factory_relu_h_w(conv7x1_1x7, 7, 1, num7x1, 1, 3, 0)
    
    concat = L.Concat(conv1x1, conv7x1)
    proj = conv_factory(concat, 1, bottom_size)
    residual = L.Eltwise(bottom, proj, operation=P.Eltwise.SUM)
    return residual

def ReductionA(bottom, num1x1_k=256, num3x3_l=256, num3x3_n=384, num3x3_m=384):
    pool = max_pool(bottom, 3, 2) #35 to 17
    conv3x3 = conv_factory_relu(bottom, 3, num3x3_n, 2, 0) # 35 to 17
    conv3x3double_1x1 = conv_factory_relu(bottom, 1, num1x1_k, 1, 0) # 35 to 35
    conv3x3double_3x3_1 = conv_factory_relu(conv3x3double_1x1, 3, num3x3_l, 1, 1) #35 to 35
    conv3x3double_3x3_2 = conv_factory_relu(conv3x3double_3x3_1, 3, num3x3_m, 2, 0) #35 to 17
    concat = L.Concat(pool, conv3x3, conv3x3double_3x3_2)
    return concat

def Inception_ResNet_A(bottom, bottom_size=384, num1x1=32, num3x3=48, num3x3double=64):
    conv1x1 = conv_factory_relu(bottom, 1, num1x1, 1, 0)
    conv3x3_1x1 = conv_factory_relu(bottom, 1, num1x1, 1, 0)
    conv3x3 = conv_factory_relu(conv3x3_1x1, 3, num1x1, 1, 1)
    conv3x3double_1x1 = conv_factory_relu(bottom, 1, num1x1, 1, 0)
    conv3x3double_3x3_1 = conv_factory_relu(conv3x3double_1x1, 3, num3x3, 1, 1)
    conv3x3double_3x3_2 = conv_factory_relu(conv3x3double_3x3_1, 3, num3x3double, 1, 1)
    concat = L.Concat(conv1x1, conv3x3, conv3x3double_3x3_2)
    proj = conv_factory(concat, 1, bottom_size)
    residual = L.Eltwise(bottom, proj, operation=P.Eltwise.SUM)
    return residual

def stem(bottom, conv1_num=32, conv2_num=32, conv3_num=64, 
         conv4_num=96, conv5_num=64, conv6_num=96, conv7_num=64, 
         conv8_num=64, conv9_num=64, conv10_num=96, conv11_num=192):
    #stage1
    conv1_3x3 = conv_factory_relu(bottom, 3, conv1_num, 2, 0)   #299*299 to 149*149  
    conv2_3x3 = conv_factory_relu(conv1_3x3, 3, conv2_num, 1, 0) #149*149 to 147*147
    conv3_3x3 = conv_factory_relu(conv2_3x3, 3, conv3_num, 1, 1) #147*147 to 147*147
    pool1 = max_pool(conv3_3x3, 3, 2)                         #147*147 to 73*73
    conv4_3x3 = conv_factory_relu(conv3_3x3, 3, conv4_num, 2, 0) #147*147 to 73*73
    concat1 = L.Concat(pool1, conv4_3x3)
    #stage2
    #for the left side
    conv5_1x1 = conv_factory_relu(concat1, 1, conv5_num, 1, 0)   # to 73
    conv6_3x3 = conv_factory_relu(conv5_1x1, 3, conv6_num, 1, 0) # 73 to 71
    #for the right side
    conv7_1x1 = conv_factory_relu(concat1, 1, conv7_num, 1, 0)  # to 73
    conv8_7x1 = conv_factory_relu_h_w(conv7_1x1, 7, 1, conv8_num, 1, 3, 0) # to 73     
    conv9_1x7 = conv_factory_relu_h_w(conv8_7x1, 1, 7, conv9_num, 1, 0, 3)  # to 73  
    conv10_3x3 = conv_factory_relu(conv9_1x7, 3, conv10_num, 1, 0)  #71
    
    concat2 = L.Concat(conv6_3x3, conv10_3x3)
    #stage3
    pool2 = max_pool(concat2, 3, 2) # to 35
    conv11_3x3 = conv_factory_relu(concat2, 3, conv11_num, 2, 0) # to 35
    concat = L.Concat(pool2, conv11_3x3)
    return concat

def InceptionResNetV2(train_lmdb, test_lmdb, input_size=299, batch_size=256, stages=[0, 5, 10, 5], first_output=32, include_acc=False):
    # now, this code can't recognize include phase, so there will only be a TEST phase data layer
    data, label = L.Data(source=train_lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
        transform_param=dict(crop_size=input_size, mean_value=[104, 117, 123], mirror=True),
        include=dict(phase=getattr(caffe_pb2, 'TRAIN')))
    data, label = L.Data(source=test_lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
        transform_param=dict(crop_size=input_size, mean_value=[104, 117, 123], mirror=True),
        include=dict(phase=getattr(caffe_pb2, 'TEST')))
    data, label = L.MemoryData(batch_size=batch_size, height=input_size, width=input_size, channels=3, ntop=2,
        transform_param=dict(mean_value=[104, 117, 123], mirror=True),
        include=dict(phase=getattr(caffe_pb2, 'TEST')))
    
    Inception_ResNet_A_input = stem(bottom=data, conv1_num=32, conv2_num=32, conv3_num=64, 
         conv4_num=96, conv5_num=64, conv6_num=96, conv7_num=64, 
         conv8_num=64, conv9_num=64, conv10_num=96, conv11_num=192)
    for i in xrange(stages[1]):
      Inception_ResNet_A_input = Inception_ResNet_A(bottom=Inception_ResNet_A_input, 
                                                    bottom_size=384, num1x1=32, num3x3=48, num3x3double=64)
    
    Inception_ResNet_B_input = ReductionA(bottom=Inception_ResNet_A_input, num1x1_k=256, num3x3_l=256, num3x3_n=384, num3x3_m=384)
    
    for i in xrange(stages[2]):
        Inception_ResNet_B_input = Inception_ResNet_B(bottom=Inception_ResNet_B_input, bottom_size=1152, num1x1=192, num1x1double=128, num7x1=160, num1x7=192)
    
    Inception_ResNet_C_input = ReductionB(bottom=Inception_ResNet_B_input, num1x1=256, num3x3=384, num3x3double=288, num3x3three=320)
    
    for i in xrange(stages[3]):
        Inception_ResNet_C_input = Inception_ResNet_C(bottom=Inception_ResNet_C_input, bottom_size=2144, num1x1=192, num1x3=224, num3x1=256)
    
    glb_pool = L.Pooling(Inception_ResNet_C_input, pool=P.Pooling.AVE, global_pooling=True)
    dropout = L.Dropout(glb_pool, dropout_ratio = 0.2) 
    fc = L.InnerProduct(dropout, num_output=1000)
    loss = L.SoftmaxWithLoss(fc, label)
    acc = L.Accuracy(fc, label, include=dict(phase=getattr(caffe_pb2, 'TEST')))
    return to_proto(loss, acc)
def make_net():
    with open('Inception_ResNet_v2.prototxt', 'w') as f:
        print(InceptionResNetV2('/path/to/caffe-train-lmdb', '/path/to/caffe-val-lmdb'), file=f)

if __name__ == '__main__':
    make_net()
    caffe.Net('Inception_ResNet_v2.prototxt', caffe.TEST)  # test loading the net
    
    
