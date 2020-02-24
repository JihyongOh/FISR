"""
"FISR_for_video_pwcnet_predict_from_img_test.py"

modified from "pwcnet_predict_from_img_pairs.py"

Run inference on a list of images pairs.

Originally written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)
"""

from __future__ import absolute_import, division, print_function
from copy import deepcopy
from .model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TEST_OPTIONS
from .visualize import display_img_pairs_w_flows
from PIL import Image
from skimage.transform import resize

import os
import h5py
import numpy as np
import glob

# TODO: Set device to use for inference
# Here, we're using a GPU (use '/device:CPU:0' to run inference on the CPU)
gpu_devices = ['/device:GPU:0']
controller = '/device:GPU:0'

# TODO: Set the path to the trained model (make sure you've downloaded it first from http://bit.ly/tfoptflow)
ckpt_path = './models/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000'

def read_mat_file(data_fname, data_name):
    # read training data (.mat file) [H, W, C, N_seq, N]
    data_file = h5py.File(data_fname, 'r')
    data = data_file[data_name].value  # [N, N_seq, C, W, H]

    # change type & reorder
    data = np.array(data, dtype=np.float32)
    data = np.swapaxes(data, 2, 4)  # [N, N_seq, H, W, C]

    return data


def YUV2RGB(yuv):
    Tinv = np.array([[0.00456621, 0., 0.00625893], [0.00456621, -0.00153632, -0.00318811], [0.00456621,  0.00791071,  0.]])
    offset = [[16], [128], [128]]
    T = 255 * Tinv
    offset = 255 * Tinv @ offset
    rgb = np.zeros(yuv.shape)
    for p in range(3):
        rgb[:, :, p] = T[p, 0]*yuv[:, :, 0]+T[p, 1]*yuv[:, :, 1]+T[p, 2]*yuv[:, :, 2] - offset[p]
    rgb = np.clip(rgb, 0, 255)
    return rgb


def write_flow(flow, filename):
    """ We modify it for our setting. """
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (N, N_seq, h, w) = flow.shape[0:4]
    print("Writing %d x %d x %d x %d x 2 flow file in .flo format" % (N, N_seq, h, w))

    N = np.array([N], dtype=np.int32)
    N_seq = np.array([N_seq], dtype=np.int32)
    h = np.array([h], dtype=np.int32)
    w = np.array([w], dtype=np.int32)

    magic.tofile(f)
    N.tofile(f)
    N_seq.tofile(f)
    h.tofile(f)
    w.tofile(f)
    flow.tofile(f)
    f.close()


def FISR_for_video_Compute_Flow(args):
    """ (when preparing 'FISR_for_video') Example to make 'E:/FISR_Github/FISR_test_folder/scene1/scene1_test_ss1_fr5.flo' from 'E:/FISR_Github/FISR_test_folder/scene1/*.png' (YUV images in folder) '"""
    """ Please check '# check' marks in this code to fit your usage. """
    
    # Configure the model for inference, starting with the default options
    nn_opts = deepcopy(_DEFAULT_PWCNET_TEST_OPTIONS)
    nn_opts['verbose'] = True
    nn_opts['ckpt_path'] = ckpt_path
    nn_opts['batch_size'] = 1
    nn_opts['gpu_devices'] = gpu_devices
    nn_opts['controller'] = controller

    # We're running the PWC-Net-large model in quarter-resolution mode
    # That is, with a 6 level pyramid, and upsampling of level 2 by 4 in each dimension as the final flow prediction
    nn_opts['use_dense_cx'] = True
    nn_opts['use_res_cx'] = True
    nn_opts['pyr_lvls'] = 6
    nn_opts['flow_pred_lvl'] = 2

    # The size of the images in this dataset are not multiples of 64, while the model generates flows padded to multiples
    # of 64. Hence, we need to crop the predicted flows to their original size
    nn_opts['adapt_info'] = (1, 192, 192, 2) # check

    # Instantiate the model in inference mode and display the model configuration
    nn = ModelPWCNet(mode='test', options=nn_opts)

    # Read data from scene folder
    data_list = glob.glob(os.path.join(args.frame_folder_path, '*.png')) # read YUV format images.
    h = args.FISR_input_size[0] # check, assumption: 2K input, then, 1080
    w = args.FISR_input_size[1] # check, assumption: 2K input, then, 1920
    num_fr = args.frame_num  # check, in our test set (=5)

    img_pairs = []
    scale = 2  # check
    ss = 1  # check
    pred = np.zeros((num_fr-1, 2, h, w, 2), dtype=np.float32) # bidirectional
    for fr in range(num_fr-1):
        rgb_1 = Image.open(data_list[ss*fr])
        rgb_1 = np.array(rgb_1, dtype=np.float32)
        rgb_1 = YUV2RGB(rgb_1) # since PWC-Net works on RGB images, we have to convert YUV img into RGB img.

        rgb_2 = Image.open(data_list[ss*(fr+1)])
        rgb_2 = np.array(rgb_2, dtype=np.float32)
        rgb_2 = YUV2RGB(rgb_2) # since PWC-Net works on RGB images, we have to convert YUV img into RGB img.

        rgb_1 = resize(rgb_1, (h * scale, w * scale))
        rgb_2 = resize(rgb_2, (h * scale, w * scale))

        img_pairs.append((np.array(rgb_1, dtype=np.uint8), np.array(rgb_2, dtype=np.uint8)))
        img_pairs.append((np.array(rgb_2, dtype=np.uint8), np.array(rgb_1, dtype=np.uint8)))
        # Generate the predictions
        flow = nn.predict_from_img_pairs(img_pairs, batch_size=1, verbose=False)
        flow = np.array(flow)
        img_pairs = []

        flow_rs = resize(flow, (flow.shape[0], h, w, 2), anti_aliasing=True)/scale
        pred[fr, :, :, :, :] = flow_rs
        print("Processing for computing flows [%5d/%5d]"%(fr+1,num_fr))
    print(pred.shape)
    flow_file_name = args.frame_folder_path +'/'+ args.frame_folder_path.split('/')[-1] + '_test_ss{}_fr{}.flo'.format(ss, num_fr)
    write_flow(pred, flow_file_name) 
    print('[*] Flow file saved!')
    
    return flow_file_name
    # display_img_pairs_w_flows(img_pairs, flow)
