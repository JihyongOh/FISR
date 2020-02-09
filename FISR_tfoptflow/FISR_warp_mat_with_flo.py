"""
FISR_warp_mat_with_flo.py
modified from 'pwcnet_predict_from_img_pairs.py'

Run inference on a list of images pairs.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)
"""

from __future__ import absolute_import, division, print_function
import h5py
import hdf5storage
import numpy as np
import cv2
import matplotlib.pyplot as plt


def read_mat_file(data_fname, data_name):
    # read training data (.mat file) [H, W, C, N_seq, N]
    data_file = h5py.File(data_fname, 'r')
    data = data_file[data_name].value  # [N, N_seq, C, W, H]

    # change type & reorder
    data = np.array(data, dtype=np.float32)
    data = np.swapaxes(data, 2, 4)  # [N, N_seq, H, W, C]

    return data


def YUV2RGB_matlab(yuv):
    Tinv = np.array(
        [[0.00456621, 0., 0.00625893], [0.00456621, -0.00153632, -0.00318811], [0.00456621, 0.00791071, 0.]])
    offset = [[16], [128], [128]]
    T = 255 * Tinv
    offset = 255 * Tinv @ offset
    rgb = np.zeros(yuv.shape)
    for p in range(3):
        rgb[:, :, p] = T[p, 0] * yuv[:, :, 0] + T[p, 1] * yuv[:, :, 1] + T[p, 2] * yuv[:, :, 2] - offset[p]
    rgb = np.clip(rgb, 0, 255)
    return rgb


def RGB2YUV_matlab(rgb):
    T = np.array([[65.481, 128.553, 24.966], [-37.797, -74.203, 112], [112, -93.786, -18.214]])
    offset = [[16], [128], [128]]
    T = T / 255
    offset = offset  # if float: offset = offset / 255
    yuv = np.zeros(rgb.shape)
    for p in range(3):
        yuv[:, :, p] = T[p, 0] * rgb[:, :, 0] + T[p, 1] * rgb[:, :, 1] + T[p, 2] * rgb[:, :, 2] + offset[p]
    yuv = np.clip(yuv, 0, 255)
    return yuv



def warp_flow(img, opt_flow):
    h, w = opt_flow.shape[0:2]  # [H, W, 2]
    # opt_flow = -opt_flow
    opt_flow[:, :, 0] += np.arange(w)
    opt_flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, opt_flow, None, cv2.INTER_LINEAR, None, cv2.BORDER_REPLICATE)
    return res


def compute_psnr(img_orig, img_out, peak):
    mse = np.mean(np.square(img_orig - img_out))
    psnr = 10 * np.log10(peak * peak / mse)
    return psnr


def read_flo_file_5dim(filename):
    """ We modify it for our setting. """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        N = np.fromfile(f, np.int32, count=1)[0]
        N_seq = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        print("Reading %d x %d x %d x %d x 2 flow file in .flo format" % (N, N_seq, h, w))
        data2d = np.fromfile(f, np.float32, count=N * N_seq * h * w * 2)
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (N, N_seq, h, w, 2))
    f.close()
    return data2d


if __name__ == '__main__':
    """ (when preparing 'train') Example to make './data/train/warped/LR_Surfing_SlamDunk_5seq_ss1_warp.mat' from './data/train/flow/LR_Surfing_SlamDunk_5seq_ss1.flo' '"""
    """ To make warped images data with .mat file by using pre-made .flo file. (FISR_pwcnet_predict_from_mat.py)"""
    """ Please check '# check' marks in this code to fit your usage. """
    img_pairs = []
    scale = 2 # check, upscaling factor for spatial resolution
    ss = 1  # check, temporal stride (1 or 2)

    # Read data from mat file
    data_path = 'E:/FISR_Github/data/train/LR_LFR/LR_Surfing_SlamDunk_5seq.mat' # our pre-made 4K dataset
    data = read_mat_file(data_path, 'LR_data')
    sz = data.shape
    flow_path = 'E:/FISR_Github/data/train/flow/LR_Surfing_SlamDunk_5seq_ss{}.flo'.format(ss)

    flow = read_flo_file_5dim(flow_path)

    pred = np.zeros((sz[0], 8 // ss, sz[2], sz[3], 3), dtype=np.float32)
    for num in range(sz[0]):
        for seq in range(sz[1] - (ss * 2 - 1)):
            rgb_1 = YUV2RGB_matlab(data[num, ss * seq, :, :, :])
            rgb_2 = YUV2RGB_matlab(data[num, ss * (seq + 1), :, :, :])
            ### 1 -> 2 ###
            flow_sample = flow[num, 2 * seq, :, :, :]
            warped_img_1 = warp_flow(rgb_2, flow_sample * 0.5) # check, middle frame
            pred[num, 2 * seq, :, :, :] = RGB2YUV_matlab(warped_img_1)
            ### 2 -> 1 ###
            flow_sample = flow[num, 2 * seq + 1, :, :, :]
            warped_img_2 = warp_flow(rgb_1, flow_sample * 0.5)
            pred[num, 2 * seq + 1, :, :, :] = RGB2YUV_matlab(warped_img_2)
        print(num)
    pred_warp = {}
    pred_warp[u'pred'] = pred
    hdf5storage.write(pred_warp, '.', 
                      'E:/FISR_Github/data/train/warped/LR_Surfing_SlamDunk_5seq_ss{}_warp.mat'.format(ss), 
                      matlab_compatible=True) # check

    '''
    plt.figure(2, figsize=(5*4, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(np.uint8(rgb_1))
    plt.subplot(1, 4, 2)
    plt.imshow(np.uint8(warped_img_1))
    plt.subplot(1, 4, 3)
    plt.imshow(np.uint8(rgb_2))
    plt.subplot(1, 4, 4)
    plt.imshow(np.uint8(warped_img_2))
    plt.show()
    '''

