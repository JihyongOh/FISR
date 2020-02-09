from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os
import h5py

def str2bool(x):
    return x.lower() in ('true')


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def _compute_psnr(img_orig, img_out, peak):
    mse = np.mean(np.square(img_orig - img_out))
    psnr = 10 * np.log10(peak*peak / mse)
    return psnr


def read_mat_file(data_fname, label_fname, data_name, label_name):
    # read training data (.mat file) [H, W, C, N_seq, N]
    data_file = h5py.File(data_fname, 'r')
    label_file = h5py.File(label_fname, 'r')
    data = data_file[data_name].value  # [N, N_seq, C, W, H]
    label = label_file[label_name].value

    # change type & reorder
    data = np.array(data, dtype=np.float32) / 255. # value range: [0,1] (norm.)
    label = np.array(label, dtype=np.float32) / 255.
    data = np.swapaxes(data, 2, 4)  # [N, N_seq, H, W, C]
    label = np.swapaxes(label, 2, 4)

    return data, label


def read_mat_file_warp(data_fname, data_name):
    # read training data (.mat file) [H, W, C, N_seq, N]
    data_file = h5py.File(data_fname, 'r')
    data = data_file[data_name].value  # [N, N_seq, C, W, H]

    # change type & reorder
    data = np.array(data, dtype=np.float32) / 255. # value range: [0,1] (norm.)
    data = np.transpose(data, (4, 3, 2, 1, 0)) # [N,N_seq,C,W,H]

    return data


def read_flo_file_5dim(filename):
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



def merge_seq_dim(data):
    # data: [N, N_seq, H, W, C], data_new: [N, H, W, C*N_seq]
    sz = data.shape
    data_new = np.transpose(data, axes=(0, 2, 3, 1, 4))  # [N, H, W, N_seq, C]
    data_new = np.reshape(data_new, (sz[0], sz[2], sz[3], sz[1]*sz[4]))  # [N, H, W, C*N_seq]
    return data_new


def split_seq_dim(data):
    # data: [N, H, W, C*N_seq], data_new: [N, N_seq, H, W, C]
    sz = data.shape
    data_new = np.reshape(data, (sz[0], sz[1], sz[2], sz[3]//3, 3))  # [N, H, W, N_seq, C]
    data_new = np.transpose(data_new, axes=(0, 3, 1, 2, 4))  # [N, N_seq, H, W, C]
    return data_new


def YUV2RGB(yuv):
    m = np.array([[1.0, 1.0, 1.0],
                  [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                  [1.4019975662231445, -0.7141380310058594, 0.00001542569043522235]])

    rgb = np.dot(yuv, m)
    rgb[:, :, 0] -= 179.45477266423404
    rgb[:, :, 1] += 135.45870971679688
    rgb[:, :, 2] -= 226.8183044444304
    return rgb


def YUV2RGB_matlab(yuv):
    Tinv = np.array([[0.00456621, 0., 0.00625893], [0.00456621, -0.00153632, -0.00318811], [0.00456621,  0.00791071,  0.]])
    offset = [[16], [128], [128]]
    T = 255 * Tinv
    offset = 255 * Tinv @ offset
    rgb = np.zeros(yuv.shape)
    for p in range(3):
        rgb[:, :, p] = T[p, 0]*yuv[:, :, 0]+T[p, 1]*yuv[:, :, 1]+T[p, 2]*yuv[:, :, 2] - offset[p]
    rgb = np.clip(rgb, 0, 255)
    return rgb

""" Used in test phase """
def get_HW_boundary(patch_boundary, h, w, pH, sH, pW, sW):
    H_low_ind = max(pH * sH - patch_boundary, 0)
    H_high_ind = min((pH + 1) * sH + patch_boundary, h)
    W_low_ind = max(pW * sW - patch_boundary, 0)
    W_high_ind = min((pW + 1) * sW + patch_boundary, w)

    add_H = 0
    add_W = 0
    if pH * sH >= patch_boundary:
        add_H = add_H + patch_boundary
    if (pH + 1) * sH + patch_boundary <= h:
        add_H = add_H + patch_boundary
    if pW * sW >= patch_boundary:
        add_W = add_W + patch_boundary
    if (pW + 1) * sW + patch_boundary <= w:
        add_W = add_W + patch_boundary

    return H_low_ind, H_high_ind, W_low_ind, W_high_ind, add_H, add_W


def trim_patch_boundary(img, patch_boundary, h, w, pH, sH, pW, sW, sf):
    if patch_boundary == 0:
        img = img
    else:
        if pH * sH < patch_boundary:
            img = img
        else:
            img = img[:, patch_boundary*sf:, :, :]
        if (pH + 1) * sH + patch_boundary > h:
            img = img
        else:
            img = img[:, :-patch_boundary*sf, :, :]
        if pW * sW < patch_boundary:
            img = img
        else:
            img = img[:, :, patch_boundary*sf:, :]
        if (pW + 1) * sW + patch_boundary > w:
            img = img
        else:
            img = img[:, :, :-patch_boundary*sf, :]

    return img



