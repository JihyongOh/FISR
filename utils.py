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


def test(self):
    # saver to save model
    self.saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    # restore check-point
    self.load(self.checkpoint_dir)

    # load pre-trained model of image restoration branch
    # saver_pretrain = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Network'))
    # saver_pretrain.restore(self.sess, 'F:\sooye\SR+HDR3\GitHub(JSI_GAN)\checkpoint_dir\JSI-GAN_x2_exp2/JSI-GAN-14751')
    print(" [*] Loading generator...")

    """" Test """
    """ Matlab data for test """
    data_path_test = self.test_data_path_LR_SDR
    label_path_test = self.test_data_path_HR_HDR
    data_test, label_test = read_mat_file(data_path_test, label_path_test, 'SDR_YUV', 'HDR_YUV')
    data_sz = data_test.shape
    label_sz = label_test.shape

    """ Make "test_img_dir" per experiment """
    test_img_dir = os.path.join(self.test_img_dir, self.model_dir)
    if not os.path.exists(test_img_dir):
        os.makedirs(test_img_dir)

    """ Testing """
    patch_boundary = 0  # set patch boundary to reduce edge effect around patch edges

    test_loss_PSNR_list_for_epoch = []
    inf_time = []
    start_time = time.time()
    test_pred_full = np.zeros((label_sz[1], label_sz[2], label_sz[3]))
    data_test_ph = tf.placeholder(tf.float32, shape=(1, None, None, data_sz[3]))
    test_pred = self.model(data_test_ph, self.scale_factor, reuse=True, scope='Network')
    for index in range(data_sz[0]):
        for p in range(self.test_patch[0] * self.test_patch[1]):
            pH = p // self.test_patch[1]
            pW = p % self.test_patch[1]
            sH = data_sz[1] // self.test_patch[0]
            sW = data_sz[2] // self.test_patch[1]

            """ Define model considering patch boundary """
            H_low_ind, H_high_ind, W_low_ind, W_high_ind, add_H, add_W = \
                get_HW_boundary(patch_boundary, data_sz[1], data_sz[2], pH, sH, pW, sW)
            # divide data into patches
            data_test_p = data_test[index, H_low_ind: H_high_ind, W_low_ind: W_high_ind, :]
            data_test_p = np.expand_dims(data_test_p, axis=0)
            # run session
            st = time.time()
            test_pred_o = self.sess.run(test_pred, feed_dict={data_test_ph: data_test_p})
            t = time.time() - st
            inf_time.append(t)
            test_pred_t = trim_patch_boundary(test_pred_o, patch_boundary, data_sz[1], data_sz[2], pH, sH, pW, sW)
            # store in pred_full
            test_pred_full[pH * sH * self.scale_factor: (pH + 1) * sH * self.scale_factor,
            pW * sW * self.scale_factor: (pW + 1) * sW * self.scale_factor, :] = np.squeeze(test_pred_t)
        # compute PSNR
        test_GT = np.squeeze(label_test[index, :, :, :])
        test_PSNR = utils.compute_psnr(test_pred_full, test_GT, 1.)
        print(" <Test> [%4d/%4d]-th images, time: %4.4f(minutes), test_PSNR: %.8f[dB]  "
              % (int(index), int(data_sz[0]), (time.time() - start_time) / 60, test_PSNR))

        # utils.save_results_yuv(test_pred_full, index, test_img_dir)
        test_loss_PSNR_list_for_epoch.append(test_PSNR)
    test_PSNR_per_epoch = np.mean(test_loss_PSNR_list_for_epoch)

    print("######### Test (average) test_PSNR: %.8f[dB]  #########" % (test_PSNR_per_epoch))
    print("######### Estimated Inference Time: %.8f[dB]  #########" % (
                np.mean(inf_time) * self.test_patch[0] * self.test_patch[1]))

def test(self):
    """" Measure the performance in YUV color space. """
    # saver to save model
    self.saver = tf.train.Saver()
    tf.global_variables_initializer().run()  # before "restore"

    # restore the checkpoint
    _, _ = self.load(self.checkpoint_dir)

    test_data_path = glob.glob(os.path.join(self.test_data_path, '*.png'))
    test_label_path = glob.glob(os.path.join(self.test_label_path, '*.png'))

    print(" Start to read flow data (test).")
    flow_path = './data/flo/LR_Surfing_SlamDunk_test_ss1.flo'
    H, W = self.test_input_size
    flow = read_flo_file_5dim(flow_path)
    print(" Successfully load.")
    flow = merge_seq_dim(flow)

    print(" Start to read warped data (test).")
    warp_path = './data/warp/LR_Surfing_SlamDunk_test_ss1_warp.mat'
    warp = read_mat_file_warp(warp_path, 'pred')  # 2K input
    print(" Successfully load.")
    warp = merge_seq_dim(warp)

    num_patch = self.test_patch  # due to memory capacity, we divide the whole image into small patches.
    patch_boundary = 32  # multiple of 32

    test_Loss_recn_list_for_epoch = []
    test_FISR_Loss_PSNR_list_for_epoch = []
    test_SR_Loss_PSNR_list_for_epoch = []
    test_FISR_Loss_SSIM_list_for_epoch = []
    test_SR_Loss_SSIM_list_for_epoch = []

    start_time = time.time()

    """ make "test_img_dir" per experiment """
    test_img_dir = os.path.join(self.test_img_dir, self.model_dir)
    check_folder(test_img_dir)

    n_in_seq = 3
    n_GT_seq = n_in_seq * 2 - 3  # 3
    n_test_in_seq = 5
    n_test_label_seq = 2 * n_test_in_seq - 3  # 7

    for scene_i in range(int(len(test_data_path) / n_test_in_seq)):
        for sample_i in range(n_test_in_seq - n_in_seq + 1):
            test_PSNR = []
            test_SSIM = []
            ###======== Read & Compose Data ========###
            for seq_i in range(n_in_seq):
                # read "n_in_seq" subsequent frames
                img_temp = np.array(Image.open(test_data_path[scene_i * n_test_in_seq + sample_i + seq_i]))
                if seq_i == 0:
                    img = img_temp
                else:
                    img = np.concatenate((img, img_temp), axis=2)
            for seq_i in range(n_GT_seq):
                label_temp = np.array(
                    Image.open(test_label_path[scene_i * n_test_label_seq + sample_i * 2 + seq_i]))
                if seq_i == 0:
                    label = label_temp
                else:
                    label = np.concatenate((label, label_temp), axis=2)

            ###======== Crop Data ========###
            # crop img for u-net (32)
            [h, w, c] = img.shape
            h = h - np.remainder(h, 32 * num_patch[0])
            w = w - np.remainder(w, 32 * num_patch[1])
            img = img[:h, :w, :]
            label = label[:h * self.scale_factor, :w * self.scale_factor, :]

            ###======== Normalize & Clip Image ========###
            img = np.array(img, dtype=np.double) / 255.
            label = np.array(label, dtype=np.double) / 255.
            img = np.expand_dims(np.clip(img, 0, 1), axis=0)
            label = np.expand_dims(np.clip(label, 0, 1), axis=0)

            ###======== Normalize & Clip Flow ========###
            flow_sample = flow[scene_i, :h, :w, 4 * sample_i:4 * sample_i + 8]
            flow_sample = flow_sample / self.data_sz[1] / 2
            flow_sample = np.expand_dims(np.clip(flow_sample, -1, 1), axis=0)

            ###======== Normalize & Clip Warp Image ========###
            warp_sample = warp[scene_i, :h, :w, 6 * sample_i:6 * sample_i + 12]
            warp_sample = np.expand_dims(np.clip(warp_sample, 0, 1), axis=0)

            ###======== Generate Input ========###
            input = np.concatenate([img, flow_sample, warp_sample], axis=3)
            test_Pred_full = np.zeros((h * self.scale_factor, w * self.scale_factor, c))

            ###======== Divide & Process due to Limited Memory ========###
            for p in range(num_patch[0] * num_patch[1]):
                pH = p // num_patch[1]
                pW = p % num_patch[1]
                sH = h // num_patch[0]
                sW = w // num_patch[1]

                # consider patch boundary
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

                ###======== Set Model ========###
                data_test_ph = tf.placeholder(tf.float32, shape=(1, sH + add_H, sW + add_W, c + 8 + 12))
                label_test_ph = tf.placeholder(tf.float32,
                                               shape=(1, sH * self.scale_factor, sW * self.scale_factor, c * 2 - 9))
                [_, _, test_Pred] = self.model(data_test_ph, self.scale_factor, reuse=True, scope='FISRnet')
                self.test_recnLoss = L2_loss(test_Pred, label_test_ph)

                ###======== Pre-process Data ========###
                img_patch = img[:, H_low_ind:H_high_ind, W_low_ind:W_high_ind, :]
                flow_sample_patch = flow_sample[:, H_low_ind:H_high_ind, W_low_ind:W_high_ind, :]
                warp_sample_patch = warp_sample[:, H_low_ind:H_high_ind, W_low_ind:W_high_ind, :]

                simg = np.concatenate([img_patch, flow_sample_patch, warp_sample_patch], axis=3)
                slabel = label[:, pH * sH * self.scale_factor: (pH + 1) * sH * self.scale_factor,
                         pW * sW * self.scale_factor: (pW + 1) * sW * self.scale_factor, :]

                # simg = input[:, pH * sH: (pH + 1) * sH, pW * sW: (pW + 1) * sW, :]
                # slabel = label[:, pH * sH * self.scale_factor: (pH + 1) * sH * self.scale_factor,
                #          pW * sW * self.scale_factor: (pW + 1) * sW * self.scale_factor, :]
                #
                # ###======== Set Model ========###
                # data_test_ph = tf.placeholder(tf.float32, shape=(1, sH, sW, c+8+12))
                # label_test_ph = tf.placeholder(tf.float32,
                #                                shape=(1, sH * self.scale_factor, sW * self.scale_factor, c * 2 - 9))
                # [_, _, test_Pred] = self.model(data_test_ph, self.scale_factor, reuse=True, scope='FISRnet')
                # self.test_recnLoss = L2_loss(test_Pred, label_test_ph)

                ###======== Run Session ========###
                test_recnLoss, test_Pred = self.sess.run(
                    [self.test_recnLoss, test_Pred], feed_dict={data_test_ph: simg, label_test_ph: slabel})
                test_Pred_full[pH * sH * self.scale_factor: (pH + 1) * sH * self.scale_factor,
                pW * sW * self.scale_factor: (pW + 1) * sW * self.scale_factor, :] = np.squeeze(test_Pred)

            ###======== Process Prediction & GT ========###
            test_pred = np.clip(test_Pred_full, 0, 1)
            test_GT = np.squeeze(label)

            ###======== Compute PSNR & Print Results========###
            for seq_i in range(n_GT_seq):
                test_PSNR.append(utils._compute_psnr(test_pred[:, :, 3 * seq_i:3 * (seq_i + 1)],
                                                     test_GT[:, :, 3 * seq_i:3 * (seq_i + 1)], 1.))
                test_SSIM.append(compare_ssim(
                    Image.fromarray((test_pred[:, :, 3 * seq_i:3 * (seq_i + 1)] * 255).astype('uint8')),
                    Image.fromarray((test_GT[:, :, 3 * seq_i:3 * (seq_i + 1)] * 255).astype('uint8'))))
            print(
                " <Test> [%4d/%4d]-th image, scene: %2d-%d, time: %4.4f(minutes), test_PSNR: fr1 (FI-SR) %.8f[dB], fr2 (SR) %.8f[dB], fr3 (FI-SR) %.8f[dB]  " \
                % (scene_i * (n_test_in_seq - n_in_seq + 1) + sample_i,
                   len(test_data_path) / n_test_in_seq * (n_test_in_seq - n_in_seq + 1),
                   scene_i, sample_i, (time.time() - start_time) / 60, test_PSNR[0], test_PSNR[1], test_PSNR[2]))
            print(
                " --------------------------------------------------------------- test_SSIM: fr1 (FI-SR) %.8f, fr2 (SR) %.8f, fr3 (FI-SR) %.8f  " \
                % (test_SSIM[0], test_SSIM[1], test_SSIM[2]))

            ###======== Save Predictions as RGB Images (YUV->RGB) ========###
            # by considering the overlapping, the frame from the later sliding window is taken for simplicity ("if sample_i == 2:")
            pred = np.uint8(test_pred * 255)  # YUV, range of [0,255]
            # check
            for seq_i in range(n_GT_seq):
                fr_name = os.path.basename(test_label_path[scene_i * n_test_label_seq + sample_i * 2 + seq_i])
                fr_name = fr_name[3:]
                rgb_img = utils.YUV2RGB_matlab(pred[:, :, seq_i * 3:(seq_i + 1) * 3])
                pred_img = Image.fromarray(rgb_img.astype('uint8'))
                pred_img.save(os.path.join(test_img_dir, 'pred_{}'.format(fr_name)))

            ###======== Append Loss & PSNR ========###
            test_Loss_recn_list_for_epoch.append(test_recnLoss)
            test_FISR_Loss_PSNR_list_for_epoch.append(test_PSNR[0])
            test_SR_Loss_PSNR_list_for_epoch.append(test_PSNR[1])
            test_FISR_Loss_SSIM_list_for_epoch.append(test_SSIM[0])
            test_SR_Loss_SSIM_list_for_epoch.append(test_SSIM[1])

            if sample_i == 2:
                test_FISR_Loss_PSNR_list_for_epoch.append(test_PSNR[2])
                test_FISR_Loss_SSIM_list_for_epoch.append(test_SSIM[2])

    ###======== Compute Mean PSNR & SSIM for Whole Scenes ========###
    test_recnLoss_per_epoch = np.mean(test_Loss_recn_list_for_epoch)
    test_FISR_PSNR_per_epoch = np.mean(test_FISR_Loss_PSNR_list_for_epoch)
    test_SR_PSNR_per_epoch = np.mean(test_SR_Loss_PSNR_list_for_epoch)
    test_FISR_SSIM_per_epoch = np.mean(test_FISR_Loss_SSIM_list_for_epoch)
    test_SR_SSIM_per_epoch = np.mean(test_SR_Loss_SSIM_list_for_epoch)

    print(
        "######### Test (average) test_PSNR: FISR %.8f[dB], SR %.8f[dB],"
        "recnLoss: %.8f  #########" \
        % (test_FISR_PSNR_per_epoch, test_SR_PSNR_per_epoch, test_recnLoss_per_epoch))
    print(
        "######### Test (average) test_SSIM: FISR %.8f, SR %.8f #########" \
        % (test_FISR_SSIM_per_epoch, test_SR_SSIM_per_epoch))


