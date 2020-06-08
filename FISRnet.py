from __future__ import division
from __future__ import print_function

from PIL import Image
from SSIM_PIL import compare_ssim
from datetime import datetime
from ops import *
import matplotlib.pyplot as plt
import glob
import time
import math
import utils  

class FISRnet(object):
    model_name = "FISRnet"

    def __init__(self, sess, args):
        self.sess = sess
        self.args = args
        self.checkpoint_dir = args.checkpoint_dir
        self.test_img_dir = args.test_img_dir
        self.text_dir = args.text_dir
        self.log_dir = args.log_dir
        self.train_data_path = args.train_data_path
        self.train_flow_data_path = args.train_flow_data_path
        self.train_flow_ss2_data_path = args.train_flow_ss2_data_path
        self.train_warped_data_path = args.train_warped_data_path
        self.train_wapred_ss2_data_path = args.train_wapred_ss2_data_path
        self.train_label_path = args.train_label_path
        self.test_data_path = args.test_data_path
        self.test_flow_data_path = args.test_flow_data_path # check
        self.test_warped_data_path = args.test_warped_data_path  # check
        self.test_label_path = args.test_label_path
        self.exp_num = args.exp_num
        self.scale_factor = args.scale_factor

        """ Hyperparameters """
        self.epoch = args.epoch
        self.init_lr = args.init_lr
        self.freq_display = args.freq_display
        self.lr_type = args.lr_type
        self.lr_stair_decay_points = args.lr_stair_decay_points
        self.lr_decreasing_factor = args.lr_decreasing_factor
        self.lr_linear_decay_point = args.lr_linear_decay_point
        self.batch_size = args.batch_size
        self.val_batch_size = args.val_batch_size
        self.val_data_size = args.val_data_size
        self.n_train_img_showed = args.n_train_img_showed

        """ Coefficients of loss or penalty """
        self.recn_lambda = args.recn_lambda
        self.tm1_lambda = args.tm1_lambda
        self.tm2_lambda = args.tm2_lambda
        self.tmm_lambda = args.tmm_lambda
        self.td_lambda = args.td_lambda
        self.ss2_lambda = args.ss2_lambda

        """ Testing settings """
        self.test_patch = args.test_patch
        self.test_input_size = args.test_input_size
        self.FISR_test_patch = args.FISR_test_patch

        """ Making settings """
        self.frame_folder_path = args.frame_folder_path
        self.FISR_input_size = args.FISR_input_size
        self.frame_num = args.frame_num
        self.FISR_test_patch = args.FISR_test_patch
        """ Print all 'args' information """
        print('Model arguments, [{:s}]'.format((str(datetime.now())[:-7])))
        for arg in vars(args):
            print('# {} : {}'.format(arg, getattr(args, arg)))

    def model(self, img, sf, reuse=False, scope="model"):
        ch = 64
        sz = img.shape
        skip = dict()
        with tf.variable_scope(scope, reuse=reuse):
            """ Multi-scale network """
            with tf.variable_scope('level_1'):
                """ x(1/4) """
                img_l1 =  tf.image.resize_images(img, (sz[1]//4, sz[2]//4),method=tf.image.ResizeMethod.BICUBIC)
                # Encoder
                with tf.variable_scope('enc'):
                    n, skip[0] = Enc_level_res(img_l1, sz[-1], ch, 2, 'level_0')
                    n, skip[1] = Enc_level_res(n, ch, ch * 2, 2, 'level_1')
                    n, skip[2] = Enc_level_res(n, ch * 2, ch * 4, 2, 'level_2')
                # Bottleneck
                n = Bottleneck_res(n, ch * 4, ch * 8, 'bottleneck')
                # Decoder
                with tf.variable_scope('dec'):
                    n = Dec_level_res(n, skip[2], ch * 8, ch * 4, 'level_2', (sz[1] // 16, sz[2] // 16))
                    n = Dec_level_res(n, skip[1], ch * 4, ch * 2, 'level_1', (sz[1] // 8, sz[2] // 8))
                    n = Dec_level_res(n, skip[0], ch * 2, ch, 'level_0', (sz[1] // 4, sz[2] // 4))
                # Final branches for FI-SR & SR only
                with tf.variable_scope('FI-SR'):
                    n2 = Conv2d(n, [3, 3, ch, ch], 'conv/0')
                    n2 = res_block(n2, ch, 'res_block/0')
                    n2 = Conv2d(relu(n2), [3, 3, ch, ch * sf * sf], 'conv/1')
                    n2 = tf.depth_to_space(relu(n2), sf, name='pixel_shuffle')
                    pred_FISR = Conv2d(relu(n2), [3, 3, ch, 6], 'conv/2')
                with tf.variable_scope('SR'):
                    n3 = Conv2d(n, [3, 3, ch, ch], 'conv/0')
                    n3 = res_block(n3, ch, 'res_block/0')
                    n3 = Conv2d(relu(n3), [3, 3, ch, ch * sf * sf], 'conv/1')
                    n3 = tf.depth_to_space(relu(n3), sf, name='pixel_shuffle')
                    pred_SR = Conv2d(n3, [3, 3, ch, 3], 'conv/2')
                fr1, fr2 = tf.split(pred_FISR, [3, 3], 3)
                pred_l1 = tf.concat([fr1, pred_SR, fr2], axis=3)

            with tf.variable_scope('level_2'):
                """ x(1/2) """
                img_l2 = tf.image.resize_images(img, (sz[1]//2, sz[2]//2),method=tf.image.ResizeMethod.BICUBIC)
                img_l2 = tf.concat((img_l2, pred_l1), axis=3)
                # Encoder
                with tf.variable_scope('enc'):
                    n, skip[0] = Enc_level_res(img_l2, sz[-1]+9, ch, 2, 'level_0')
                    n, skip[1] = Enc_level_res(n, ch, ch * 2, 2, 'level_1')
                    n, skip[2] = Enc_level_res(n, ch * 2, ch * 4, 2, 'level_2')
                # Bottleneck
                n = Bottleneck_res(n, ch * 4, ch * 8, 'bottleneck')
                # Decoder
                with tf.variable_scope('dec'):
                    n = Dec_level_res(n, skip[2], ch * 8, ch * 4, 'level_2', (sz[1] // 8, sz[2] // 8))
                    n = Dec_level_res(n, skip[1], ch * 4, ch * 2, 'level_1', (sz[1] // 4, sz[2] // 4))
                    n = Dec_level_res(n, skip[0], ch * 2, ch, 'level_0', (sz[1] // 2, sz[2] // 2))
                # Final branches for FI-SR & SR only
                with tf.variable_scope('FI-SR'):
                    n2 = Conv2d(n, [3, 3, ch, ch], 'conv/0')
                    n2 = res_block(n2, ch, 'res_block/0')
                    n2 = Conv2d(relu(n2), [3, 3, ch, ch * sf * sf], 'conv/1')
                    n2 = tf.depth_to_space(relu(n2), sf, name='pixel_shuffle')
                    pred_FISR = Conv2d(relu(n2), [3, 3, ch, 6], 'conv/2')
                with tf.variable_scope('SR'):
                    n3 = Conv2d(n, [3, 3, ch, ch], 'conv/0')
                    n3 = res_block(n3, ch, 'res_block/0')
                    n3 = Conv2d(relu(n3), [3, 3, ch, ch * sf * sf], 'conv/1')
                    n3 = tf.depth_to_space(relu(n3), sf, name='pixel_shuffle')
                    pred_SR = Conv2d(n3, [3, 3, ch, 3], 'conv/2')
                fr1, fr2 = tf.split(pred_FISR, [3, 3], 3)
                pred_l2 = tf.concat([fr1, pred_SR, fr2], axis=3)

            with tf.variable_scope('level_3'):
                """ original level """
                img_l3 = tf.concat((img, pred_l2), axis=3)
                # Encoder
                with tf.variable_scope('enc'):
                    n, skip[0] = Enc_level_res(img_l3, sz[-1]+9, ch, 2, 'level_0')
                    n, skip[1] = Enc_level_res(n, ch, ch * 2, 2, 'level_1')
                    n, skip[2] = Enc_level_res(n, ch * 2, ch * 4, 2, 'level_2')
                # Bottleneck
                n = Bottleneck_res(n, ch * 4, ch * 8, 'bottleneck')
                # Decoder
                with tf.variable_scope('dec'):
                    n = Dec_level_res(n, skip[2], ch * 8, ch * 4, 'level_2', (sz[1] // 4, sz[2] // 4))
                    n = Dec_level_res(n, skip[1], ch * 4, ch * 2, 'level_1', (sz[1] // 2, sz[2] // 2))
                    n = Dec_level_res(n, skip[0], ch * 2, ch, 'level_0', (sz[1], sz[2]))
                # Final branches for FI-SR & SR only
                with tf.variable_scope('FI-SR'):
                    n2 = Conv2d(n, [3, 3, ch, ch], 'conv/0')
                    n2 = res_block(n2, ch, 'res_block/0')
                    n2 = Conv2d(relu(n2), [3, 3, ch, ch * sf * sf], 'conv/1')
                    n2 = tf.depth_to_space(relu(n2), sf, name='pixel_shuffle')
                    pred_FISR = Conv2d(relu(n2), [3, 3, ch, 6], 'conv/2')
                with tf.variable_scope('SR'):
                    n3 = Conv2d(n, [3, 3, ch, ch], 'conv/0')
                    n3 = res_block(n3, ch, 'res_block/0')
                    n3 = Conv2d(relu(n3), [3, 3, ch, ch * sf * sf], 'conv/1')
                    n3 = tf.depth_to_space(relu(n3), sf, name='pixel_shuffle')
                    pred_SR = Conv2d(n3, [3, 3, ch, 3], 'conv/2')
                fr1, fr2 = tf.split(pred_FISR, [3, 3], 3)
                pred_l3 = tf.concat([fr1, pred_SR, fr2], axis=3)

            return pred_l1, pred_l2, pred_l3

    def build_model(self):
        """ Read training data """
        data_path = self.train_data_path
        label_path = self.train_label_path
        flow_path = self.train_flow_data_path
        flow_path_ss2 = self.train_flow_ss2_data_path
        warp_path = self.train_warped_data_path
        warp_path_ss2 = self.train_wapred_ss2_data_path

        print(" Start to read 4K data.")
        data, label = read_mat_file(data_path, label_path, 'LR_data', 'HR_data')  # [B,N_seg,H,W,C]
        print(" Successfully load.")
        # 5dim, LR: [B, 5, 96, 96, 3], HR: [B, 7, 192, 192, 3]
        data = merge_seq_dim(data)  # 4dim, LR: [B, 96, 96, 3*5]
        label = merge_seq_dim(label)  # 4dim, LR: [B, 96, 96, 3*7]

        self.data_sz = data.shape  # LR: [B, 96, 96, 3*5]
        self.label_sz = label.shape  # LR: [B, 96*2, 96*2, 3*7]

        print(" Start to read flow data.")
        flow = read_flo_file_5dim(flow_path)  # [N, 8, 96, 96, 2] => 8: numbers of bidirectional(x2) flows between 5 frames(stride1), 2: x,y directions
        flow = merge_seq_dim(flow)  # [N, 8, 96, 96, 2] => [N, 96, 96, 8*2]
        flow = flow/self.data_sz[1]/2

        flow_ss2 = read_flo_file_5dim(flow_path_ss2)  # [N, 4, 96, 96, 2] => 4: numbers of bidirectional(x2) flows between 3 frames(stride2), 2: x,y directions
        print(" Successfully load.")
        flow_ss2 = merge_seq_dim(flow_ss2)  # [N, 4, 96, 96, 2] => [N, 96, 96, 4*2]
        flow_ss2 = flow_ss2/self.data_sz[1]/2

        print(" Start to read warped data.")
        warp = read_mat_file_warp(warp_path, 'pred')
        warp_ss2 = read_mat_file_warp(warp_path_ss2, 'pred')
        print(" Successfully load.")
        warp = merge_seq_dim(warp)
        warp_ss2 = merge_seq_dim(warp_ss2)


        """ Split data into val/train """
        self.data_val = data[-self.val_data_size:, :, :, :]
        self.label_val = label[-self.val_data_size:, :, :, :]
        self.flow_val = flow[-self.val_data_size:, :, :, :]
        self.flow_ss2_val = flow_ss2[-self.val_data_size:, :, :, :]
        self.warp_val = warp[-self.val_data_size:, :, :, :]
        self.warp_ss2_val = warp_ss2[-self.val_data_size:, :, :, :]

        self.data = data[:-self.val_data_size, :, :, :]
        self.label = label[:-self.val_data_size, :, :, :]
        self.flow = flow[:-self.val_data_size, :, :, :]
        self.flow_ss2 = flow_ss2[:-self.val_data_size, :, :, :]
        self.warp = warp[:-self.val_data_size, :, :, :]
        self.warp_ss2 = warp_ss2[:-self.val_data_size, :, :, :]

        # calculate number of iterations
        self.train_iter = math.floor((self.data_sz[0] - self.val_data_size) / self.batch_size)
        self.val_iter = math.floor(self.val_data_size / self.val_batch_size)

        """ Learning rate schedule """
        self.global_step = tf.Variable(initial_value=0, trainable=False)
        if self.lr_type == "stair_decay":
            self.epoch_lr_to_be_decayed_boundaries = [y * (self.train_iter) for y in
                                                      self.lr_stair_decay_points]
            self.epoch_lr_to_be_decayed_value = [self.init_lr * (self.lr_decreasing_factor ** y) for y in
                                                 range(len(self.lr_stair_decay_points) + 1)]
            self.lr = tf.train.piecewise_constant(self.global_step, self.epoch_lr_to_be_decayed_boundaries,
                                                  self.epoch_lr_to_be_decayed_value)
            print("lr_type: stair_decay")

        elif self.lr_type == "linear_decay":
            self.lr = tf.placeholder(tf.float32, name='learning_rate')
            print("lr_type: linear_decay")

        else:
            self.lr = tf.placeholder(tf.float32, name='learning_rate')  # no decay
            print("lr_type: no decay")

        """ Multiple data samples-related parameters  """
        self.n_total_in_seq = 5  # N_seq: numbers of multiple data samples
        self.n_in_seq = 3 # numbers of input consecutive frames (input to network)
        self.n_out_groups_sride1 = self.n_total_in_seq - 3 + 1  # 3, numbers of output groups for multiple data samples with temporal stride 1
        self.n_ovlp_stride1 = self.n_out_groups_sride1 - 1  # 2, numbers of overlapping (1,1,....,1)
        self.n_out_groups_sride2 = self.n_total_in_seq - 5 + 1  # 1, numbers of output groups for multiple data samples with temporal stride 2
        self.n_final_ovlp_seq = (self.n_in_seq - 1) * self.n_out_groups_sride1 + 1  # 7, numbers of ground truth (GT) frames (bottom green boxes in Fig. 3)

        """ Defining a model """
        # define variables for data (input)
        self.data_ph = tf.placeholder(tf.float32, shape=(self.batch_size, self.data_sz[1], self.data_sz[2], self.data_sz[3]))
        # define variables for label (GT)
        self.label_ph = tf.placeholder(tf.float32, shape=(self.batch_size, self.label_sz[1],self.label_sz[2], self.label_sz[3])) # level 3 (org.)
        self.label_l2 = tf.image.resize_images(self.label_ph, (self.label_sz[1]//2, self.label_sz[2]//2),method=tf.image.ResizeMethod.BICUBIC) # level 2 (x1/2)
        self.label_l1 = tf.image.resize_images(self.label_ph, (self.label_sz[1]//4, self.label_sz[2]//4),method=tf.image.ResizeMethod.BICUBIC) # level 1 (x1/4)
        # define variables for flow & warped data
        self.flow_ph = tf.placeholder(tf.float32,
                                          shape=(self.batch_size, self.data_sz[1], self.data_sz[2], 16))
        self.flow_ss2_ph = tf.placeholder(tf.float32,
                                      shape=(self.batch_size, self.data_sz[1], self.data_sz[2], 8))
        self.warp_ph = tf.placeholder(tf.float32,
                                          shape=(self.batch_size, self.data_sz[1], self.data_sz[2], 24))
        self.warp_ss2_ph = tf.placeholder(tf.float32,
                                      shape=(self.batch_size, self.data_sz[1], self.data_sz[2], 12))

        self.data_ph_5dim = tf_split_seq_dim(self.data_ph) # 5dim, convert to [bs, 5, h/2, w/2, ch] format
        self.GT_groups = tf_split_seq_dim(self.label_ph) # 5dim, convert to [bs, 7, h, w, ch] format
        self.GT_groups_l2 = tf_split_seq_dim(self.label_l2) # 5dim, convert to [bs, 7, h/2, w/2, ch] format
        self.GT_groups_l1 = tf_split_seq_dim(self.label_l1) # 5dim, convert to [bs, 7, h/4, w/4, ch] format

        """ Multiple data sample training strategy (a novel input/output framework) for stride1 (by recurrent modeling)"""
        for process_i in range(self.n_out_groups_sride1):
            """ Tensor_slicer_~: slicing a tensor w.r.t an order(process_i) from the predefined variables along a channel axis (index=3). """
            if process_i == 0:
                data_only = Tensor_slicer_recurrent(self.data_ph, process_i) #  [B, 96, 96, 3*5] => slicing & get a corresponding tensor => [bs, 5, 96, 96, 3*3]: first input (3 subsequent frames)
                flow_only = Tensor_slicer_recurrent_flow(self.flow_ph, process_i) # [B, 96, 96, 16] => [N, 8, 96, 96, 8(=4x2)]
                warp_only = Tensor_slicer_recurrent_warp(self.warp_ph, process_i) # [B, 96, 96, 24] => [N, 8, 96, 96, 12(=4x3)]
                input = tf.concat((data_only, flow_only, warp_only), axis=3)
                [self.Pred_groups_l1, self.Pred_groups_l2, self.Pred_groups] = self.model(input
                                              , self.scale_factor, reuse=False, scope='FISRnet') # [bs, h, w, ch*3]
                self.Pred_groups = tf_split_seq_dim(self.Pred_groups)  # [bs, 3, h, w, ch], not overlapped yet.
                self.Pred_groups_l2 = tf_split_seq_dim(self.Pred_groups_l2)  # [bs, 3, h, w, ch], not overlapped yet.
                self.Pred_groups_l1 = tf_split_seq_dim(self.Pred_groups_l1)  # [bs, 3, h, w, ch], not overlapped yet.

            else:
                data_only = Tensor_slicer_recurrent(self.data_ph, process_i)
                flow_only = Tensor_slicer_recurrent_flow(self.flow_ph, process_i)
                warp_only = Tensor_slicer_recurrent_warp(self.warp_ph, process_i)
                input = tf.concat((data_only, flow_only, warp_only), axis=3)
                [Pred_temp_l1, Pred_temp_l2, Pred_temp] = self.model(input
                                       , self.scale_factor, reuse=True, scope='FISRnet')
                Pred_temp = tf_split_seq_dim(Pred_temp)  # [bs, 3, h, w, ch], not overlapped yet.
                Pred_temp_l2 = tf_split_seq_dim(Pred_temp_l2)
                Pred_temp_l1 = tf_split_seq_dim(Pred_temp_l1)
                self.Pred_groups = tf.concat([self.Pred_groups, Pred_temp],axis=1) # [bs, 3*3, h, w, ch], not overlapped yet.
                self.Pred_groups_l2 = tf.concat([self.Pred_groups_l2, Pred_temp_l2],axis=1) # [bs, 3*3, h/2, w/2, ch], not overlapped yet.
                self.Pred_groups_l1 = tf.concat([self.Pred_groups_l1, Pred_temp_l1],axis=1) # [bs, 3*3, h/4, w/4, ch], not overlapped yet.

        self.Final_Ovlp_Seq = Groups2Ovlp(self.Pred_groups) # [bs, 3*3, h, w, ch] => [bs, n_final_ovlp_seq(=7), h, w, ch]
        self.Final_Ovlp_Seq_l2 = Groups2Ovlp(self.Pred_groups_l2) # [bs, n_final_ovlp_seq, h/2, w/2, ch]
        self.Final_Ovlp_Seq_l1 = Groups2Ovlp(self.Pred_groups_l1) # [bs, n_final_ovlp_seq, h/4, w/4, ch]

        """ Temporal Loss  """
        """ type1~4 for stride1 """
        # type1. Reconstruction Loss for stride1 (Eq.(6))
        self.recnLoss = tf.constant(0, dtype=tf.float32)
        for process_i in range(self.n_out_groups_sride1):
            temp_Pred_3imgs = Tensor_slicer(self.Pred_groups, process_i * 3, self.n_in_seq)
            temp_Pred_3imgs_l2 = Tensor_slicer(self.Pred_groups_l2, process_i * 3, self.n_in_seq)
            temp_Pred_3imgs_l1 = Tensor_slicer(self.Pred_groups_l1, process_i * 3, self.n_in_seq)
            temp_GT_3imgs = Tensor_slicer(self.GT_groups, process_i * 2, self.n_in_seq)
            temp_GT_3imgs_l2 = Tensor_slicer(self.GT_groups_l2, process_i * 2, self.n_in_seq)
            temp_GT_3imgs_l1 = Tensor_slicer(self.GT_groups_l1, process_i * 2, self.n_in_seq)
            L2_loss_l3 = L2_loss(temp_Pred_3imgs, temp_GT_3imgs)
            L2_loss_l2 = L2_loss(temp_Pred_3imgs_l2, temp_GT_3imgs_l2)
            L2_loss_l1 = L2_loss(temp_Pred_3imgs_l1, temp_GT_3imgs_l1)
            self.recnLoss += (L2_loss_l3+
                              L2_loss_l2*2+
                              L2_loss_l1*4) # multi-scale loss

        # type2. Temporal Matching Loss for stride1 (Eq.(1)) (between overlapped frames)
        self.tmLoss = tf.constant(0, dtype=tf.float32)
        for process_i in range(self.n_ovlp_stride1):
            temp_Front_1img = Tensor_slicer(self.Pred_groups, process_i * 3 + 2, 1)
            temp_Back_1img = Tensor_slicer(self.Pred_groups, process_i * 3 + 3, 1)
            temp_Front_1img_l2 = Tensor_slicer(self.Pred_groups_l2, process_i * 3 + 2, 1)
            temp_Back_1img_l2 = Tensor_slicer(self.Pred_groups_l2, process_i * 3 + 3, 1)
            temp_Front_1img_l1 = Tensor_slicer(self.Pred_groups_l1, process_i * 3 + 2, 1)
            temp_Back_1img_l1 = Tensor_slicer(self.Pred_groups_l1, process_i * 3 + 3, 1)
            self.tmLoss += (L2_loss(temp_Front_1img, temp_Back_1img) + \
                           L2_loss(temp_Front_1img_l2, temp_Back_1img_l2)*2 + \
                           L2_loss(temp_Front_1img_l1, temp_Back_1img_l1)*4)

        # type3. Temporal Matching Mean Loss for stride1 (Eq.(3))
        self.tmmLoss = tf.constant(0, dtype=tf.float32)
        for process_i in range(self.n_ovlp_stride1):
            temp_Ovlp1_1img = Tensor_slicer(self.Pred_groups, process_i * 3 + 2, 1)
            temp_Ovlp2_1img = Tensor_slicer(self.Pred_groups, process_i * 3 + 3, 1)
            temp_GT_1img = Tensor_slicer(self.GT_groups, (process_i+1) * 2, 1)
            temp_Ovlp1_1img_l2 = Tensor_slicer(self.Pred_groups_l2, process_i * 3 + 2, 1)
            temp_Ovlp2_1img_l2 = Tensor_slicer(self.Pred_groups_l2, process_i * 3 + 3, 1)
            temp_GT_1img_l2 = Tensor_slicer(self.GT_groups_l2, (process_i + 1) * 2, 1)
            temp_Ovlp1_1img_l1 = Tensor_slicer(self.Pred_groups_l1, process_i * 3 + 2, 1)
            temp_Ovlp2_1img_l1 = Tensor_slicer(self.Pred_groups_l1, process_i * 3 + 3, 1)
            temp_GT_1img_l1 = Tensor_slicer(self.GT_groups_l1, (process_i + 1) * 2, 1)
            self.tmmLoss += (L2_loss((temp_Ovlp1_1img + temp_Ovlp2_1img) / 2, temp_GT_1img) +\
                             L2_loss((temp_Ovlp1_1img_l2 + temp_Ovlp2_1img_l2) / 2, temp_GT_1img_l2)*2 +\
                             L2_loss((temp_Ovlp1_1img_l1 + temp_Ovlp2_1img_l1) / 2, temp_GT_1img_l1)*4)

        # type4. Temporal Difference Loss for stride1 (Eq.(4))
        self.tdLoss = tf.constant(0, dtype=tf.float32)
        for process_i in range(self.n_final_ovlp_seq - 1):
            temp_Pred_front_1img = Tensor_slicer(self.Final_Ovlp_Seq, process_i, 1)
            temp_Pred_back_1img = Tensor_slicer(self.Final_Ovlp_Seq, process_i + 1, 1)
            temp_Pred_diff_1img = temp_Pred_back_1img - temp_Pred_front_1img
            temp_GT_front_1img = Tensor_slicer(self.GT_groups, process_i, 1)
            temp_GT_back_1img = Tensor_slicer(self.GT_groups, process_i + 1, 1)
            temp_GT_diff_1img = temp_GT_back_1img - temp_GT_front_1img

            temp_Pred_front_1img_l2 = Tensor_slicer(self.Final_Ovlp_Seq_l2, process_i, 1)
            temp_Pred_back_1img_l2 = Tensor_slicer(self.Final_Ovlp_Seq_l2, process_i + 1, 1)
            temp_Pred_diff_1img_l2 = temp_Pred_back_1img_l2 - temp_Pred_front_1img_l2
            temp_GT_front_1img_l2 = Tensor_slicer(self.GT_groups_l2, process_i, 1)
            temp_GT_back_1img_l2 = Tensor_slicer(self.GT_groups_l2, process_i + 1, 1)
            temp_GT_diff_1img_l2 = temp_GT_back_1img_l2 - temp_GT_front_1img_l2

            temp_Pred_front_1img_l1 = Tensor_slicer(self.Final_Ovlp_Seq_l1, process_i, 1)
            temp_Pred_back_1img_l1 = Tensor_slicer(self.Final_Ovlp_Seq_l1, process_i + 1, 1)
            temp_Pred_diff_1img_l1 = temp_Pred_back_1img_l1 - temp_Pred_front_1img_l1
            temp_GT_front_1img_l1 = Tensor_slicer(self.GT_groups_l1, process_i, 1)
            temp_GT_back_1img_l1 = Tensor_slicer(self.GT_groups_l1, process_i + 1, 1)
            temp_GT_diff_1img_l1 = temp_GT_back_1img_l1 - temp_GT_front_1img_l1

            self.tdLoss += (L2_loss(temp_Pred_diff_1img, temp_GT_diff_1img) +\
                            L2_loss(temp_Pred_diff_1img_l2, temp_GT_diff_1img_l2)*2 +\
                            L2_loss(temp_Pred_diff_1img_l1, temp_GT_diff_1img_l1)*4)
        
        # Total Loss for stride1
        self.totalLoss_s1 = self.recn_lambda * self.recnLoss + self.tm1_lambda * self.tmLoss \
                          + self.tmm_lambda * self.tmmLoss + self.td_lambda * self.tdLoss


        """ type5~7 for stride2 (ss2) """
        # input setting for stride2
        data_ph_5dim_s2 = [] # temporary input for stride2
        for idx in range(3):
            data_ph_5dim_s2.append(Tensor_slicer(self.data_ph_5dim, idx * 2, 1)) #
        data_ph_5dim_s2 = tf.concat(data_ph_5dim_s2, axis=1)  # 5dim, [bs,3*1, h, w, 3] for ss2
        self.data_only_ph_5dim_ss2 = tf_merge_seq_dim(data_ph_5dim_s2)  # 4dim, [bs, h, w, 3*3] for ss2
        self.data_ph_5dim_s2 = tf.concat((self.data_only_ph_5dim_ss2, self.flow_ss2_ph, self.warp_ss2_ph), axis=3) # final input for stride2

        # Predictions setting for stride2
        self.recnLoss_ss2 = tf.constant(0, dtype=tf.float32)
        for process_i in range(self.n_out_groups_sride2):
            # in our case, just 1 iteration.
            [self.Pred_groups_ss2_l1, self.Pred_groups_ss2_l2, self.Pred_groups_ss2] = self.model(self.data_ph_5dim_s2, self.scale_factor, reuse=True,
                                              scope='FISRnet')  # [bs, 3, h, w, ch]
            self.Pred_groups_ss2 = tf_split_seq_dim(self.Pred_groups_ss2)  # [bs, 3, h, w, ch], not overlapped yet.
            self.Pred_groups_ss2_l2 = tf_split_seq_dim(self.Pred_groups_ss2_l2)  # [bs, 3, h, w, ch], not overlapped yet.
            self.Pred_groups_ss2_l1 = tf_split_seq_dim(self.Pred_groups_ss2_l1)  # [bs, 3, h, w, ch], not overlapped yet.

        # GT setting for stride2
        GT_groups_ss2 = []
        for GT_index in range(3):
            GT_groups_ss2.append(Tensor_slicer(self.GT_groups, GT_index * 2 + 1, 1))
        self.GT_groups_ss2 = tf.concat(GT_groups_ss2, axis=1)  # [bs,3*1, h, w, ch] for ss2, 5dim
        GT_groups_ss2_l2 = []
        for GT_index in range(3):
            GT_groups_ss2_l2.append(Tensor_slicer(self.GT_groups_l2, GT_index * 2 + 1, 1))
        self.GT_groups_ss2_l2 = tf.concat(GT_groups_ss2_l2, axis=1)  # [bs,3*1, h, w, ch] for ss2, 5dim
        GT_groups_ss2_l1 = []
        for GT_index in range(3):
            GT_groups_ss2_l1.append(Tensor_slicer(self.GT_groups_l1, GT_index * 2 + 1, 1))
        self.GT_groups_ss2_l1 = tf.concat(GT_groups_ss2_l1, axis=1)  # [bs,3*1, h, w, ch] for ss2, 5dim

        # type5. Reconstruction Loss for stride2 (Eq.(7))
        self.recnLoss_ss2 += (L2_loss(self.Pred_groups_ss2, self.GT_groups_ss2) + \
                             L2_loss(self.Pred_groups_ss2_l2, self.GT_groups_ss2_l2)*2 + \
                             L2_loss(self.Pred_groups_ss2_l1, self.GT_groups_ss2_l1)*4)

        # type6. Temporal Difference Loss for stride2 (Eq.(5))
        self.tdLoss_ss2 = tf.constant(0, dtype=tf.float32)
        for process_i in range(3 - 1):
            temp_Pred_front_1img_ss2 = Tensor_slicer(self.Pred_groups_ss2, process_i, 1)
            temp_Pred_back_1img_ss2 = Tensor_slicer(self.Pred_groups_ss2, process_i + 1, 1)
            temp_Pred_diff_1img_ss2 = temp_Pred_back_1img_ss2 - temp_Pred_front_1img_ss2

            temp_GT_front_1img_ss2 = Tensor_slicer(self.GT_groups_ss2, process_i, 1)
            temp_GT_back_1img_ss2 = Tensor_slicer(self.GT_groups_ss2, process_i + 1, 1)
            temp_GT_diff_1img_ss2 = temp_GT_back_1img_ss2 - temp_GT_front_1img_ss2

            temp_Pred_front_1img_ss2_l2 = Tensor_slicer(self.Pred_groups_ss2_l2, process_i, 1)
            temp_Pred_back_1img_ss2_l2 = Tensor_slicer(self.Pred_groups_ss2_l2, process_i + 1, 1)
            temp_Pred_diff_1img_ss2_l2 = temp_Pred_back_1img_ss2_l2 - temp_Pred_front_1img_ss2_l2

            temp_GT_front_1img_ss2_l2 = Tensor_slicer(self.GT_groups_ss2_l2, process_i, 1)
            temp_GT_back_1img_ss2_l2 = Tensor_slicer(self.GT_groups_ss2_l2, process_i + 1, 1)
            temp_GT_diff_1img_ss2_l2 = temp_GT_back_1img_ss2_l2 - temp_GT_front_1img_ss2_l2

            temp_Pred_front_1img_ss2_l1 = Tensor_slicer(self.Pred_groups_ss2_l1, process_i, 1)
            temp_Pred_back_1img_ss2_l1 = Tensor_slicer(self.Pred_groups_ss2_l1, process_i + 1, 1)
            temp_Pred_diff_1img_ss2_l1 = temp_Pred_back_1img_ss2_l1 - temp_Pred_front_1img_ss2_l1

            temp_GT_front_1img_ss2_l1 = Tensor_slicer(self.GT_groups_ss2_l1, process_i, 1)
            temp_GT_back_1img_ss2_l1 = Tensor_slicer(self.GT_groups_ss2_l1, process_i + 1, 1)
            temp_GT_diff_1img_ss2_l1 = temp_GT_back_1img_ss2_l1 - temp_GT_front_1img_ss2_l1

            self.tdLoss_ss2 += (L2_loss(temp_Pred_diff_1img_ss2, temp_GT_diff_1img_ss2) + \
                                L2_loss(temp_Pred_diff_1img_ss2_l2, temp_GT_diff_1img_ss2_l2)*2 +\
                                L2_loss(temp_Pred_diff_1img_ss2_l1, temp_GT_diff_1img_ss2_l1)*4)

        # type7. Temporal Matching Loss for stride2 (Eq.(2))
        self.tmLoss_ss2 = tf.constant(0, dtype=tf.float32)
        Pred_groups_ss1 = []
        for idx in range(3):
            Pred_groups_ss1.append(Tensor_slicer(self.Final_Ovlp_Seq, idx * 2 + 1, 1))
        self.Pred_groups_ss1 = tf.concat(Pred_groups_ss1, axis=1)  # 5dim, [bs,3*1, h, w, ch] for stride2
        Pred_groups_ss1_l2 = []
        for idx in range(3):
            Pred_groups_ss1_l2.append(Tensor_slicer(self.Final_Ovlp_Seq_l2, idx * 2 + 1, 1))
        self.Pred_groups_ss1_l2 = tf.concat(Pred_groups_ss1_l2, axis=1)  # 5dim, [bs,3*1, h, w, ch] for stride2
        Pred_groups_ss1_l1 = []
        for idx in range(3):
            Pred_groups_ss1_l1.append(Tensor_slicer(self.Final_Ovlp_Seq_l1, idx * 2 + 1, 1))
        self.Pred_groups_ss1_l1 = tf.concat(Pred_groups_ss1_l1, axis=1)  # 5dim, [bs,3*1, h, w, ch] for stride2
        self.tmLoss_ss2 += L2_loss(self.Pred_groups_ss2, self.Pred_groups_ss1) + \
                           L2_loss(self.Pred_groups_ss2_l2, self.Pred_groups_ss1_l2)*2 + \
                           L2_loss(self.Pred_groups_ss2_l1, self.Pred_groups_ss1_l1)*4
        
        # Total Loss for stride2
        self.totalLoss_ss2 = self.recn_lambda * self.recnLoss_ss2 + self.td_lambda * self.tdLoss_ss2 \
                         + self.tm2_lambda * self.tmLoss_ss2

        """ Total Loss & PSNR """
        self.total_loss = self.totalLoss_s1 + self.ss2_lambda * self.totalLoss_ss2
        self.train_PSNR = tf.reduce_mean(
            tf.image.psnr(self.Final_Ovlp_Seq, self.GT_groups, max_val=1.0))  # conversion of img range: [0,1]

        """ Optimizer """
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optim = tf.train.AdamOptimizer(self.lr) \
                .minimize(self.total_loss, global_step=self.global_step)

        """" Validation (independent implementation for clarity)"""
        # define variables for data (val input)
        self.val_input_ph = tf.placeholder(tf.float32, shape=(self.val_batch_size, self.data_sz[1], self.data_sz[2], self.data_sz[3]))
        # define variables for label (val GT)
        self.val_output_ph = tf.placeholder(tf.float32, shape=(self.val_batch_size, self.data_sz[1]*self.scale_factor,
                                                               self.data_sz[2]*self.scale_factor, self.label_sz[3]))
        self.val_GT_groups = tf_split_seq_dim(self.val_output_ph) # converts to 5dim, [bs, 7, h, w, ch] format
        # define variable for flow
        self.val_flow_ph = tf.placeholder(tf.float32,
                                      shape=(self.val_batch_size, self.data_sz[1], self.data_sz[2], 16))
        self.val_warp_ph = tf.placeholder(tf.float32,
                                      shape=(self.val_batch_size, self.data_sz[1], self.data_sz[2], 24))

        """ Recurrent modeling again (reuse FISRnet) for val. """
        for process_i in range(self.n_out_groups_sride1):
            if process_i == 0:
                data_only_val = Tensor_slicer_recurrent(self.val_input_ph, process_i)
                flow_only_val = Tensor_slicer_recurrent_flow(self.val_flow_ph, process_i)
                warp_only_val = Tensor_slicer_recurrent_warp(self.val_warp_ph, process_i)
                input_val = tf.concat((data_only_val, flow_only_val, warp_only_val), axis=3)
                [_, _, self.val_Pred_groups] = self.model(input_val
                                                  , self.scale_factor, reuse=True,
                                              scope='FISRnet')  # [bs, h, w, ch*3]
                self.val_Pred_groups = tf_split_seq_dim(self.val_Pred_groups)  # [bs, 3, h, w, ch], not overlapped yet.
            else:
                data_only_val = Tensor_slicer_recurrent(self.val_input_ph, process_i)
                flow_only_val = Tensor_slicer_recurrent_flow(self.val_flow_ph, process_i)
                warp_only_val = Tensor_slicer_recurrent_warp(self.val_warp_ph, process_i)
                input_val = tf.concat((data_only_val, flow_only_val, warp_only_val), axis=3)
                [_, _, Pred_temp] = self.model(input_val
                                       , self.scale_factor, reuse=True, scope='FISRnet')
                Pred_temp = tf_split_seq_dim(Pred_temp)  # [bs, 3, h, w, ch], not overlapped yet.
                self.val_Pred_groups = tf.concat([self.val_Pred_groups, Pred_temp],
                                             axis=1)  # [bs, 3*3, h, w, ch], not overlapped yet.
                
        self.val_Final_Ovlp_Seq = Groups2Ovlp(self.val_Pred_groups) # [bs, n_final_ovlp_seq, h, w, ch]

        # Simply check recnLoss and PSNR for tendency.
        self.val_recnLoss = L2_loss(self.val_Final_Ovlp_Seq, self.val_GT_groups)
        self.val_PSNR = tf.reduce_mean(
            tf.image.psnr(self.val_Final_Ovlp_Seq, self.val_GT_groups, max_val=1.0))  # conversion of img range: [0,1]

        """" Summary for tensorboard """
        # summaries for train (not multiplied by weight parameter lambda, itself)
        self.recnLoss_sum = tf.summary.scalar("Reconstruction Loss for stride1 (Eq.(6))", self.recnLoss)
        self.tmLoss_sum = tf.summary.scalar("Temporal Matching Loss for stride1 (Eq.(1))", self.tmLoss)
        self.tmmLoss_sum = tf.summary.scalar("Temporal Matching Mean Loss (Eq.(3))", self.tmmLoss)
        self.tdLoss_sum = tf.summary.scalar("Temporal Difference Loss for stride1 (Eq.(4))", self.tdLoss)
        self.totalLoss_s1_sum = tf.summary.scalar("Total Loss for stride1", self.totalLoss_s1)

        self.recnLoss_ss2_sum = tf.summary.scalar("Reconstruction Loss for stride2 (Eq.(7))", self.recnLoss_ss2)
        self.tdLoss_ss2_sum = tf.summary.scalar("Temporal Difference Loss for stride2 (Eq.(5))", self.tdLoss_ss2)
        self.tmLoss_ss2_sum = tf.summary.scalar("Temporal Matching Loss for stride2 (Eq.(2))", self.tmLoss_ss2)
        self.totalLoss_ss2_sum = tf.summary.scalar("Total Loss for stride2", self.totalLoss_ss2)

        self.total_loss_sum = tf.summary.scalar("Total Loss", self.total_loss)
        self.train_PSNR_sum = tf.summary.scalar("train_PSNR", self.train_PSNR)

        # summaries for val
        self.val_recnLoss_sum = tf.summary.scalar("val_recnLoss", self.val_recnLoss)
        self.val_PSNR_sum = tf.summary.scalar("val_PSNR", self.val_PSNR)

        # summaries for images (train)
        images_sum = []
        for s in range(self.n_final_ovlp_seq):
            """" showed in YUV color space. """
            images_sum.append(tf.summary.image('Seq%d_Pred' % s,
                                              tf.image.convert_image_dtype(
                                                  tf.clip_by_value(
                                                      self.Final_Ovlp_Seq[0:(self.n_train_img_showed + 1), s, :, :, :],
                                                      0, 1), dtype=tf.uint8)))
            images_sum.append(tf.summary.image('Seq%d_GT' % s, tf.image.convert_image_dtype(
                self.GT_groups[0:(self.n_train_img_showed + 1), s, :, :, :], dtype=tf.uint8)))

        # merging summaries
        self.train_ss1_summary_loss = tf.summary.merge([self.recnLoss_sum, self.tmLoss_sum,
                                                self.tmmLoss_sum, self.tdLoss_sum,self.totalLoss_s1_sum])
        self.train_ss2_summary_loss = tf.summary.merge([self.recnLoss_ss2_sum, self.tdLoss_ss2_sum,
                                                        self.tmLoss_ss2_sum, self.totalLoss_ss2_sum])
        self.total_train_summary_loss = tf.summary.merge([self.total_loss_sum,
                                                        self.train_PSNR_sum])
        
        self.images_summary = tf.summary.merge(images_sum)
        self.Final_train_summary = tf.summary.merge(
            [self.train_ss1_summary_loss, self.train_ss2_summary_loss, self.total_train_summary_loss])
        self.Final_val_summary = tf.summary.merge([self.val_recnLoss_sum, self.val_PSNR_sum])

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver(max_to_keep=1)

        # summary writer (tensorboard)
        summary_dir = os.path.join(self.log_dir, self.model_dir)
        check_folder(summary_dir)

        self.writer = tf.summary.FileWriter(summary_dir, self.sess.graph)

        counter = 1

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / (self.train_iter))
            start_batch_id = checkpoint_counter - start_epoch * (self.train_iter)
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")
            
        # loop for epoch
        start_time = time.time()
        lr = self.lr # lr_type == 'stair_decay', 'no_decay' 인 경우에는 그대로 lr이 쓰임
        for epoch in range(start_epoch, self.epoch):
            """ define lists for display """
            train_PSNR_list_for_epoch = []
            recnLoss_list_for_epoch = []
            tmLoss_list_for_epoch = []
            tmmLoss_list_for_epoch = []
            tdLoss_list_for_epoch = []
            totalLoss_s1_list_for_epoch = []

            recnLoss_ss2_list_for_epoch = []
            tdLoss_ss2_list_for_epoch = []
            tmLoss_ss2_list_for_epoch = []
            totalLoss_ss2_list_for_epoch = []

            total_loss_list_for_epoch = []

            # shuffle indices(order) of whole training data per each epoch
            rand_idx = np.random.permutation(self.data_sz[0] - self.val_data_size)
            for idx in range(self.train_iter):
                data_batch = self.data[rand_idx[self.batch_size * idx:self.batch_size * (idx + 1)], :, :, :]
                label_batch = self.label[rand_idx[self.batch_size * idx:self.batch_size * (idx + 1)], :, :, :]
                flow_batch = self.flow[rand_idx[self.batch_size * idx:self.batch_size * (idx + 1)], :, :, :]
                flow_batch_ss2 = self.flow_ss2[rand_idx[self.batch_size * idx:self.batch_size * (idx + 1)], :, :, :]
                warp_batch = self.warp[rand_idx[self.batch_size * idx:self.batch_size * (idx + 1)], :, :, :]
                warp_batch_ss2 = self.warp_ss2[rand_idx[self.batch_size * idx:self.batch_size * (idx + 1)], :, :, :]
                if self.lr_type == "linear_decay":
                    lr = self.init_lr if epoch < self.lr_linear_decay_point else self.init_lr * (self.epoch - epoch) / (
                            self.epoch - self.lr_linear_decay_point)  # linear decay
                    feed_dict = {self.lr: lr, self.data_ph: data_batch, self.label_ph: label_batch,
                                 self.flow_ph: flow_batch, self.flow_ss2_ph: flow_batch_ss2, self.warp_ph: warp_batch, self.warp_ss2_ph: warp_batch_ss2}
                else:
                    # self.lr_type == "stair_decay" or "no_decay"
                    feed_dict = {self.data_ph: data_batch, self.label_ph: label_batch,
                                 self.flow_ph: flow_batch, self.flow_ss2_ph: flow_batch_ss2, self.warp_ph: warp_batch, self.warp_ss2_ph: warp_batch_ss2}

                # run the session by feeding (one iteration for train)
                _, Final_summary_str, images_summary_str, \
                recnLoss, tmLoss, \
                tmmLoss, tdLoss, totalLoss_s1, \
                recnLoss_ss2, tdLoss_ss2, \
                tmLoss_ss2, totalLoss_ss2, total_loss, train_PSNR, lr_per_epoch = self.sess.run(
                    [self.optim, self.Final_train_summary, self.images_summary, self.recnLoss,
                     self.tmLoss, self.tmmLoss,
                     self.tdLoss, self.totalLoss_s1, self.recnLoss_ss2,
                     self.tdLoss_ss2, self.tmLoss_ss2,
                     self.totalLoss_ss2, self.total_loss, self.train_PSNR, self.lr],
                    feed_dict=feed_dict)
                self.writer.add_summary(Final_summary_str, counter)

                # display the training status
                if np.mod(idx, self.freq_display) == 0:
                    self.writer.add_summary(images_summary_str,counter)
                    print(
                        "Epoch: [%3d], [%4d/%4d]-th batch, time: %4.2f(min.), "
                        "train_PSNR: %.3f, recnLoss: %.6f, tmLoss: %.6f, tmmLoss: %.6f, tdLoss: %.6f, "
                        "totalLoss_s1: %.6f,recnLoss_ss2: %.6f,"
                        "tdLoss_ss2: %.6f, tmLoss_ss2: %.6f, totalLoss_ss2: %.6f, total_loss: %.6f" \
                        % (epoch, idx, self.train_iter, (time.time() - start_time) / 60,
                           train_PSNR, recnLoss, tmLoss, tmmLoss, tdLoss,
                           totalLoss_s1, recnLoss_ss2,
                           tdLoss_ss2, tmLoss_ss2,totalLoss_ss2, total_loss))

                counter += 1
                train_PSNR_list_for_epoch.append(train_PSNR)
                recnLoss_list_for_epoch.append(recnLoss)
                tmLoss_list_for_epoch.append(tmLoss)
                tmmLoss_list_for_epoch.append(tmmLoss)
                tdLoss_list_for_epoch.append(tdLoss)
                totalLoss_s1_list_for_epoch.append(totalLoss_s1)

                recnLoss_ss2_list_for_epoch.append(recnLoss_ss2)
                tdLoss_ss2_list_for_epoch.append(tdLoss_ss2)
                tmLoss_ss2_list_for_epoch.append(tmLoss_ss2)
                totalLoss_ss2_list_for_epoch.append(totalLoss_ss2)

                total_loss_list_for_epoch.append(total_loss)

            train_PSNR_per_epoch = np.mean(train_PSNR_list_for_epoch)
            recnLoss_per_epoch = np.mean(recnLoss_list_for_epoch)
            tmLoss_per_epoch = np.mean(tmLoss_list_for_epoch)
            tmmLoss_per_epoch = np.mean(tmmLoss_list_for_epoch)
            tdLoss_per_epoch = np.mean(tdLoss_list_for_epoch)
            totalLoss_s1_per_epoch = np.mean(totalLoss_s1_list_for_epoch)

            recnLoss_ss2_per_epoch = np.mean(recnLoss_ss2_list_for_epoch)
            tdLoss_ss2_per_epoch = np.mean(tdLoss_ss2_list_for_epoch)
            tmLoss_ss2_per_epoch = np.mean(tmLoss_ss2_list_for_epoch)
            totalLoss_ss2_per_epoch = np.mean(totalLoss_ss2_list_for_epoch)

            total_loss_per_epoch = np.mean(total_loss_list_for_epoch)

            # display the training status (average) per epoch
            print(
                "# (average) Epoch: [%4d], LR: %1.10f, time: %4.2f(minutes), "
                "train_PSNR: %.3f, recnLoss: %.6f, tmLoss: %.6f, tmmLoss: %.6f, tdLoss: %.6f, "
                        "totalLoss_s1: %.6f,recnLoss_ss2: %.6f,"
                        "tdLoss_ss2: %.6f, tmLoss_ss2: %.6f, totalLoss_ss2: %.6f, total_loss: %.6f" \
                % (epoch, lr_per_epoch, (time.time() - start_time) / 60,
                   train_PSNR_per_epoch, recnLoss_per_epoch, tmLoss_per_epoch, tmmLoss_per_epoch, tdLoss_per_epoch,
                   totalLoss_s1_per_epoch, recnLoss_ss2_per_epoch, tdLoss_ss2_per_epoch, tmLoss_ss2_per_epoch,
                   totalLoss_ss2_per_epoch, total_loss_per_epoch
                   ))

            """ For validation """
            val_Loss_recn_list_for_epoch = []
            val_Loss_PSNR_list_for_epoch = []
            for val_idx in range(self.val_iter):
                data_batch_val = self.data_val[self.val_batch_size * val_idx:self.val_batch_size * (val_idx + 1), :, :, :]
                label_batch_val = self.label_val[self.val_batch_size * val_idx:self.val_batch_size * (val_idx + 1), :, :, :]
                flow_batch_val = self.flow_val[self.val_batch_size * val_idx:self.val_batch_size * (val_idx + 1), :, :, :]
                warp_batch_val = self.warp_val[self.val_batch_size * val_idx:self.val_batch_size * (val_idx + 1), :, :, :]
                val_recnLoss, val_PSNR, Final_val_summary = self.sess.run(
                    [self.val_recnLoss, self.val_PSNR, self.Final_val_summary],
                    feed_dict={self.val_input_ph: data_batch_val, self.val_output_ph: label_batch_val,
                               self.val_flow_ph: flow_batch_val, self.val_warp_ph: warp_batch_val})

                val_Loss_recn_list_for_epoch.append(val_recnLoss)
                val_Loss_PSNR_list_for_epoch.append(val_PSNR)

            self.writer.add_summary(Final_val_summary, counter)
            val_recnLoss_per_epoch = np.mean(val_Loss_recn_list_for_epoch)
            val_PSNR_per_epoch = np.mean(val_Loss_PSNR_list_for_epoch)


            print(
                "######### Validation (average),Epoch: [%4d/%4d]-th epoch, time: %4.2f(min.), val_PSNR: %.3f[dB], "
                "recnLoss: %.6f #########" \
                % (epoch, self.epoch, (time.time() - start_time) / 60,
                   val_PSNR_per_epoch,
                   val_recnLoss_per_epoch))

            """ Save model """
            self.save_checkpoint(self.checkpoint_dir, self.global_step.eval())


    def test(self):
        input = tf.placeholder(tf.float32,
                                      shape=(8, 192,  192, 29))
        _,_,_ = self.model(input
                   , self.scale_factor, reuse=False, scope='FISRnet')
        vl = [v for v in tf.global_variables() if
              "FISRnet" in v.name]
        self.saver = tf.train.Saver(var_list=vl)
        
        """" Measure the performance in YUV color space. """
        # saver to save model
        tf.global_variables_initializer().run()  # before "restore"

        # restore the checkpoint
        _, _ = self.load(self.checkpoint_dir)

        test_data_path = sorted(glob.glob(os.path.join(self.test_data_path, '*.png')))
        test_label_path = sorted(glob.glob(os.path.join(self.test_label_path, '*.png')))

        print(" Start to read flow data (test).")
        flow_path = self.test_flow_data_path
        H, W = self.test_input_size
        flow = read_flo_file_5dim(flow_path)
        print(" Successfully load.")
        flow = merge_seq_dim(flow)

        print(" Start to read warped data (test).")
        warp_path = self.test_warped_data_path
        warp = read_mat_file_warp(warp_path, 'pred') # 2K input
        print(" Successfully load.")
        warp = merge_seq_dim(warp)

        num_patch = self.test_patch # due to memory capacity, we divide the whole image into small patches.
        patch_boundary = 32  # multiple of 32
        
        test_FISR_Loss_PSNR_list_for_epoch = []
        test_SR_Loss_PSNR_list_for_epoch = []
        test_FISR_Loss_SSIM_list_for_epoch = []
        test_SR_Loss_SSIM_list_for_epoch = []

        inf_time = []
        start_time = time.time()

        """ make "test_img_dir" per experiment """
        test_img_dir = os.path.join(self.test_img_dir, self.model_dir)
        check_folder(test_img_dir)

        n_in_seq = 3
        n_GT_seq = n_in_seq * 2 - 3  # 3
        n_test_in_seq = 5
        n_test_label_seq = 2 * n_test_in_seq - 3 # 7

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

                ###======== Crop Data for 32 multiple ========###
                # crop img for u-net (32x32)
                h, w = self.test_input_size
                c = n_in_seq*3
                h = h - np.remainder(h, 32*num_patch[0])
                w = w - np.remainder(w, 32*num_patch[1])
                img = img[:h, :w, :] # now, it is divided by 32 with no remainder.
                label = label[:h * self.scale_factor, :w * self.scale_factor, :]

                ###======== Normalize & Clip Image ========###
                img = np.array(img, dtype=np.double) / 255.
                label = np.array(label, dtype=np.double) / 255.
                img = np.expand_dims(np.clip(img, 0, 1), axis=0)
                label = np.expand_dims(np.clip(label, 0, 1), axis=0)

                ###======== Normalize & Clip Flow ========###
                flow_sample = flow[scene_i, :h, :w, 4*sample_i:4*sample_i+8]
                flow_sample = flow_sample/96/2
                flow_sample = np.expand_dims(np.clip(flow_sample, -1, 1), axis=0)

                ###======== Normalize & Clip Warp Image ========###
                warp_sample = warp[scene_i, :h, :w, 6*sample_i:6*sample_i+12]
                warp_sample = np.expand_dims(np.clip(warp_sample, 0, 1), axis=0)

                ###======== Generate Input ========###
                input = np.concatenate([img, flow_sample, warp_sample], axis=3)
                test_Pred_full = np.zeros((h*self.scale_factor, w*self.scale_factor, c))

                ###======== Divide & Process due to Limited Memory ========###
                for p in range(num_patch[0]*num_patch[1]):
                    pH = p // num_patch[1] # patch index (priority: w=>h)
                    pW = p % num_patch[1]  # patch index
                    sH = h // num_patch[0] # patch size
                    sW = w // num_patch[1] # patch size

                    # process data considering patch boundary
                    H_low_ind, H_high_ind, W_low_ind, W_high_ind, add_H, add_W = \
                        get_HW_boundary(patch_boundary, h, w, pH, sH, pW, sW)

                    ###======== Set Model ========###
                    data_test_ph = tf.placeholder(tf.float32,
                                                  shape=(1, sH + add_H, sW + add_W, c + 8 + 12))
                    label_test_ph = tf.placeholder(tf.float32,
                                                   shape=(1, sH * self.scale_factor, sW * self.scale_factor, c * 2 - 9))
                    [_, _, test_Pred] = self.model(data_test_ph, self.scale_factor, reuse=True, scope='FISRnet')

                    ###======== Pre-process Data ========###
                    simg = input[:, H_low_ind:H_high_ind, W_low_ind:W_high_ind, :]
                    slabel = label[:, pH * sH * self.scale_factor: (pH + 1) * sH * self.scale_factor,
                             pW * sW * self.scale_factor: (pW + 1) * sW * self.scale_factor, :]

                    ###======== Run Session ========###
                    rs_time = time.time()
                    test_Pred_patch = self.sess.run(
                        test_Pred, feed_dict={data_test_ph: simg, label_test_ph: slabel})
                    inf_time.append(time.time() - rs_time)

                    # trim patch boundary
                    test_Pred_trim = trim_patch_boundary(test_Pred_patch, patch_boundary, h, w, pH, sH, pW,
                                                      sW, self.scale_factor)
                    # store in pred_full
                    test_Pred_full[pH * sH * self.scale_factor: (pH + 1) * sH * self.scale_factor,
                             pW * sW * self.scale_factor: (pW + 1) * sW * self.scale_factor, :] = np.squeeze(test_Pred_trim)

                ###======== Process Prediction & GT ========###
                test_pred = np.clip(test_Pred_full, 0, 1)
                test_GT = np.squeeze(label)

                ###======== Compute PSNR & Print Results========###
                for seq_i in range(n_GT_seq):
                    test_PSNR.append(utils._compute_psnr(test_pred[:, :, 3 * seq_i:3 * (seq_i + 1)],
                                                         test_GT[:, :, 3 * seq_i:3 * (seq_i + 1)], 1.))
                    test_SSIM.append(compare_ssim(Image.fromarray((test_pred[:, :, 3 * seq_i:3 * (seq_i + 1)] * 255).astype('uint8')),
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
                pred = np.uint8(test_pred * 255) # YUV, range of [0,255]
                # check
                for seq_i in range(n_GT_seq):
                    fr_name = os.path.basename(test_label_path[scene_i * n_test_label_seq + sample_i * 2 + seq_i])
                    fr_name = fr_name[3:]
                    rgb_img = utils.YUV2RGB_matlab(pred[:, :, seq_i * 3:(seq_i + 1) * 3])
                    pred_img = Image.fromarray(rgb_img.astype('uint8'))
                    pred_img.save(os.path.join(test_img_dir, 'pred_{}'.format(fr_name)))

                ###======== Append Loss & PSNR ========###
                test_FISR_Loss_PSNR_list_for_epoch.append(test_PSNR[0])
                test_SR_Loss_PSNR_list_for_epoch.append(test_PSNR[1])
                test_FISR_Loss_SSIM_list_for_epoch.append(test_SSIM[0])
                test_SR_Loss_SSIM_list_for_epoch.append(test_SSIM[1])

                if sample_i == 2:
                    test_FISR_Loss_PSNR_list_for_epoch.append(test_PSNR[2])
                    test_FISR_Loss_SSIM_list_for_epoch.append(test_SSIM[2])

        ###======== Compute Mean PSNR & SSIM for Whole Scenes ========###
        test_FISR_PSNR_per_epoch = np.mean(test_FISR_Loss_PSNR_list_for_epoch)
        test_SR_PSNR_per_epoch = np.mean(test_SR_Loss_PSNR_list_for_epoch)
        test_FISR_SSIM_per_epoch = np.mean(test_FISR_Loss_SSIM_list_for_epoch)
        test_SR_SSIM_per_epoch = np.mean(test_SR_Loss_SSIM_list_for_epoch)

        print(
            "######### Test (average) test_PSNR: FISR %.8f[dB], SR %.8f[dB]  #########" \
            % (test_FISR_PSNR_per_epoch, test_SR_PSNR_per_epoch))
        print(
            "######### Test (average) test_SSIM: FISR %.8f, SR %.8f #########" \
            % (test_FISR_SSIM_per_epoch, test_SR_SSIM_per_epoch))
        print("######### Estimated Inference Time (per one output 4K frame): %.8f[s]  #########" % (
                    np.mean(inf_time) * self.test_patch[0] * self.test_patch[1]))

    def FISR_for_video(self, flow_file_name, warp_file_name):
        input = tf.placeholder(tf.float32,
                               shape=(8, 192, 192, 29))
        _, _, _ = self.model(input
                             , self.scale_factor, reuse=False, scope='FISRnet')
        vl = [v for v in tf.global_variables() if
              "FISRnet" in v.name]
        self.saver = tf.train.Saver(var_list=vl)
        
        """" Make joint spatial-temporal upscaling (FISR) frames for input frames in one folder """
        # saver to save model
        tf.global_variables_initializer().run()  # before "restore"

        # restore the checkpoint
        _, _ = self.load(self.checkpoint_dir)

        test_data_path = glob.glob(os.path.join(self.frame_folder_path, '*.png')) # YUV
        num_fr = self.frame_num

        """ Make (ex))"E:/FISR_Github/FISR_test_folder/scene1/FISR_frames" to save frames  """
        FISR_img_dir = os.path.join(self.frame_folder_path,
                                    'FISR_frames')
        check_folder(FISR_img_dir)

        print(" Start to read flow data (FISR test).")
        flow_path = flow_file_name
        flow = read_flo_file_5dim(flow_path)  # [N, 8, h, w, 2], here: [N, 2, h, w, 2]
        print(" Successfully load.")
        flow = np.concatenate((flow[0:num_fr - 2, :, :, :, :], flow[1:num_fr - 1, :, :, :, :]),
                              axis=1)  # [N-1, 2+2(bidirectional), h, w, 2],
        flow = merge_seq_dim(flow)

        print(" Start to read warped data (FISR test).")
        warp_path = warp_file_name
        warp = read_mat_file_warp(warp_path, 'pred')  # [N, 2, h, w, 3]
        warp = np.concatenate((warp[0:num_fr - 2, :, :, :, :], warp[1:num_fr - 1, :, :, :, :]),
                              axis=1)  # [N-1, 2+2(bidirectional), h, w, 3]
        print(" Successfully load.")
        warp = merge_seq_dim(warp)



        num_patch = self.FISR_test_patch  # due to memory capacity, we divide the whole image into small patches.
        patch_boundary = 32  # multiple of 32


        inf_time = []
        start_time = time.time()

        """ make "test_img_dir" per experiment """
        test_img_dir = os.path.join(self.test_img_dir, self.model_dir)
        check_folder(test_img_dir)

        n_in_seq = 3
        n_test_in_seq = 5
        H, W = self.FISR_input_size

        for fr in range(num_fr - 2):
            ###======== Read & Compose Data ========###
            for seq_i in range(n_in_seq):
                img_temp = np.array(Image.open(test_data_path[fr + seq_i]))
                if seq_i == 0:
                    img = img_temp
                else:
                    img = np.concatenate((img, img_temp), axis=2)

            ###======== Crop Data for 32 multiple ========###
            # crop img for u-net (32x32)
            c = n_in_seq * 3
            h = H - np.remainder(H, 32 * num_patch[0])
            w = W - np.remainder(W, 32 * num_patch[1])
            img = img[:h, :w, :]  # now, it is divided by 32 with no remainder.

            ###======== Normalize & Clip Image ========###
            img = np.array(img, dtype=np.double) / 255.
            img = np.expand_dims(np.clip(img, 0, 1), axis=0)

            ###======== Normalize & Clip Flow ========###
            flow_sample = flow[fr, :h, :w, :]
            flow_sample = flow_sample / 96 / 2
            flow_sample = np.expand_dims(np.clip(flow_sample, -1, 1), axis=0)

            ###======== Normalize & Clip Warp Image ========###
            warp_sample = warp[fr, :h, :w, :]
            warp_sample = np.expand_dims(np.clip(warp_sample, 0, 1), axis=0)

            ###======== Generate Input ========###
            input = np.concatenate([img, flow_sample, warp_sample], axis=3)
            test_Pred_full = np.zeros((h * self.scale_factor, w * self.scale_factor, c))

            ###======== Divide & Process due to Limited Memory ========###
            for p in range(num_patch[0] * num_patch[1]):
                pH = p // num_patch[1]  # patch index (priority: w=>h)
                pW = p % num_patch[1]  # patch index
                sH = h // num_patch[0]  # patch size
                sW = w // num_patch[1]  # patch size

                # process data considering patch boundary
                H_low_ind, H_high_ind, W_low_ind, W_high_ind, add_H, add_W = \
                    get_HW_boundary(patch_boundary, h, w, pH, sH, pW, sW)

                ###======== Set Model ========###
                data_test_ph = tf.placeholder(tf.float32,
                                              shape=(1, sH + add_H, sW + add_W, c + 8 + 12))
                [_, _, test_Pred] = self.model(data_test_ph, self.scale_factor, reuse=True, scope='FISRnet')

                ###======== Pre-process Data ========###
                simg = input[:, H_low_ind:H_high_ind, W_low_ind:W_high_ind, :]

                ###======== Run Session ========###
                rs_time = time.time()
                test_Pred_patch = self.sess.run(
                    test_Pred, feed_dict={data_test_ph: simg})
                inf_time.append(time.time() - rs_time)

                # trim patch boundary
                test_Pred_trim = trim_patch_boundary(test_Pred_patch, patch_boundary, h, w, pH, sH, pW,
                                                     sW, self.scale_factor)
                # store in pred_full
                test_Pred_full[pH * sH * self.scale_factor: (pH + 1) * sH * self.scale_factor,
                pW * sW * self.scale_factor: (pW + 1) * sW * self.scale_factor, :] = np.squeeze(test_Pred_trim)

            ###======== Process Prediction & GT ========###
            test_pred = np.clip(test_Pred_full, 0, 1)


            ###======== Save Predictions as both RGB & YUV Images  ========###
            # by considering the overlapping, the frame from the later sliding window is taken for simplicity
            pred = np.uint8(test_pred * 255)  # YUV, range of [0,255]
            for seq_i in range(3):
                rgb_img = utils.YUV2RGB_matlab(pred[:, :, seq_i * 3:(seq_i + 1) * 3])
                pred_img = Image.fromarray(rgb_img.astype('uint8'))
                fr_name = FISR_img_dir + '/pred_{}.png'.format(
                    str(fr * 2 + seq_i).zfill(math.ceil(math.log10(2*(num_fr-1)))))
                pred_img.save(fr_name)
                # fit the video format. ex) if num_fr = 60 => math.ceil(math.log10(num_fr))) = 2
                yuv_img = pred[:, :, seq_i * 3:(seq_i + 1) * 3]
                yuv_img = Image.fromarray(yuv_img.astype('uint8'))
                fr_name = FISR_img_dir + '/pred_YUV_{}.png'.format(
                    str(fr * 2 + seq_i).zfill(math.ceil(math.log10(2 * (num_fr - 1)))))
                yuv_img.save(fr_name)
            
            print(
                " <FISR processing> [%4d/%4d]-th input multiple data sample (stride1), time: %4.4f(minutes)  " \
                % (fr + 1, num_fr - 2, (time.time() - start_time) / 60))

        print("######### Estimated Inference Time (per one output 4K frame): %.8f[s]  #########" % (
                np.mean(inf_time) * self.test_patch[0] * self.test_patch[1]))

    @property
    def model_dir(self):
        return "{}_exp{}".format(
            self.model_name, self.exp_num)


    def save_checkpoint(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name),
                        global_step=step)  # "self.model_name+'.model'": file name of checkpoint

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
