from utils import *
##################################################################################
# Layers
##################################################################################

# convolution layer
def Conv2d(x, shape, name):
    w = tf.get_variable(name + '/w', shape, initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    b = tf.get_variable(name + '/b', shape[3], initializer=tf.constant_initializer(0))
    n = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', name=name) + b
    return n

##################################################################################
# Activation function
##################################################################################

def relu(x):
    return tf.nn.relu(x)

##################################################################################
# Loss Functions
##################################################################################

def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss


def L2_loss(x,y):
    loss = tf.reduce_mean(tf.square(x - y))
    return loss

##################################################################################
# Network Blocks
##################################################################################

# residual block
def res_block(x, c, name):
    with tf.variable_scope(name):
        n = Conv2d(relu(x), [3, 3, c, c], 'conv/0')
        n = Conv2d(relu(n), [3, 3, c, c], 'conv/1')
        n = x + n
    return n


# encoder level
def Enc_level_res(x, c1, c, pool_factor, name):
    with tf.variable_scope(name):
        n = Conv2d(x, [3, 3, c1, c], 'conv/0')
        n = res_block(n, c, 'res_block/0')
        n = tf.nn.relu(res_block(n, c, 'res_block/1'))
        skip = n
        n = tf.nn.max_pool(n, [1, pool_factor, pool_factor, 1], [1, pool_factor, pool_factor, 1],'SAME')
    return n, skip


# bottleneck
def Bottleneck_res(x, c1, c, name):
    with tf.variable_scope(name):
        n = Conv2d(x, [3, 3, c1, c], 'conv/0')
        n = relu(res_block(n, c, 'res_block/0'))
    return n


# decoder level
def Dec_level_res(x, skip, c1, c, name, size):
    with tf.variable_scope(name):
        n = tf.image.resize_images(x, [size[0], size[1]], method=tf.image.ResizeMethod.BILINEAR)
        n = relu(Conv2d(n, [3, 3, c1, c], 'resize'))
        n = tf.concat([n, skip], 3)

        n = Conv2d(n, [3, 3, c * 2, c], 'conv/0')
        n = res_block(n, c, 'res_block/0')
        n = relu(res_block(n, c, 'res_block/1'))
    return n

##################################################################################
# Tensor slicer-related functions for easier tensor operation
##################################################################################
def Tensor_slicer(tensor_group, start_idx, n_in_seq):
    """ along axis=1"""
    # tensor_group(5dim): [bs, (5-2)*3, h, w, ch]
    result_tensor = tf.slice(tensor_group,
                         [0, start_idx, 0, 0, 0],
                         [-1, n_in_seq, -1, -1, -1])
    return result_tensor


def Tensor_slicer_recurrent(tensor_group, order):
    """ along axis=3"""
    # tensor_group(4dim): [bs, h, w, ch*5]
    result_tensor = tf.slice(tensor_group,
                         [0, 0, 0, 3*order],
                         [-1, -1, -1, 3*3]) # including 3 subsequent samples per each frame with 3 channels.
    return result_tensor


def Tensor_slicer_recurrent_flow(tensor_group, order):
    """ along axis=3"""
    # tensor_group(4dim): [bs, h, w, ch*5]
    result_tensor = tf.slice(tensor_group,
                         [0, 0, 0, 4*order],
                         [-1, -1, -1, 4*2]) 
    # including 4 bidirectional flows => total ch = 8(=4x2(x,y directions)) => (stride=4(=2(flows)x2(x,y directions)))
    return result_tensor


def Tensor_slicer_recurrent_warp(tensor_group, order):
    """ along axis=3"""
    # tensor_group(4dim): [bs, h, w, ch*5]
    result_tensor = tf.slice(tensor_group,
                         [0, 0, 0, 6*order],
                         [-1, -1, -1, 6*2]) 
    # including 4 bidirectional middle-warped frames= > total ch = 12(=4x3(YUV ch)) => (stride=6(=2(frames)x3(YUV ch)))
    return result_tensor


def Groups2Ovlp(tensor_group):
    # tensor_group: [bs, 9(=(5-2)*3), h, w, ch]
    sz = tensor_group.shape #
    temp_groups = []
    n_out_groups_step1 = int(sz[1]//3) # 9/3 =3
    for process_i in range(n_out_groups_step1):
        if process_i == 0:
            temp_First_1img = Tensor_slicer(tensor_group,process_i*3,1)
            temp_groups.append(temp_First_1img)

        temp_Middle_1img = Tensor_slicer(tensor_group,process_i*3+1,1)
        temp_groups.append(temp_Middle_1img)

        if not process_i == (n_out_groups_step1 - 1):
            temp_Ovlp1_1img = Tensor_slicer(tensor_group, process_i*3 + 2, 1)  # along axis=1
            temp_Ovlp2_1img = Tensor_slicer(tensor_group, process_i*3 + 3, 1)  # along axis=1
            Ovlp_1img = (temp_Ovlp1_1img + temp_Ovlp2_1img) / 2 # reduce_mean
            temp_groups.append(Ovlp_1img)

        else:
            temp_Last_2imgs = Tensor_slicer(tensor_group, (process_i)*3 + 2, 1)
            temp_groups.append(temp_Last_2imgs)
        Final_Ovlp_Seq = tf.concat(temp_groups, axis=1)
        # Final_Ovlp_Seq: [bs, 7(=n_seq*2+1), h, w, ch]

    return Final_Ovlp_Seq


def tf_merge_seq_dim(data):
    # data: [N, N_seq, H, W, C], data_new: [N, H, W, C*N_seq]
    sz = data.shape
    data_new = tf.transpose(data, perm=(0, 2, 3, 1, 4))  # [N, H, W, N_seq, C]
    data_new = tf.reshape(data_new, [sz[0], sz[2], sz[3], sz[1]*sz[4]])  # [N, H, W, C*N_seq]
    return data_new


def tf_split_seq_dim(data):
    # data: [N, H, W, C*N_seq], data_new: [N, N_seq, H, W, C]
    sz = data.shape
    data_new = tf.reshape(data, [sz[0], sz[1], sz[2], sz[3]//3, 3])  # [N, H, W, N_seq, C]
    data_new = tf.transpose(data_new, perm=[0, 3, 1, 2, 4])  # [N, N_seq, H, W, C]
    return data_new


