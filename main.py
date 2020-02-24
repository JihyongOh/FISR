"""
-------------------------------------------------------------------------------------------------------------
<Official Tensorflow Code>
Paper: "FISR: Deep Joint Frame Interpolation and Super-Resolution with A Multi-scale Temporal Loss", AAAI2020
Written by Jihyong Oh and Soo Ye Kim
-------------------------------------------------------------------------------------------------------------
"""

from __future__ import print_function
import argparse, os # argparse: CLI tool, diverse option switch, it makes this convenient.
import tensorflow as tf
from utils import show_all_variables
from utils import check_folder
import utils
from FISRnet import FISRnet
from FISR_tfoptflow.FISR_for_video_pwcnet_predict_from_img_test import FISR_for_video_Compute_Flow
from FISR_tfoptflow.FISR_for_video_warp_img_with_flo import FISR_for_video_Warp_Img
""" Select GPU number """
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

""" Parsing and Configurations """
def parse_args():
    desc = "FISR: Deep Joint Frame Interpolation and Super-Resolution with A Multi-scale Temporal Loss"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--net_type', type=str, default='FISRnet', choices=['FISRnet'], help='The type of Net')
    parser.add_argument('--fraction_gpu', type=float, default=1.0, help='The fraction rate of gpu')
    parser.add_argument('--phase', type=str, default='FISR_for_video', choices=['train', 'test', 'FISR_for_video'])
    parser.add_argument('--scale_factor', type=float, default=2, help='scale factor for SR')
    
    """ Information of directories """
    """
    note: we constructed 10,086 training samples in total before starting the training process 
            to avoid heavy training time required for loading 4K frames at every iteration.
    """
    parser.add_argument('--train_data_path', type=str, default='./data/train/LR_LFR/LR_Surfing_SlamDunk_5seq.mat', 
                        help='train_data_path (input, LR LFR), where 4K .mat file (pre-made) is located.') # matlab format: (96,96,3,5,10086)
    parser.add_argument('--train_flow_data_path', type=str, default='./data/train/flow/LR_Surfing_SlamDunk_5seq_ss1.flo',
                        help='train flow (tesmporal stride1) data path (input, LR LFR)')
    parser.add_argument('--train_flow_ss2_data_path', type=str, default='./data/train/flow/LR_Surfing_SlamDunk_5seq_ss2.flo',
                        help='train flow (tesmporal stride2) data path (input, LR LFR)')
    parser.add_argument('--train_warped_data_path', type=str, default='./data/train/warped/LR_Surfing_SlamDunk_5seq_ss1_warp.mat',
                        help='train warped (tesmporal stride1) data path (input, LR LFR)')
    parser.add_argument('--train_wapred_ss2_data_path', type=str, default='./data/train/warped/LR_Surfing_SlamDunk_5seq_ss2_warp.mat',
                        help='train warped (tesmporal stride1) data path (input, LR LFR)')
    parser.add_argument('--train_label_path', type=str, default='./data/train/HR_HFR/HR_Surfing_SlamDunk_5seq.mat', help='train_label_path (GT, HR HRF)')

    parser.add_argument('--test_data_path', type=str, default='./data/test/LR_LFR',
                        help='test_data_path (input, LR LFR), where .png files (YUV) are loocated.')
    parser.add_argument('--test_flow_data_path', type=str, default='./data/test/flow/LR_Surfing_SlamDunk_test_ss1.flo',
                        help='test flow data path (input, LR LFR)')
    parser.add_argument('--test_warped_data_path', type=str,default='./data/test/warped/LR_Surfing_SlamDunk_test_ss1_warp.mat',
                        help='test warped data path (input, LR LFR)')
    parser.add_argument('--test_label_path', type=str, default='./data/test/HR_HFR',
                        help='test_label_path (GT, HR HFR), where .png files (YUV) are loocated')

    parser.add_argument('--test_img_dir', type=str, default='./test_img_dir', help='test_img_dir path')
    parser.add_argument('--text_dir', type=str, default='./text_dir', help='text_dir path')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint_dir', help='checkpoint_dir')
    parser.add_argument('--log_dir', type=str, default='./logdir',
                    help='Directory name to save training logs')
    
    """ Hyperparameters for Training (when [phase=='train']) """
    parser.add_argument('--exp_num', type=int, default=1, help='The experiment number')
    parser.add_argument('--epoch', type=int, default=100, help='The number of epochs to run')
    parser.add_argument('--freq_display', type=int, default=100, help='The iterations frequency for display')
    parser.add_argument('--init_lr', type=float, default=0.0001, help='The initial learning rate')
    parser.add_argument('--lr_type', type=str, default='stair_decay',
                        choices=['linear_decay', 'stair_decay', 'no_decay'])
    parser.add_argument('--lr_stair_decay_points', type=int, nargs='+',
                        help='stair_decay - The points where the lr to be decayed', default=[80, 90])
    parser.add_argument('--lr_decreasing_factor', type=float, default=0.1, help='stair_decay - lr_decreasing_factor')
    parser.add_argument('--lr_linear_decay_point', type=int, default=50, help='linear_decay - lr point where linearly decreasing starts')
    parser.add_argument('--batch_size', type=int, default=8, help='The size of batch size.')
    parser.add_argument('--n_train_img_showed', type=int, default=3, help='The numbers of train image and predicted image for figures in tensorboard.')
    parser.add_argument('--val_batch_size', type=int, default=2, help='The size of validation batch size.')
    parser.add_argument('--val_data_size', type=int, default=320, help='The size of validation data size.')

    """ Weighting Parameters Lambda for Temporal Loss (when [phase=='train']) """
    parser.add_argument('--recn_lambda', type=float, default=1.0, help='Lambda for Reconstruction Loss (Eq.(6),(7))')
    parser.add_argument('--tm1_lambda', type=float, default=1.0, help='Lambda for Temporal Matching Loss for stride1 (Eq.(1))')
    parser.add_argument('--tm2_lambda', type=float, default=0.1, help='Lambda for Temporal Matching Loss for stride2 (Eq.(2))')
    parser.add_argument('--tmm_lambda', type=float, default=1.0, help='Lambda for Temporal Matching Mean Loss (Eq.(3))')
    parser.add_argument('--td_lambda', type=float, default=0.1, help='Lambda for Temporal Difference Loss (Eq.(4),(5))')
    parser.add_argument('--ss2_lambda', type=float, default=1.0, help='Lambda for Total Loss (stride2)')
    
    
    """ Settings for Testing (when [phase=='test']) """
    parser.add_argument('--test_patch', type=tuple, default=(2, 2),
                        help='Divide img into patches in case of low memory')
    parser.add_argument('--test_input_size', type=tuple, default=(1080, 1920),
                        help='Input size for test, default=2K')
    
    
    """ Settings for FISRing (when [phase=='FISR_for_video']) """
    parser.add_argument('--frame_folder_path', type=str, 
                        default='E:/FISR_Github/FISR_test_folder/scene1', help='Folder root, which contains .png files  (YUV) of frames.')
    parser.add_argument('--FISR_input_size', type=tuple, default=(1080, 1920),
                        help='Input size for FISR, default=2K')
    parser.add_argument('--frame_num', type=int,
                        default=5, help='Numbers of frames to convert')
    parser.add_argument('--FISR_test_patch', type=tuple, default=(2, 2),
                        help='Divide img into patches in case of low memory')

    
    return check_args(parser.parse_args())

def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --text_dir
    check_folder(args.text_dir)

    # --log_dir
    check_folder(args.log_dir)

    # --test_img_dir
    check_folder(args.test_img_dir)
    
    return args

def main():
    args = parse_args()
    if args is None:
        exit()
    models = [FISRnet] 

    if args.phase == 'train':
        """ to record the parser in txt format """
        with open(args.text_dir + '/exp_' + str(args.exp_num) + '.txt', 'a') as log:
            log.write('----- Model parameters -----\n')
            for arg in vars(args):
                log.write('{} : {}\n'.format(arg, getattr(args, arg)))

        """ Open session """
        with tf.Session(config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=args.fraction_gpu,
                                          allow_growth=True))) as sess:
            model_net = None
            for model in models:
                if args.net_type == model.model_name:
                    model_net = model(sess,
                                      args)
            if model_net is None:
                raise Exception("[!] There is no option for " + args.net_type)
    
            # build graph
            print("[*] Exp: ", args.exp_num)
            model_net.build_model()
            # show network architecture
            show_all_variables()
            # launch the graph in a session
            print("Model:", args.net_type)
            print("[*] Training starts")
            model_net.train()
            print("[*] Exp: ", args.exp_num)
            print("[*] Training finished! ")
    
        """ test after training """
        tf.reset_default_graph()  # to delete all graph
        """ to load a lighter train data to reduce a time consumption for loading """
        args.train_data_path = './data/LR_sample_5seq.mat'
        args.train_label_path = './data/HR_sample_5seq.mat'
        """ Open session """
        with tf.Session(config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=args.fraction_gpu,
                                          allow_growth=True))) as sess:

            model_net = None
            for model in models:
                if args.net_type == model.model_name:
                    model_net = model(sess, args)
            if model_net is None:
                raise Exception("[!] There is no option for " + args.net_type)

            # build graph again
            print("[*] Exp: ", args.exp_num)
            print("[*] Building FISRnet ...")
            model_net.build_model()
            print("[*] Successfully build.")
            # show network architecture
            show_all_variables()
            # launch the graph in a session
            print("[*] Testing starts")
            model_net.test()
            print("[*] Exp: ", args.exp_num)
            print("[*] Testing finished! ")
    
    
    elif args.phase == 'test':
        """ measure the performance (PSNR & SSIM) """
        
        """ to load a lighter train data to reduce a time consumption for loading """
        args.train_data_path = './data/LR_sample_5seq.mat'
        args.train_label_path = './data/HR_sample_5seq.mat'
        """ Open session """
        with tf.Session(config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=args.fraction_gpu,
                                          allow_growth=True))) as sess:
    
            model_net = None
            for model in models:
                if args.net_type == model.model_name:
                    model_net = model(sess, args)
            if model_net is None:
                raise Exception("[!] There is no option for " + args.net_type)
    
            # build graph
            print("[*] Exp: ", args.exp_num)
            print("[*] Building FISRnet ...")
            model_net.build_model()
            print("[*] Successfully build.")
            # show network architecture
            show_all_variables()
            # launch the graph in a session
            print("Model:", args.net_type)
            print("[*] Testing starts")
            model_net.test()
            print("[*] Exp: ", args.exp_num)
            print("[*] Testing finished! ")


    elif args.phase == 'FISR_for_video':
        """" Make joint spatial-temporal upscaling (FISR) frames for input frames in one folder """
        """ Open session for computing optical flows by PWC-Net """
        print("[*] Computing and making flow file starts")
        flow_file_name = FISR_for_video_Compute_Flow(args) # make flow file in '.flo' format.
        tf.reset_default_graph()  # to delete all graph
        print("[*] Making warp file starts")
        warp_file_name = FISR_for_video_Warp_Img(args,flow_file_name) # make warp file in '.mat' format.
        
        """ to load a lighter train data to reduce a time consumption for loading """
        args.train_data_path = './data/LR_sample_5seq.mat'
        args.train_label_path = './data/HR_sample_5seq.mat'
        """ Open session """
        with tf.Session(config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=args.fraction_gpu,
                                          allow_growth=True))) as sess:

            model_net = None
            for model in models:
                if args.net_type == model.model_name:
                    model_net = model(sess, args)
            if model_net is None:
                raise Exception("[!] There is no option for " + args.net_type)

            # build graph
            print("[*] Exp: ", args.exp_num)
            print("[*] Building FISRnet ...")
            model_net.build_model()
            print("[*] Successfully build.")
            # show network architecture
            show_all_variables()
            # launch the graph in a session
            print("Model:", args.net_type)
            print("[*] FISR Testing starts")
            model_net.FISR_for_video(flow_file_name, warp_file_name)
            print("[*] Exp: ", args.exp_num)
            print("[*] FISR Testing finished! ")
            

if __name__ == '__main__':
    main()
