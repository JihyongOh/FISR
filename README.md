# FISR

[![ArXiv](https://img.shields.io/badge/ArXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/1912.07213)
[![AAAI2020](https://img.shields.io/badge/AAAI2020-Paper-<COLOR>.svg)](https://ojs.aaai.org/index.php/AAAI/article/view/6788)
[![GitHub Stars](https://img.shields.io/github/stars/JihyongOh/FISR?style=social)](https://github.com/JihyongOh/FISR)
![visitors](https://visitor-badge.glitch.me/badge?page_id=JihyongOh/FISR)

**This is the official repository of FISR (AAAI2020).**

We provide the training and test code along with the trained weights and the dataset (train+test) used for FISR. 
If you find this repository useful, please consider citing our [paper](https://arxiv.org/abs/1912.07213).

**Reference**:  
> Soo Ye Kim*, Jihyong Oh*, and Munchurl Kim, "FISR: Deep Joint Frame Interpolation and Super-Resolution with a Multi-scale Temporal Loss", *AAAI Conference on Artificial Intelligence*, 2020. (* *equal contribution*)

**BibTeX**
```bibtex
@inproceedings{kim2020fisr,
  title={FISR: Deep Joint Frame Interpolation and Super-Resolution with a Multi-scale Temporal Loss},
  author={Kim, Soo Ye and Oh, Jihyong and Kim, Munchurl},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2020}
}
```

### Requirements
Our code is implemented using Tensorflow, and was tested under the following setting:  
* Python 3.7 
* Tensorflow 1.13 
* CUDA 10.0  
* cuDNN 7.1.4  
* NVIDIA TITAN Xp GPU
* Windows 10

## Test code
### Quick Start
1. Download the source code in a directory of your choice **\<source_path\>**.
2. Download our 4K test dataset from [this link]( https://www.dropbox.com/s/101g9kdobgwl8x6/test.zip?dl=0) and unzip the 'test' folder in **\<source_path\>/data/test**, then you can get an input dataset (LR LFR), a flow data, a warped data and an output dataset (HR HFR) placed in **\<source_path\>/data/test/LR_LFR**, **\<source_path\>/data/test/flow** , **\<source_path\>/data/test/warped**  and **\<source_path\>/data/test/HR_HFR**, respectively. 
```
FISR
└── data
   └── test
       ├── flow
           ├── LR_Surfing_SlamDunk_test_ss1.flo
       ├── HR_HFR
           ├── HR_vid_1_fr_07171_seq_2.png
           ├── HR_vid_1_fr_07171_seq_3.png
           └── ...
       ├── LR_LFR
           ├── LR_vid_1_fr_07171_seq_1.png 
           ├── LR_vid_1_fr_07171_seq_3.png
           └── ...
       ├── warped
           ├── LR_Surfing_SlamDunk_test_ss1_warp.mat  
```
3. Download the pre-trained weights from [this link](https://www.dropbox.com/s/hfzzddfocmmazso/FISRnet_exp1.zip?dl=0) and then unzip it to place in **\<source_path\>/checkpoint_dir/FISRnet_exp1**.
```
FISR
└── checkpoint_dir
   └── FISRnet_exp1
       ├── checkpoint
       ├── FISRnet-122000.data-00000-of-00001
       ├── FISRnet-122000.index
       ├── FISRnet-122000.meta
           
```
4. Run **main.py** with the following options in parse_args: 

**(i) For testing on our 4K test dataset input:**  

```bash
python main.py --phase 'test' --exp_num 1 --test_data_path './data/test/LR_LFR' --test_flow_data_path './data/test/flow/LR_Surfing_SlamDunk_test_ss1.flo' --test_warped_data_path './data/test/warped/LR_Surfing_SlamDunk_test_ss1_warp.mat' --test_label_path './data/test/HR_HFR'
```

**(ii) For FISR testing on a single folder, which contains a single scene (.png file input in YUV format):**  
'--phase' as **'FISR_for_video'**, ‘--exp_num' as **1**, ‘--frame_num' as **numbers of input frames you want to convert, in our example, 5**, ‘--frame_folder_path’ as **folder path that you want to apply FISRnet, in our example, 'E:/FISR_Github/FISR_test_folder/scene1'**, '--FISR_input_size' as **(1080, 1920)** (e.g. for 2K), and make sure that you have placed all necessary files (we consider 'relative paths' for a convenience) in 'FISR_tfoptflow' for computing flows and warping images automatically. Please also refer the description below on **How to make flow and warped files by using PWC-Net**.
```
FISR
└── FISR_test_folder
   └── scene1
       ├── seq_0001.png
       ├── seq_0002.png
       ├── seq_0003.png
       ├── seq_0004.png
       └── ...
           
```

Example:

```bash
python main.py --phase 'FISR_for_video' --exp_num 1 --frame_num 5 --frame_folder_path 'E:/FISR_Github/FISR_test_folder/scene1' --FISR_input_size (1080, 1920)
```

### Description
* **Running the test option** will read the three input files, which are the LR LFR input sequences as **.png** files, the flows as pre-made **.flo** file and the warped frames as pre-made **.mat** file, respectively. Then it will save the predicted HR HFR results (FISR) in .png format in **\<source_path\>/test_img_dir/FISRnet_exp1**. The 70 YUV FISR results with 10 scenes will be converted into RGB files and **the performances (PSNR & SSIM on VFI-SR & SR separately) will be measured as in the paper**. **For faster testing** (to acquire performances only), you may comment the line for saving the predicted images as .png files.
* Please note that the uploaded version of our pre-trained FISRnet weights now yields **PSNR - VFI-SR: 37.86 [dB], SR: 48.07 [dB], SSIM – VFI-SR: 0.9743, SR: 0.9921**. 
* You can check our FISR results on our 4K test dataset by downloading [this link](https://www.dropbox.com/s/dym0kolu8niaty2/FISRnet_exp1.zip?dl=0) and then place them in **\<source_path\>/test_img_dir/FISRnet_exp1**.
* **Running the FISR_for_video option** will read all **.png** files (YUV format) in the designated folder and save the predicted FISR results in .png files (both RGB&YUV format) in **\<source_path\>/FISR_test_folder/scene1/FISR_frames**. 
* (will be updated as soon as possible) If you wish to convert the produced .png files to a **.yuv video**, you could run the provided **PNGtoYUV.m** Matlab code, which would save the .yuv video in the same location as the .png files. **Please set the directory name accordingly.** (Encoding the raw .yuv video to a compressed video format such as .mp4 can be done using ffmpeg).
* **Due to GPU memory constraints**, the full 4K frame may fail to be tested at one go. The '--test_patch' option defines the number of patches (H, W) to divide the input frame (e.g. (1, 1) means the full 4K frame will be entered, (2, 2) means that it will be divided into 2x2 2K frame patches and processed serially). You may modify this variable so that the testing works with your GPU.
* **How to make flow and warped files by using PWC-Net:** you can download all the related tensorflow files from [this link](https://github.com/philferriere/tfoptflow) including weights, and then modify the folder name ‘tfoptflow’ as ‘FISR_tfoptflow’. Insert(download) all our modified versions for ‘test’ and 'FISR_for_video' phase placed in the ‘FISR_tfoptflow’ folder. For 'test', you can make .flo flow file by using ‘FISR_pwcnet_predict_from_img_test.py’ and then get .mat warped file by using ‘FISR_warp_mat_with_flo.py’. For 'FISR_for_video', you can sequentially get a '.flo' file and a '.mat' warped file for FISR testing in 'scene1'(our example), according to YUV format files in 'scene1'(our example) folder. 

## Training code
### Quick Start
1. Download the source code in a directory of your choice **\<source_path\>**.
2. Download our train dataset from [this link]( https://www.dropbox.com/s/n71hzqis6hpggcs/train.zip?dl=0) and unzip the 'train' folder in **\<source_path\>/data/train**, then you can get an input dataset (LR LFR), a two flow data (stride 1&2), a two warped data (stride 1&2) and an output dataset (HR HFR) placed in **\<source_path\>/data/train/LR_LFR**, **\<source_path\>/data/train/flow** , **\<source_path\>/data/train/warped**  and **\<source_path\>/data/train/HR_HFR**, respectively. 
 ```
FISR
└── data
   └── train
       ├── flow
           ├── LR_Surfing_SlamDunk_5seq_ss1.flo
           ├── LR_Surfing_SlamDunk_5seq_ss2.flo
       ├── HR_HFR
           ├── HR_Surfing_SlamDunk_5seq.mat
       ├── LR_LFR
           ├── LR_Surfing_SlamDunk_5seq.mat
       ├── warped
           ├── LR_Surfing_SlamDunk_5seq_ss1_warp.mat  
           ├── LR_Surfing_SlamDunk_5seq_ss2_warp.mat
```
3. Run **main.py** with the following options in parse_args:  
```bash
python main.py --phase 'train' --exp_num 7 --train_data_path './data/train/LR_LFR/LR_Surfing_SlamDunk_5seq.mat' --train_flow_data_path './data/train/flow/LR_Surfing_SlamDunk_5seq_ss1.flo' --train_flow_ss2_data_path './data/train/flow/LR_Surfing_SlamDunk_5seq_ss2.flo --train_warped_data_path './data/train/warped/LR_Surfing_SlamDunk_5seq_ss1_warp.mat' --train_warped_ss2_data_path './data/train/warped/LR_Surfing_SlamDunk_5seq_ss2_warp.mat' --train_label_path './data/train/HR_HFR/HR_Surfing_SlamDunk_5seq.mat'
```
`--exp_num` should be set accordingly.

### Description
* **Running the train option** will train FISRnet and save the trained weights in **\<source_path\>/checkpoint_dir/FISRnet_exp~**.
* The trained model can be tested with **test** or **FISR_for_video** options.
* If you wish to compare with the provided weights, **change the '--exp_num' option** before training to another number than 1 to avoid overwriting the provided pre-trained weights so that the new weights are saved in a different folder (e.g. FISRnet_exp7).
* **The training process can be monitored using Tensorboard.** The log directory for using Tensorboard is **\<source_path\>/logdir/FISRnet_exp1**. Please note that the images are viewed in YUV itself (not RGB). 
* **How to make flow and warped files by using PWC-Net:** you can download all the related tensorflow files from [this link](https://github.com/philferriere/tfoptflow) including weights, and then modify the folder name ‘tfoptflow’ as ‘FISR_tfoptflow’. Insert(download) all our modified versions for ‘train’ phase in ‘FISR_tfoptflow’ folder. You can make .flo flow file by using ‘FISR_warp_mat_with_flo.py’ and then get .mat warped file by using ‘FISR_warp_mat_with_flo.py’. 


## Contact
Please contact us via email (jhoh94@kaist.ac.kr or sooyekim@kaist.ac.kr) for any problems regarding the released code.

## License
The source code is free for research and education use only. Any commercial use should get formal permission first.

