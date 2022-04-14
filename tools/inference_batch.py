# coding:utf-8
# @Author     : HT
# @Time       : 2022/3/13 16:20
# @File       : inference.py
# @Software   : PyCharm

# encoding:utf-8
from mmdet.apis import init_detector, inference_detector
import os
import mmcv
if __name__ == '__main__':
    config_file = 'configs/yolo/yolov3_d53_mstrain-416_273e_coco.py'
    # config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = '../../mmdetection_miner_result\work_dirs\epoch_264.pth'
    # checkpoint_file = 'G:\mmdetection_miner\check_file/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth'
    # 根据配置文件和 checkpoint 文件构建模型
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    # 测试单张图片并展示结果
    path=r'G:\roadway_collect_dataset\recordData_process\RGBD_r_16\rgb'
    save_path=r'G:\mmdetection_miner\vritual_image'
    for filename in os.listdir(path):
        img=path+'/'+filename
        save_img = save_path + '/' + filename
        # img = r'..\data\ht_cumt_rgbd\test2014\RGBD_bk_5_237.jpg'  # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
        result = inference_detector(model, img)
        # 在一个新的窗口中将结果可视化
        # model.show_result(img, result)
        # 或者将可视化结果保存为图片
        model.show_result(img, result, out_file=save_img)

    # img='demo/demo.jpg'
    # result = inference_detector(model, img)