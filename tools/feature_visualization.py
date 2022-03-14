# coding:utf-8
# @Author     : HT
# @Time       : 2022/2/26 15:03
# @File       : feature_visualization.py
# @Software   : PyCharm

import cv2
import mmcv
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import math

def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:,0,:,:]*0
    heatmaps = []
    for c in range(feature_map.shape[1]):
        heatmap+=feature_map[:,c,:,:]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmaps.append(heatmap)

    return heatmaps

def featuremap_process(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    feature_map = feature_map.cpu().numpy()

    return feature_map
def draw_feature_map(features,img=None,save_dir = 'feature_map',name = None):
    i=False
    if i:
        for heat_maps in features:
            heat_maps=heat_maps.unsqueeze(0)
            heatmaps = featuremap_2_heatmap(heat_maps)
            # 这里的h,w指的是你想要把特征图resize成多大的尺寸
            # heatmap = cv2.resize(heatmap, (h, w))
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)
                # 下面这行将热力图转换为RGB格式 ，如果注释掉就是灰度图
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap
                plt.imshow(superimposed_img,cmap='gray')
                plt.show()
    else:
        img = img.detach()
        img = img.cpu().numpy()
        img = img[0]
        img = np.transpose(img, (1, 2, 0))
        for featuremap in features:
            heatmaps = featuremap_2_heatmap(featuremap)
            # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
            for heatmap in heatmaps:
                heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)# 将热力图的大小调整为与原始图像相同
                # superimposed_img = heatmap * 0.5 + img * 0.3
                superimposed_img = heatmap * 1 + img * 0
                # superimposed_img = np.concatenate([heatmap * 0.5,img*0.3])
                # superimposed_img = heatmap
                plt.imshow(superimposed_img, cmap='gray')
                plt.show()
                # 下面这些是对特征图进行保存，使用时取消注释
                # cv2.imshow("1",superimposed_img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # cv2.imwrite(os.path.join(save_dir,name +str(i)+'.png'), superimposed_img)
                # i=i+1

        # img = img.detach()
        # img = img.cpu().numpy()
        # img = img[0]
        # img=np.transpose(img,(1,2,0))
        # for featuremap in features:
        #     heatmaps = featuremap_2_heatmap(featuremap)
        #     # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
        #     for heatmap in heatmaps:
        #         heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
        #         heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        #         heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
        #         # superimposed_img = heatmap * 0.5 + img * 0.3
        #         superimposed_img = heatmap * 0.2 + img * 0.7
        #         # superimposed_img = np.concatenate([heatmap * 0.5,img*0.3])
        #         # superimposed_img = heatmap
        #         plt.imshow(superimposed_img,cmap='gray')
        #         plt.show()
        #         # 下面这些是对特征图进行保存，使用时取消注释
        #         # cv2.imshow("1",superimposed_img)
        #         # cv2.waitKey(0)
        #         # cv2.destroyAllWindows()
        #         # cv2.imwrite(os.path.join(save_dir,name +str(i)+'.png'), superimposed_img)
        #         # i=i+1
def draw_feature_map_channel(features,save_dir = 'feature_map',name = None):

    if isinstance(features,torch.Tensor):
        for onebat_features in features:
            features_channel=onebat_features.shape[0]
            features_channel_sqrt=math.ceil(features_channel ** 0.5)
            for i,onebat_onechan_features in enumerate(onebat_features):
                onebat_onechan_features=featuremap_process(onebat_onechan_features)
                onebat_onechan_features = np.uint8(255 * onebat_onechan_features)
                plt.subplot(features_channel_sqrt, features_channel_sqrt, i+1)
                # plt.title(str(i))
                plt.axis('off')
                plt.imshow(onebat_onechan_features,cmap='gray')

                # plt.tight_layout()  # 调整整体空白
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                                wspace=None, hspace=None)  # 调整子图间距
            # plt.figure(figsize=(features_channel_sqrt*10, features_channel_sqrt*10))
            root_path=r'G:\mmdetection_miner\feature_map'
            save_path=os.path.join(root_path,save_dir,'feature_map.jpg')
            plt.savefig(save_path)
            plt.show()
            # ————————————————
            # 版权声明：本文为CSDN博主「opencv_fjc」的原创文章，遵循CC
            # 4.0
            # BY - SA版权协议，转载请附上原文出处链接及本声明。
            # 原文链接：https: // blog.csdn.net / opencv_fjc / article / details / 109156375
            # plt.imshow(onebat_onechan_features,cmap='gray')
            # plt.show()
    else:
        print('ht_error')
