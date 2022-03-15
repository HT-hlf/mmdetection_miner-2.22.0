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

def creat_dir(out_dir):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:,0,:,:]*0
    for c in range(feature_map.shape[1]):
        heatmap+=feature_map[:,c,:,:]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap

def featuremap_2_heatmap_ht(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:,0,:,:]*0
    for c in range(feature_map.shape[1]):
        heatmap+=feature_map[:,c,:,:]
    heatmap = heatmap.cpu().numpy()
    # heatmap = np.mean(heatmap, axis=0)
    heatmap_max = np.max(heatmap)
    heatmap_min = np.min(heatmap)

    # heatmap = np.maximum(heatmap, 0)
    heatmap = (heatmap-heatmap_min) /(heatmap_max-heatmap_min)

    return heatmap
def featuremap_process(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    feature_map = feature_map.cpu().numpy()

    return feature_map
def draw_feature_map(features,img=None,save_dir = 'feature_map',name = None):
    # img = mmcv.imread(img_path)
    img = img.detach()
    img = img.cpu().numpy()
    img = img[0]
    img = np.transpose(img, (1, 2, 0))
    img=np.uint8(255 * img)
    heatmaps=[]
    for featuremap in features:
        heatmap = featuremap_2_heatmap_ht(featuremap)
        heatmap = cv2.resize(heatmap[0], (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
        heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
        heatmaps.append(np.expand_dims(heatmap,2))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        # superimposed_img = heatmap * 0.4 + img * 0.5
        # superimposed_img=np.uint8(superimposed_img)
        # superimposed_img = heatmap * 1 + img * 0
        # superimposed_img = np.concatenate([heatmap * 0.5,img*0.3])
        # superimposed_img = heatmap

        superimposed_img=cv2.add(img, heatmap)

        # plt.imshow(superimposed_img, cmap='gray')
        # plt.show()
        # plt.imshow(img, cmap='gray')
        # plt.show()
        #
        plt.subplot(2, 2, 1)
        # plt.title(str(i))
        plt.axis('off')
        # plt.imshow(img, cmap='gray')
        plt.imshow(img)
        plt.subplot(2, 2, 2)
        # plt.title(str(i))
        plt.axis('off')
        # plt.imshow(superimposed_img, cmap='gray')
        plt.imshow(superimposed_img)
        plt.subplot(2, 2, 3)
        # plt.title(str(i))
        plt.axis('off')
        # plt.imshow(img, cmap='gray')
        plt.imshow(heatmap)
        plt.show()

        # 下面这些是对特征图进行保存，使用时取消注释
        # cv2.imshow("1",superimposed_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite(os.path.join(save_dir,name +str(i)+'.png'), superimposed_img)
        # i=i+1
    # heatmaps=np.concatenate(heatmaps,axis=2)
    # superimposed_img = cv2.add(img, heatmaps)
    # plt.subplot(2, 2, 1)
    # # plt.title(str(i))
    # plt.axis('off')
    # # plt.imshow(img, cmap='gray')
    # plt.imshow(img)
    # plt.subplot(2, 2, 2)
    # # plt.title(str(i))
    # plt.axis('off')
    # # plt.imshow(superimposed_img, cmap='gray')
    # plt.imshow(superimposed_img)
    # plt.subplot(2, 2, 3)
    # # plt.title(str(i))
    # plt.axis('off')
    # # plt.imshow(img, cmap='gray')
    # plt.imshow(heatmaps)
    # plt.show()

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
def draw_feature_map_channel(features,save_image_path,save_dir = 'conv_ht',name = None):

    if isinstance(features,torch.Tensor):
        for onebat_features in features:
            features_channel=onebat_features.shape[0]
            features_channel_sqrt=math.ceil(features_channel ** 0.5)

            save_path = os.path.join(save_image_path, save_dir)
            creat_dir(save_path)

            for i,onebat_onechan_features in enumerate(onebat_features):
                onebat_onechan_features=featuremap_process(onebat_onechan_features)
                onebat_onechan_features = np.uint8(255 * onebat_onechan_features)
                plt.subplot(features_channel_sqrt, features_channel_sqrt, i+1)
                # plt.title(str(i))
                plt.axis('off')
                plt.imshow(onebat_onechan_features,cmap='gray')
                save_path_ele = save_path + '/' + str(i)+'.jpg'
                mmcv.imwrite(onebat_onechan_features,save_path_ele)

                # plt.tight_layout()  # 调整整体空白
            plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                                wspace=None, hspace=None)  # 调整子图间距
            # plt.figure(figsize=(features_channel_sqrt*10, features_channel_sqrt*10))
            save_path_sum=save_path+'.jpg'
            plt.savefig(save_path_sum)
            # plt.show()
            # ————————————————
            # 版权声明：本文为CSDN博主「opencv_fjc」的原创文章，遵循CC
            # 4.0
            # BY - SA版权协议，转载请附上原文出处链接及本声明。
            # 原文链接：https: // blog.csdn.net / opencv_fjc / article / details / 109156375
            # plt.imshow(onebat_onechan_features,cmap='gray')
            # plt.show()
    else:
        print('ht_error')
