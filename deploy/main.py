"""
AI凉茶机 — 余弦相似度分类器使用示例
====================================

演示 CosSimClassifier 的基本用法。
"""

from libs.PipeLine import PipeLine, ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
import os
import ujson
from media.media import *
from media.sensor import *
from time import *
import nncase_runtime as nn
import ulab.numpy as np
import time
import image
import aidemo
import random
import gc
import sys

# 输出中文
sys.stdout.reconfigure(encoding="utf-8")
# 优先搜索当前路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classifier import CosSimClassifier






# 自定义人脸检测任务类
class FaceDetApp(AIBase):
    def __init__(self,kmodel_path,model_input_size,anchors,confidence_threshold=0.25,nms_threshold=0.3,rgb888p_size=[1280,720],display_size=[1920,1080],debug_mode=0):
        super().__init__(kmodel_path,model_input_size,rgb888p_size,debug_mode)
        # kmodel路径
        self.kmodel_path=kmodel_path
        # 检测模型输入分辨率
        self.model_input_size=model_input_size
        # 置信度阈值
        self.confidence_threshold=confidence_threshold
        # nms阈值
        self.nms_threshold=nms_threshold
        # 检测任务锚框
        self.anchors=anchors
        # sensor给到AI的图像分辨率，宽16字节对齐
        self.rgb888p_size=[ALIGN_UP(rgb888p_size[0],16),rgb888p_size[1]]
        # 视频输出VO分辨率，宽16字节对齐
        self.display_size=[ALIGN_UP(display_size[0],16),display_size[1]]
        # debug模式
        self.debug_mode=debug_mode
        # 实例化Ai2d，用于实现模型预处理
        self.ai2d=Ai2d(debug_mode)
        # 设置Ai2d的输入输出格式和类型
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT,nn.ai2d_format.NCHW_FMT,np.uint8, np.uint8)

    # 配置预处理操作，这里使用了pad和resize，Ai2d支持crop/shift/pad/resize/affine，具体代码请打开/sdcard/app/libs/AI2D.py查看
    def config_preprocess(self,input_image_size=None):
        with ScopedTiming("set preprocess config",self.debug_mode > 0):
            # 初始化ai2d预处理配置，默认为sensor给到AI的尺寸，可以通过设置input_image_size自行修改输入尺寸
            ai2d_input_size=input_image_size if input_image_size else self.rgb888p_size
            # 设置padding预处理
            self.ai2d.pad(self.get_pad_param(), 0, [104,117,123])
            # 设置resize预处理
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            # 构建预处理流程,参数为预处理输入tensor的shape和预处理输出的tensor的shape
            self.ai2d.build([1,3,ai2d_input_size[1],ai2d_input_size[0]],[1,3,self.model_input_size[1],self.model_input_size[0]])

    # 自定义后处理，results是模型输出的array列表，这里使用了aidemo的face_det_post_process列表
    def postprocess(self,results):
        with ScopedTiming("postprocess",self.debug_mode > 0):
            res = aidemo.face_det_post_process(self.confidence_threshold,self.nms_threshold,self.model_input_size[0],self.anchors,self.rgb888p_size,results)
            if len(res)==0:
                return res
            else:
                return res[0]

    # 计算padding参数
    def get_pad_param(self):
        dst_w = self.model_input_size[0]
        dst_h = self.model_input_size[1]
        # 计算最小的缩放比例，等比例缩放
        ratio_w = dst_w / self.rgb888p_size[0]
        ratio_h = dst_h / self.rgb888p_size[1]
        if ratio_w < ratio_h:
            ratio = ratio_w
        else:
            ratio = ratio_h
        new_w = (int)(ratio * self.rgb888p_size[0])
        new_h = (int)(ratio * self.rgb888p_size[1])
        dw = (dst_w - new_w) / 2
        dh = (dst_h - new_h) / 2
        top = (int)(round(0))
        bottom = (int)(round(dh * 2 + 0.1))
        left = (int)(round(0))
        right = (int)(round(dw * 2 - 0.1))
        return [0,0,0,0,top, bottom, left, right]

# 自定义人脸关键点任务类
class FaceLandMarkApp(AIBase):
    def __init__(self,kmodel_path,model_input_size,rgb888p_size=[1920,1080],display_size=[1920,1080],debug_mode=0):
        super().__init__(kmodel_path,model_input_size,rgb888p_size,debug_mode)
        # kmodel路径
        self.kmodel_path=kmodel_path
        # 关键点模型输入分辨率
        self.model_input_size=model_input_size
        # sensor给到AI的图像分辨率，宽16字节对齐
        self.rgb888p_size=[ALIGN_UP(rgb888p_size[0],16),rgb888p_size[1]]
        # 视频输出VO分辨率，宽16字节对齐
        self.display_size=[ALIGN_UP(display_size[0],16),display_size[1]]
        # debug模式
        self.debug_mode=debug_mode
        # 目标矩阵
        self.matrix_dst=None
        self.ai2d=Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT,nn.ai2d_format.NCHW_FMT,np.uint8, np.uint8)

    # 配置预处理操作，这里使用了affine，Ai2d支持crop/shift/pad/resize/affine，具体代码请打开/sdcard/app/libs/AI2D.py查看
    def config_preprocess(self,det,input_image_size=None):
        with ScopedTiming("set preprocess config",self.debug_mode > 0):
            # 初始化ai2d预处理配置，默认为sensor给到AI的尺寸，可以通过设置input_image_size自行修改输入尺寸
            ai2d_input_size=input_image_size if input_image_size else self.rgb888p_size
            # 计算目标矩阵，并获取仿射变换矩阵
            self.matrix_dst = self.get_affine_matrix(det)
            affine_matrix = [self.matrix_dst[0][0],self.matrix_dst[0][1],self.matrix_dst[0][2],
                             self.matrix_dst[1][0],self.matrix_dst[1][1],self.matrix_dst[1][2]]
            # 设置仿射变换预处理
            self.ai2d.affine(nn.interp_method.cv2_bilinear,0, 0, 127, 1,affine_matrix)
            # 构建预处理流程,参数为预处理输入tensor的shape和预处理输出的tensor的shape
            self.ai2d.build([1,3,ai2d_input_size[1],ai2d_input_size[0]],[1,3,self.model_input_size[1],self.model_input_size[0]])

    # 自定义后处理，results是模型输出的array列表，这里使用了aidemo库的invert_affine_transform接口
    def postprocess(self,results):
        with ScopedTiming("postprocess",self.debug_mode > 0):
            pred=results[0]
            # （1）将人脸关键点输出变换模型输入
            half_input_len = self.model_input_size[0] // 2
            pred = pred.flatten()
            for i in range(len(pred)):
                pred[i] += (pred[i] + 1) * half_input_len
            # （2）获取仿射矩阵的逆矩阵
            matrix_dst_inv = aidemo.invert_affine_transform(self.matrix_dst)
            matrix_dst_inv = matrix_dst_inv.flatten()
            # （3）对每个关键点进行逆变换
            half_out_len = len(pred) // 2
            for kp_id in range(half_out_len):
                old_x = pred[kp_id * 2]
                old_y = pred[kp_id * 2 + 1]
                # 逆变换公式
                new_x = old_x * matrix_dst_inv[0] + old_y * matrix_dst_inv[1] + matrix_dst_inv[2]
                new_y = old_x * matrix_dst_inv[3] + old_y * matrix_dst_inv[4] + matrix_dst_inv[5]
                pred[kp_id * 2] = new_x
                pred[kp_id * 2 + 1] = new_y
            return pred

    def get_affine_matrix(self,bbox):
        # 获取仿射矩阵，用于将边界框映射到模型输入空间
        with ScopedTiming("get_affine_matrix", self.debug_mode > 1):
            # 从边界框提取坐标和尺寸
            x1, y1, w, h = map(lambda x: int(round(x, 0)), bbox[:4])
            # 计算缩放比例，使得边界框映射到模型输入空间的一部分
            scale_ratio = (self.model_input_size[0]) / (max(w, h) * 1.5)
            # 计算边界框中心点在模型输入空间的坐标
            cx = (x1 + w / 2) * scale_ratio
            cy = (y1 + h / 2) * scale_ratio
            # 计算模型输入空间的一半长度
            half_input_len = self.model_input_size[0] / 2
            # 创建仿射矩阵并进行设置
            matrix_dst = np.zeros((2, 3), dtype=np.float)
            matrix_dst[0, 0] = scale_ratio
            matrix_dst[0, 1] = 0
            matrix_dst[0, 2] = half_input_len - cx
            matrix_dst[1, 0] = 0
            matrix_dst[1, 1] = scale_ratio
            matrix_dst[1, 2] = half_input_len - cy
            return matrix_dst

# 人脸标志解析
class FaceLandMark:
    def __init__(self,face_det_kmodel,face_landmark_kmodel,det_input_size,landmark_input_size,anchors,confidence_threshold=0.25,nms_threshold=0.3,rgb888p_size=[1920,1080],display_size=[1920,1080],debug_mode=0):
        # 人脸检测模型路径
        self.face_det_kmodel=face_det_kmodel
        # 人脸标志解析模型路径
        self.face_landmark_kmodel=face_landmark_kmodel
        # 人脸检测模型输入分辨率
        self.det_input_size=det_input_size
        # 人脸标志解析模型输入分辨率
        self.landmark_input_size=landmark_input_size
        # anchors
        self.anchors=anchors
        # 置信度阈值
        self.confidence_threshold=confidence_threshold
        # nms阈值
        self.nms_threshold=nms_threshold
        # sensor给到AI的图像分辨率，宽16字节对齐
        self.rgb888p_size=[ALIGN_UP(rgb888p_size[0],16),rgb888p_size[1]]
        # 视频输出VO分辨率，宽16字节对齐
        self.display_size=[ALIGN_UP(display_size[0],16),display_size[1]]
        # debug_mode模式
        self.debug_mode=debug_mode

        # 人脸关键点不同部位关键点列表
        self.dict_kp_seq = [
            [43, 44, 45, 47, 46, 50, 51, 49, 48],              # left_eyebrow
            [97, 98, 99, 100, 101, 105, 104, 103, 102],        # right_eyebrow
            [35, 36, 33, 37, 39, 42, 40, 41],                  # left_eye
            [89, 90, 87, 91, 93, 96, 94, 95],                  # right_eye
            [34, 88],                                          # pupil
            [72, 73, 74, 86],                                  # bridge_nose
            [77, 78, 79, 80, 85, 84, 83],                      # wing_nose
            [52, 55, 56, 53, 59, 58, 61, 68, 67, 71, 63, 64],  # out_lip
            [65, 54, 60, 57, 69, 70, 62, 66],                  # in_lip
            [1, 9, 10, 11, 12, 13, 14, 15, 16, 2, 3, 4, 5, 6, 7, 8, 0, 24, 23, 22, 21, 20, 19, 18, 32, 31, 30, 29, 28, 27, 26, 25, 17]  # basin
        ]

        # 人脸关键点不同部位（顺序同dict_kp_seq）颜色配置，argb
        self.color_list_for_osd_kp = [
            (255, 0, 255, 0),
            (255, 0, 255, 0),
            (255, 255, 0, 255),
            (255, 255, 0, 255),
            (255, 255, 0, 0),
            (255, 255, 170, 0),
            (255, 255, 255, 0),
            (255, 0, 255, 255),
            (255, 255, 220, 50),
            (255, 30, 30, 255)
        ]
        # 人脸检测实例
        self.face_det=FaceDetApp(self.face_det_kmodel,model_input_size=self.det_input_size,anchors=self.anchors,confidence_threshold=self.confidence_threshold,nms_threshold=self.nms_threshold,rgb888p_size=self.rgb888p_size,display_size=self.display_size,debug_mode=0)
        # 人脸标志解析实例
        self.face_landmark=FaceLandMarkApp(self.face_landmark_kmodel,model_input_size=self.landmark_input_size,rgb888p_size=self.rgb888p_size,display_size=self.display_size)
        # 配置人脸检测的预处理
        self.face_det.config_preprocess()

    # run函数
    def run(self,input_np):
        # 执行人脸检测
        det_boxes=self.face_det.run(input_np)
        landmark_res=[]
        for det_box in det_boxes:
            # 对每一个检测到的人脸解析关键部位
            self.face_landmark.config_preprocess(det_box)
            res=self.face_landmark.run(input_np)
            landmark_res.append(res)
        return det_boxes,landmark_res

    def extract_color_mouth(self, landmark_res, img):
        x_min = float('inf')
        x_max = float('-inf')
        y_min = float('inf')
        y_max = float('-inf')

        mouth_kp_indices = self.dict_kp_seq[7]  #嘴唇

        # 获取所有嘴唇关键点的坐标
        for i in mouth_kp_indices:
            x = landmark_res[i * 2]      # x坐标
            y = landmark_res[i * 2 + 1]  # y坐标
            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)

        # 确保边界有效
        if x_min >= x_max or y_min >= y_max:
            return None

        y_dis = y_max - y_min
        y_max = y_max - y_dis/2
        # 提取嘴唇区域 [C, H, W] -> 取y方向区域再取x方向区域
        mouth_region = img[:, int(y_min):int(y_max), int(x_min):int(x_max)]

        # 计算RGB各通道均值
        # 分别计算每个通道的均值
        r_mean = np.mean(mouth_region[0])  # R通道
        g_mean = np.mean(mouth_region[1])  # G通道
        b_mean = np.mean(mouth_region[2])  # B通道

        mouth_color = np.array([r_mean, g_mean, b_mean])

        return mouth_color

    def extract_color_left_eye(self, landmark_res, img):
        x_min = float('inf')
        x_max = float('-inf')
        y_min = float('inf')
        y_max = float('-inf')

        left_eye_kp_indices = [38,39,41,42]  #左眼

        # 获取所有左眼关键点的坐标
        for i in left_eye_kp_indices:
            x = landmark_res[i * 2]      # x坐标
            y = landmark_res[i * 2 + 1]  # y坐标
            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)

        # 确保边界有效
        if x_min >= x_max or y_min >= y_max:
            return None

        # 提取嘴唇区域 [C, H, W] -> 取y方向区域再取x方向区域
        left_eye_region = img[:, int(y_min):int(y_max), int(x_min):int(x_max)]

        # 计算RGB各通道均值
        # 分别计算每个通道的均值
        r_mean = np.mean(left_eye_region[0])  # R通道
        g_mean = np.mean(left_eye_region[1])  # G通道
        b_mean = np.mean(left_eye_region[2])  # B通道

        left_eye_color = np.array([r_mean, g_mean, b_mean])

        return left_eye_color

    def extract_color_right_eye(self, landmark_res, img):
        x_min = float('inf')
        x_max = float('-inf')
        y_min = float('inf')
        y_max = float('-inf')

        right_eye_kp_indices = [90,91,93,94]  #右眼

        # 获取所有左眼关键点的坐标
        for i in right_eye_kp_indices:
            x = landmark_res[i * 2]      # x坐标
            y = landmark_res[i * 2 + 1]  # y坐标
            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)

        # 确保边界有效
        if x_min >= x_max or y_min >= y_max:
            return None

        # 提取嘴唇区域 [C, H, W] -> 取y方向区域再取x方向区域
        right_eye_region = img[:, int(y_min):int(y_max), int(x_min):int(x_max)]

        # 计算RGB各通道均值
        # 分别计算每个通道的均值
        r_mean = np.mean(right_eye_region[0])  # R通道
        g_mean = np.mean(right_eye_region[1])  # G通道
        b_mean = np.mean(right_eye_region[2])  # B通道

        right_eye_color = np.array([r_mean, g_mean, b_mean])

        return right_eye_color

    def extract_color_face(self, landmark_res, img):
        x_min = float('inf')
        x_max = float('-inf')
        y_min = float('inf')
        y_max = float('-inf')

        face_kp_indices = self.dict_kp_seq[9]  #脸颊

        # 获取所有左眼关键点的坐标
        for i in face_kp_indices:
            x = landmark_res[i * 2]      # x坐标
            y = landmark_res[i * 2 + 1]  # y坐标
            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)

        # 确保边界有效
        if x_min >= x_max or y_min >= y_max:
            return None

        # 提取嘴唇区域 [C, H, W] -> 取y方向区域再取x方向区域
        face_region = img[:, int(y_min):int(y_max), int(x_min):int(x_max)]

        # 计算RGB各通道均值
        # 分别计算每个通道的均值
        r_mean = np.mean(face_region[0])  # R通道
        g_mean = np.mean(face_region[1])  # G通道
        b_mean = np.mean(face_region[2])  # B通道

        face_color = np.array([r_mean, g_mean, b_mean])

        return face_color

    def rgb_to_hsl(self,r, g, b):
        """将R,G,B(0~255)转换为H(0~360), S(0~100), L(0~100)"""
        r_norm = r / 255.0
        g_norm = g / 255.0
        b_norm = b / 255.0

        max_val = max(r_norm, g_norm, b_norm)
        min_val = min(r_norm, g_norm, b_norm)
        delta = max_val - min_val

        # 亮度 L
        l = (max_val + min_val) / 2.0

        # 饱和度 S (当max=min时饱和度为0)
        if delta == 0:
            s = 0.0
            h = 0.0
        else:
            if l <= 0.5:
                s = delta / (max_val + min_val)
            else:
                s = delta / (2.0 - max_val - min_val)

            # 色相 H
            if max_val == r_norm:
                h = (g_norm - b_norm) / delta
                if g_norm < b_norm:
                    h += 6.0
            elif max_val == g_norm:
                h = (b_norm - r_norm) / delta + 2.0
            else:  # max == b_norm
                h = (r_norm - g_norm) / delta + 4.0

            h /= 6.0  # 归一化到0~1

        return h * 360.0, s * 100.0, l * 100.0




def main():

    # 显示模式，默认"lcd"
    display_mode="lcd"
    display_size=[640,480]
    # 人脸检测模型路径
    face_det_kmodel_path="/sdcard/examples/kmodel/face_detection_320.kmodel"
    # 人脸关键标志模型路径
    face_landmark_kmodel_path="/sdcard/examples/kmodel/face_landmark.kmodel"
    # 其它参数
    anchors_path="/sdcard/examples/utils/prior_data_320.bin"
    rgb888p_size=[1280,960]
    face_det_input_size=[320,320]
    face_landmark_input_size=[192,192]
    confidence_threshold=0.5
    nms_threshold=0.2
    anchor_len=4200
    det_dim=4
    anchors = np.fromfile(anchors_path, dtype=np.float)
    anchors = anchors.reshape((anchor_len,det_dim))

    # 初始化PipeLine，只关注传给AI的图像分辨率，显示的分辨率
    sensor = Sensor(width=1280, height=960) # 构建摄像头对象
    pl = PipeLine(rgb888p_size=rgb888p_size, display_size=display_size, display_mode=display_mode)
    pl.create(sensor=sensor)  # 创建PipeLine实例
    flm=FaceLandMark(face_det_kmodel_path,face_landmark_kmodel_path,det_input_size=face_det_input_size,landmark_input_size=face_landmark_input_size,anchors=anchors,confidence_threshold=confidence_threshold,nms_threshold=nms_threshold,rgb888p_size=rgb888p_size,display_size=display_size)
    try:
        while True:
            os.exitpoint()
            with ScopedTiming("total",1):
                #img是CHW格式，颜色模式是RGB
                img=pl.get_frame()                          # 获取当前帧
                det_boxes,landmark_res=flm.run(img)         # 推理当前帧
                key_parts_dict = [0,0,0]
                '''
                0是舌苔，1是眼睛，2是脸色
                mouth：0表示正常舌苔，1表示不正常舌苔
                left_eye:0表示没黑眼圈，1表示有
                face：0表示面色正常，1表示不正常
                '''
                if landmark_res:
                    mouth_color = flm.extract_color_mouth(landmark_res[0],img)
                    left_eye_color = flm.extract_color_left_eye(landmark_res[0],img)
                    right_eye_color = flm.extract_color_right_eye(landmark_res[0],img)
                    face_color = flm.extract_color_face(landmark_res[0],img)
                    if mouth_color is not None:
                        hue_mouth, sat_mouth, lum_mouth = flm.rgb_to_hsl(mouth_color[0], mouth_color[1], mouth_color[2])
                        if (hue_mouth<=40 or 320 <= hue_mouth <= 360) and sat_mouth <= 50 and lum_mouth <= 50:
                            key_parts_dict[0] = 0
                        else:
                            key_parts_dict[0] = 1


                    if left_eye_color is not None:
                        hue_eye, sat_eye, lum_eye = flm.rgb_to_hsl(left_eye_color[0], left_eye_color[1], left_eye_color[2])
                        if 250<=hue_eye<=300 and lum_eye <= 30 and sat_eye>=10:
                            key_parts_dict[1]=1
                        else:
                            key_parts_dict[1]=0

                    if face_color is not None:
                        hue_face, sat_face, lum_face = flm.rgb_to_hsl(face_color[0], face_color[1], face_color[2])
                        if (0 <= hue_face <= 30 or 330 <= hue_face <= 360) and 15 <= sat_face <= 50 and 20 <= lum_face <= 70:
                            key_parts_dict[2]=0
                        else:
                            key_parts_dict[2]=1

                    break

                gc.collect()
    except Exception as e:
        sys.print_exception(e)
    finally:
        flm.face_det.deinit()
        flm.face_landmark.deinit()
        pl.destroy()

    clf = CosSimClassifier()

    result = clf.predict(answers, key_parts_dict)

    # 查看与所有体质的相似度
    detail = clf.predict_with_scores(answers, key_parts_dict)
    print(f"\n各体质余弦相似度:")
    for code, score in sorted(detail["scores"].items(), key=lambda x: -x[1]):
        bar = "#" * int(score * 20) + "-" * (20 - int(score * 20))
        print(f"  {code}: {score:.4f}  {bar}")


if __name__ == "__main__":
    main()
