import cfg

import numpy as np
import cv2
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.backend.tensorflow_backend import set_session
from PIL import Image, ImageDraw

from .nms import nms
from .network import East

graph = tf.get_default_graph()  

# 机器学习常用到的激活函数，注意exp存在溢出问题（已采用安全算法替代）
def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return .5 * (1 + np.tanh(.5 * x))

# OpenCV2用压缩图像，获取压缩后的图像大小
def cv2_resize_image(im, max_img_size=cfg.max_predict_img_size):
    height, width = im.shape[:2]
    im_width = np.minimum(width, max_img_size)
    if im_width == max_img_size < width:
        im_height = int((im_width / width) * height)
    else:
        im_height = height
    o_height = np.minimum(im_height, max_img_size)
    if o_height == max_img_size < im_height:
        o_width = int((o_height / im_height) * im_width)
    else:
        o_width = im_width
    d_wight = o_width - (o_width % 32)
    d_height = o_height - (o_height % 32)
    return d_wight, d_height

class TextDetection(object) :
    def __init__(self, h5file):
        # 配置Tensorflow GPU设置
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True # 允许动态增长显存使用
        set_session(tf.Session(config=config))

        self.east = East()
        self.east_detect = self.east.east_network()
        self.east_detect.load_weights(h5file)

    # 绘制预测结果的特征点到act文件
    def draw_predict_act(self, img_path, activation_pixels, y, d_wight, d_height):
        with Image.open(img_path) as im:
            im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
            draw = ImageDraw.Draw(im)
            for i, j in zip(activation_pixels[0], activation_pixels[1]):
                px = (j + 0.5) * cfg.pixel_size
                py = (i + 0.5) * cfg.pixel_size
                line_width, line_color = 1, 'red'
                if y[i, j, 1] >= cfg.side_vertex_pixel_threshold:
                    if y[i, j, 2] < cfg.trunc_threshold:
                        line_width, line_color = 2, 'yellow'
                    elif y[i, j, 2] >= 1 - cfg.trunc_threshold:
                        line_width, line_color = 2, 'green'
                draw.line([(px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                        (px + 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                        (px + 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                        (px - 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                        (px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size)],
                        width=line_width, fill=line_color)
        im.save(img_path + '_act.jpg')

    def predict(self, rgb_image, img_path, pixel_threshold=cfg.pixel_threshold, quiet=True):
        height, width = rgb_image.shape[:2]

        d_wight, d_height = cv2_resize_image(rgb_image, cfg.max_predict_img_size)

        scale_ratio_w = d_wight / width
        scale_ratio_h = d_height / height

        rgb_image = cv2.resize(rgb_image, (0, 0), fx=scale_ratio_w, fy=scale_ratio_h)

        # 将图片转换到数组信息
        img = image.img_to_array(rgb_image)
        img = preprocess_input(img, mode='tf')
        x = np.expand_dims(img, axis=0)

        # 建立新Tensorflow会话，执行模型预测过程
        global graph
        with graph.as_default():
            y = self.east_detect.predict(x)

        y = np.squeeze(y, axis=0)
        y[:, :, :3] = sigmoid(y[:, :, :3])
        cond = np.greater_equal(y[:, :, 0], pixel_threshold)
        activation_pixels = np.where(cond)
        pre_scores, pre_polygons = nms(y, activation_pixels)

        #self.draw_predict_act(img_path, activation_pixels, y, d_wight, d_height)

        # 以上是模型预测过程，输出得到探测的多边形顶点得分和图像多边形顶点坐标（相对坐标）
        scores = []
        polygons = []
        for score, geo in zip(pre_scores, pre_polygons):
            if np.amin(score) > 0:
                scores.append(score)
                # 对多边形顶点坐标进行修正，还原到绝对坐标
                polygon = geo / [scale_ratio_w, scale_ratio_h]
                polygons.append(polygon)    
        return polygons, scores