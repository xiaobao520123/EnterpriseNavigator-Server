import numpy as np
import cv2

import cfg

from east.text_detection import TextDetection
from ocr.text_recognition import TextRecognition
from eval.text_process import TextProcess

# 载入所需要的模型和字典
def init_model():
    detection_h5 = cfg.detection_model_weights
    recognition_pb = cfg.recognition_model_file
    label_txt = cfg.label_dict_file

    east_detect = TextDetection(detection_h5)
    recognition = TextRecognition(recognition_pb, label_txt, seq_len=cfg.seq_len)

    text_process = TextProcess()
    return east_detect, recognition, text_process

east_detect_model, recognition_model, text_process = init_model()

# 根据顶点获得文字框的盒子
def box_with_points(points, h, w):
    vertex_row_coords = [point[1] for point in points]  # y
    vertex_col_coords = [point[0] for point in points]

    bbox = [np.amin(vertex_row_coords), np.amin(vertex_col_coords), np.amax(vertex_row_coords),
            np.amax(vertex_col_coords)]
    bbox = list(map(int, bbox))
    return bbox

# 将图像修正至OCR模型对应的图像尺寸格式，生成图像填充图
def generate_padded_image(cropped_image, test_size=cfg.ocr_recognition_img_size):
    height, width = cropped_image.shape[:2]
    if height >= width:
        scale = test_size / height
        resized_image = cv2.resize(cropped_image, (0, 0), fx=scale, fy=scale)
        left_bordersize = (test_size - resized_image.shape[1]) // 2
        right_bordersize = test_size - resized_image.shape[1] - left_bordersize
        image_padded = cv2.copyMakeBorder(resized_image, top=0, bottom=0, left=left_bordersize,
                                            right=right_bordersize, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    else:
        scale = test_size / width
        resized_image = cv2.resize(cropped_image, (0, 0), fx=scale, fy=scale)
        top_bordersize = (test_size - resized_image.shape[0]) // 2
        bottom_bordersize = test_size - resized_image.shape[0] - top_bordersize
        image_padded = cv2.copyMakeBorder(resized_image, top=top_bordersize, bottom=bottom_bordersize, left=0,
                                            right=0, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    image_padded = np.float32(image_padded) / 255.

    return np.expand_dims(image_padded, 0)

# 根据多边形顶点坐标抠图，得到填充图像列表
def crop_text_image_by_polygons(rgb_image, polygons):
    image_padded_list = []
    for polygon in polygons:
        # 从得到的文字框顶点信息，回到原图剪裁文字，将文字图像抠出
        bbox = box_with_points(polygon, rgb_image.shape[0], rgb_image.shape[1])
        masked_image = np.uint8(rgb_image)
        cropped_image = masked_image[max(0, bbox[0]):min(bbox[2], masked_image.shape[0]),
                        max(0, bbox[1]):min(bbox[3], masked_image.shape[1]), :]

        # 生成剪裁文字图像填充信息
        image_padded = generate_padded_image(cropped_image)
        image_padded_list.append(image_padded)
    return image_padded_list

# 文字探测与文字识别
def detection(img_path):
    # 从文件路径中加载图像文件
    bgr_image = cv2.imread(img_path)
    # OpenCV因为历史原因只支持RGB序列色彩
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # 使用AdvancedEast算法得到文字框和文字框的得分
    polygons, polygon_scores = east_detect_model.predict(rgb_image, img_path)

    # 得到多边形顶点数据后，将文字部分图像抠下来，生成填充图像列表
    image_padded_list = crop_text_image_by_polygons(rgb_image, polygons)
    if len(image_padded_list) == 0: # 没有在图片上探测到文字
        return None
    # 将填充图像列表拼接成numpy数组（适配多Batch，一次性将图上所有抠下来到文字小图送进tensor）
    image_padded_concat = np.concatenate(image_padded_list, axis=0)

    # 使用AttentionOCR算法从剪裁的文字图片中识别出文字，得到文字结果和文字每一个字符对应的概率
    ocr_results, probs = recognition_model.predict(image_padded_concat)

    # 初始化返回值
    detection_text = []

    for polygon, polygon_score, char_list, char_prob_vector in zip(polygons, polygon_scores, ocr_results, probs):
        # 将识别出的字符连成一个短语/句子
        line = ''.join(char_list)
        char_prob_list = char_prob_vector.tolist()

        polygon_point_list = np.reshape(polygon, (4, 2)).tolist()

        detection_text.append({
            "words": line,
            "words_char_score": char_prob_list,
            "polygon": polygon_point_list,
            "polygon_score": polygon_score
        })

    return detection_text

# 从图像文件中识别出企业实体名称
def find_enterprise_from_image(img_path):
    # 获得图像中所有文字信息
    detection_text = detection(img_path)
    if not detection_text or len(detection_text) == 0:
        return None, None
    # 从文字信息中过滤出企业实体
    enterpise, polygon = text_process.find_enterprise(detection_text)
    return enterpise, polygon
    