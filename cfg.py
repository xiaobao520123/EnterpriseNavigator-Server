import os
# Server Configuartions
# Text Detection Configuration
# 探测文字时图片压缩尺寸（256、384、512、640、736等）
max_predict_img_size = 512

# AdvancedEast卷积网络参数
shrink_ratio = 0.2
shrink_side_ratio = 0.6
epsilon = 1e-4

num_channels = 3
feature_layers_range = range(5, 1, -1)
feature_layers_num = len(feature_layers_range)
pixel_size = 2 ** feature_layers_range[-1]
locked_layers = False

# AdvancedEast探测阈值参数
# 像素阈值
pixel_threshold = 0.9
# 边缘顶点像素阈值
side_vertex_pixel_threshold = 0.9
# 多边形头尾顶点阈值
trunc_threshold = 0.5

# Text Recognition Configuration
# AttentionOCR识别时图片尺寸（注：需与模型训练时图片大小对应）
ocr_recognition_img_size = 256
# 无句子末尾标志（EOS）的最长短语长度（注：+1腾出一位给EOS；需与模型训练时的seq_len对应）
seq_len = 32+1

# Text Result Processing Configuration
# 企业实体名称长度阈值
enterprise_len_threshold = 15
# 图像多边形顶点分值权重
polygon_vertex_power = 0.7
# 图像多边形面积分值权重
polygon_area_power = 0.5

# Other Configuration
# Tensorflow可见GPU单位（从序号0开始）
gpus = [0]
visible_devices = len(gpus)
visible_devices_str = ','.join([str(i) for i in gpus])

# 允许识别的文件类型
allowed_extenstions = set(['bmp', 'png', 'jpg', 'jpeg'])
# 上传的文件保存位置
upload_folder = './uploads'

# 模型、字典文件路径
# AdvancedEast模型权重文件路径
detection_model_weights = './models/AdvancedEAST/east_model_weights_3T512.h5'
# AttentionOCR模型文件路径
recognition_model_file = './models/AttentionOCR/text_recognition_qcc.pb'
# 标签字典文件路径
label_dict_file = './dict/icdar_labels.txt'

# huggingface model settings
east_model_url = "https://huggingface.co/xiaobao520123/east_model/resolve/main/east_model_weights_3T512.h5"
attention_ocr_model_url = "https://huggingface.co/xiaobao520123/attention_ocr_text_qcc/resolve/main/text_recognition_qcc.pb"

# 服务器设置
server_host="0.0.0.0"
server_port=6660

# Client Configuration
# 识别结果表格保存目录
table_output_folder = './static/output'
# 识别预览图图片保存目录
image_preview_folder = './static/img/preview'
# 识别结果打框线条宽度
answer_line_width = 5
# 识别结果打框线条颜色
answer_line_color = 'green'

# API Configuration
allowed_extenstions = set(['bmp', 'png', 'jpg', 'jpeg'])

'''
    if you implement this app on a remote server, replace the address with that of your server
'''
api_qcc_batch_detection = r"http://localhost:6660/api/batch"