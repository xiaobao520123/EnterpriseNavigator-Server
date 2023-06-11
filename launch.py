import os
from PIL import Image

import cfg

# 设置CUDA可见的GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.visible_devices_str

UPLOAD_FOLDER = cfg.upload_folder
ALLOWED_EXTENSIONS = cfg.allowed_extenstions

from flask import Flask, jsonify, flash
from flask import request, render_template
from flask_bootstrap import Bootstrap
from flask_cors import CORS
from werkzeug.utils import secure_filename

import util

def prepare_environment():
    if os.path.exists(cfg.table_output_folder) == False:
        os.makedirs(cfg.table_output_folder)
    if os.path.exists(cfg.image_preview_folder) == False:
        os.makedirs(cfg.image_preview_folder)
    if os.path.exists(cfg.upload_folder) == False:
        os.makedirs(cfg.upload_folder)
    if os.path.exists(cfg.detection_model_weights) == False:
        util.download_url(cfg.east_model_url, cfg.detection_model_weights)
    if os.path.exists(cfg.recognition_model_file) == False:
        util.download_url(cfg.attention_ocr_model_url, cfg.recognition_model_file) 

prepare_environment()

import text


app = Flask(__name__)
bootstrap = Bootstrap(app)
CORS(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SECRET_KEY'] = os.urandom(24)

# 检查是不是允许上传的文件类型
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 对上传的图片文件进行预处理
def preprocess_upload_image(image):
    filename = image.filename
    # 转换文件名到安全文件名
    filename = secure_filename(filename)
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # 保存图片文件
    image.save(img_path)
    
    # 转换颜色空间到RGB空间（去除alpha通道）
    img = Image.open(img_path).convert('RGB')
    img.save(img_path)
    return img_path

# 处理上传的文件，从中识别企业实体
def detect_enterprise_from_upload_file(file):
    img_path = preprocess_upload_image(file)
    enterprise, polygon = text.find_enterprise_from_image(img_path)
    # 如果企业实体识别失败，返回0，成功返回1
    success = int(enterprise is not None)

    result = {'success': success,
              'enterprise': enterprise,
              'polygon': polygon
              }
    return result

# 生成错误返回
def generate_err_msg(err_msg=''):
    return jsonify({
                'success': 0,
                'err_msg': err_msg
            })       

# 客户端主页
@app.route('/', methods=['GET'])
def qcc_index():
    return render_template("index.html")

# 客户端批次上传请求处理
@app.route('/upload', methods=['POST'])
def qcc_upload():
    if request.method == 'POST':
        # Flask的一个BUG，必须先转发POST请求才能获取表单参数
        r, delay = util.batch_upload(request.headers, request.get_data())

        if 'file_selector' not in request.files:
            return 'invalid arguments'
        
        if r.status_code != 200:
            return 'Request failed, error code: %d' % r.status_code

        json_data = r.json()
        
        # 转存图片文件
        file_list = request.files.getlist('file_selector')
        util.save_uploaded_file(file_list)
        
        # 处理服务器请求结果
        collections = util.process_detection_result(json_data)
        # 将识别结果写入至表格文件中
        excel_filepath = util.write_batch_detection_result(collections)

        # 计算识别速度和响应时间
        average_speed = round(len(collections) / delay, 2)
        delay = round(delay, 4)

        return render_template("view.html", 
            collections=collections, 
            delay=delay, 
            average_speed=average_speed, 
            excel_file=excel_filepath)
    return 'Access denied!'

# API——批量上传文件
@app.route('/api/batch', methods=['POST'])
def handle_api_batch():
    if request.method == 'POST':
        result = {}
        if 'file_selector' not in request.files:
            return 'arguments error, need <file_selector>!'
        files = request.files.getlist('file_selector')
        for file in files:
            if not file or file.filename == '':
                flash('invalid file')
                continue

            filename = file.filename
            # 检查是不是允许的文件类型
            if allowed_file(filename):
                result[filename] = detect_enterprise_from_upload_file(file)
            else:
                flash('file type not allowed')
        return jsonify(result)
    return 'Access denied!' 

# API——单个图片文件（适用于APP端）
@app.route('/api/qcc', methods=['POST'])
def handle_app_qcc():
    if request.method == 'POST':
        if 'data' not in request.files:
            return generate_err_msg('need argument <data>')
        file = request.files['data']
        
        if not file or file.filename == '':
            return generate_err_msg('Invalid file')

        filename = file.filename
        # 检查是不是允许的文件类型
        if allowed_file(filename):
            raw_data = detect_enterprise_from_upload_file(file)
            return jsonify(raw_data)
        else:
            return generate_err_msg('file type not allowed')
    return 'Access denied!'

if __name__ == '__main__':
    app.run(host=cfg.server_host, port=cfg.server_port, debug=True)