from flask import Flask, request, jsonify
from flask_cors import CORS  # 导入 CORS
import os
import base64
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
from yoloseg import YOLOSeg

# 创建一个 Flask 应用实例
app = Flask(__name__)
CORS(app)  # 启用 CORS，允许所有域名访问

# 设置上传文件的保存路径
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_transform():
    normalize = transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    )
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return transform

def read_labels(path):
    out = []
    with open(path, "r") as f:
        for line in f:
            out.append(line.strip())
    return out

# 确保上传文件夹存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 定义一个路由，当用户访问根路径时执行
@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/upload', methods=['POST'])
def upload():
    # 获取请求中的参数
    data = request.json
    print(data)
    img_64 = data.get('img_64')
    model = data.get('model')

    if not img_64 or not model:
        return jsonify({'code': 400, 'msg': 'Missing img_64 or model parameter'})

    try:
        # 解码 Base64 图片
        img_data = base64.b64decode(img_64)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # 保存图片到临时文件
        filename = 'temp_image.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filepath, img)

        # 如果 model 参数为 'detected'，返回 Base64 编码的图片
        if model == 'detected':
            # Detect Objects
            boxes, scores, class_ids, masks = yoloseg(img)

            # Draw detections
            combined_img = yoloseg.draw_masks(img)
            cv2.imwrite(filepath, combined_img)

            with open(filepath, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

            # 添加 data: 前缀和 MIME 类型
            encoded_image_with_prefix = f'data:image/jpeg;base64,{encoded_image}'

            return jsonify({
                'code': 200,
                'msg': 'File uploaded and processed successfully',
                'data': {
                    'image': encoded_image_with_prefix,  # 带前缀的 Base64 编码图片
                    'class': None  # 不需要返回分类结果
                }
            })

        # 如果 model 参数为 'classification'，返回分类结果
        elif model == 'classification':
            # 这里可以添加图片分类的逻辑
            img_file = filepath
            input_shape = [1, 3, 224, 224]
            input_image = Image.open(img_file)
            preprocess = preprocess_transform()
            input_data = preprocess(input_image).numpy().reshape(input_shape)

            in_name = "input"

            output = mobi_shl_session.run(None, {in_name: input_data})[0].reshape(-1)
            result = reversed(output.argsort()[-5:])
            labels = read_labels("labels_map.txt")
            for r in result:
                return jsonify({
                    'code': 200,
                    'msg': 'File uploaded and processed successfully',
                    'data': {
                        'image': None,  # 不需要返回图片
                        'class': labels[r],  # 返回分类结果
                        'prob': output[r] * 0.1
                    }
                })

        # 如果 model 参数无效
        else:
            return jsonify({'code': 400, 'msg': 'Invalid model parameter'})

    except Exception as e:
        return jsonify({'code': 500, 'msg': f'Error processing image: {str(e)}'})

# 运行应用
if __name__ == '__main__':
    mobi_model_path = "mobilenetv2-10.onnx"
    mobi_shl_session = ort.InferenceSession(
        mobi_model_path,
        providers=["ShlExecutionProvider"],
        provider_options=[
            {
                # "debug_run_time": 1,
                # "debug_level": "CSINN_DEBUG_LEVEL_DEBUG",
                # "profiler_level": "CSINN_PROFILER_LEVEL_TIMER",
            },
        ],
    )

    # Initialize YOLOv5 Instance Segmentator
    model_path = "models/yolov8m-seg.onnx"
    yoloseg = YOLOSeg(model_path, conf_thres=0.5, iou_thres=0.3)
    app.run(host='0.0.0.0', port=5000, debug=True)