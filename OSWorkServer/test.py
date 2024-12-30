import requests
import base64
import json

# 测试接口的 URL
url = "http://127.0.0.1:5000/upload"

# 读取图片文件并转换为 Base64 编码
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(imagce_file.read()).decode('utf-8')

# 测试函数
def test_upload(image_path, model):
    # 将图片转换为 Base64 编码
    img_64 = image_to_base64(image_path)

    # 构造请求数据
    data = {
        "img_64": img_64,
        "model": model
    }

    # 发送 POST 请求
    response = requests.post(url, json=data)

    # 打印响应结果
    print("Status Code:", response.status_code)
    print("Response JSON:", json.dumps(response.json(), indent=4))

# 测试示例
if __name__ == "__main__":
    # 测试图片路径
    image_path = "/home/aqing/Desktop/workpro/cat.jpg"  # 替换为你的测试图片路径

    # 测试模型参数
    model = "detected"  # 可以是 "detected" 或 "classification"

    # 调用测试函数
    test_upload(image_path, model)