import os
import torch
import numpy as np
import torchvision.utils as vutils
from flask import Flask, request, jsonify
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
from network.Transformer import Transformer

app = Flask(__name__)

# 参数配置
input_dir = "./../files" #test_img
output_dir = "./../files/output"#test_output
model_path = "./pretrained_model"
style = "Hosoda"#"Shinkai"
gpu = 0
valid_ext = [".jpg", ".png", ".jpeg",".JPG"]

# 检查目录
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# 加载预训练模型
model = Transformer()
model.load_state_dict(torch.load(os.path.join(model_path, style + "_net_G_float.pth")))
model.eval()

disable_gpu = gpu == -1 or not torch.cuda.is_available()

if disable_gpu:
    print("CPU mode")
    model.float()
else:
    print("GPU mode")
    model.cuda()

@app.route("/runcomicimage", methods=["POST"])
def transform_image():
    try:
        data = request.get_json()
        file_name = data.get('arg1', '默认值1')
        #file_name = request.json.get("file_name")
        if not file_name:
            return jsonify({"error": "File name not provided"}), 400

        file_path = os.path.join(input_dir, file_name)
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404

        ext = os.path.splitext(file_name)[1]
        if ext not in valid_ext:
            return jsonify({"error": "Invalid file extension"}), 400

        # 加载图片
        input_image = Image.open(file_path).convert("RGB")
        input_image = np.asarray(input_image)
        input_image = input_image[:, :, [2, 1, 0]]  # RGB -> BGR
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)
        input_image = -1 + 2 * input_image  # 预处理 (-1, 1)
        input_image = Variable(input_image).cuda() if not disable_gpu else Variable(input_image).float()

        # 前向传播
        output_image = model(input_image)[0]
        output_image = output_image[[2, 1, 0], :, :]  # BGR -> RGB
        output_image = output_image.data.cpu().float() * 0.5 + 0.5

        # 保存图片
        output_path = os.path.join(output_dir, file_name[:-4] + "_" + style + ".jpg")
        vutils.save_image(output_image, output_path)
        outputfile = file_name[:-4] + "_" + style + ".jpg"
        return jsonify({"status": "success", "result": outputfile}), 200
        #return jsonify({"message": "Transformation successful", "output_path": output_path}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5600)
