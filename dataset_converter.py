import torch
import paddle
import os

# I3D_features from ./data to ./data_paddle
# DataType from torch.tensor to paddle.tensor
ori_path = 'data/I3D_features/'
files = os.listdir(ori_path)
det_path = 'data_paddle/I3D_features/'
print("Total files: {}".format(len(files)))

for file in files:
    data_torch = torch.load(ori_path + file)
    data_np = data_torch.numpy()
    # data_paddle = paddle.to_tensor(data_np)
    # if os.path.exists(det_path + file):
    #     continue
    paddle.save(data_np,det_path + file)
    data_paddle_loaded = paddle.load(det_path + file)
    # print(data_paddle==data_paddle_loaded)
    # break
files = os.listdir(det_path)
print("{} files has been converted!: ".format(len(files)))
