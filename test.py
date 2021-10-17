import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.nn.layer.transformer import Transformer
import torch
from scipy.spatial.distance import cdist


# q = np.random.rand(32, 32, 512).astype(
#         "float32")
# k = np.random.rand(32, 32, 512).astype(
#         "float32")
# # params
# tgt_len = 32
# bsz = 32
# num_heads = 8
# head_dim = 64
# # run paddle
# q_paddle = paddle.to_tensor(q)
# k_paddle = paddle.to_tensor(k)
# q_paddle = paddle.tensor.reshape(x=q_paddle, shape=[0, 0, num_heads, head_dim])
# q_paddle = paddle.tensor.transpose(x=q_paddle, perm=[0, 2, 1, 3])
# k_paddle = paddle.tensor.reshape(x=k_paddle, shape=[0, 0, num_heads, head_dim])
# k_paddle = paddle.tensor.transpose(x=k_paddle, perm=[0, 2, 1, 3])
# paddle_output = paddle.matmul(x=q_paddle, y=k_paddle, transpose_y=True)
# # run torch
# q = q.transpose(1,0,2)
# k = k.transpose(1,0,2)
# q_torch = torch.from_numpy(q)
# k_torch = torch.from_numpy(k)
# q_torch = q_torch.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
# k_torch = k_torch.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
# torch_output = torch.bmm(q_torch, k_torch.transpose(1, 2))
#
# print(paddle_output)
# print(torch_output)

# data_np = np.random.rand(1024).astype(
#         "float32")

# data_np = [8.7500]
# data_paddle = paddle.to_tensor(data_np)
# data_torch = torch.tensor(data_np)
#
# out_torch = data_torch.median()
# out_paddle = data_paddle.median()
#
# print(1)

np_data = np.random.rand(2).astype("float32")
np_output = np.median(np_data)
torch_data = torch.from_numpy(np_data)
paddle_data = paddle.to_tensor(np_data)
torch_output = torch_data.median()
paddle_output = paddle_data.median()
print(np_output)
print(paddle_output)
print(torch_output)
print(1)