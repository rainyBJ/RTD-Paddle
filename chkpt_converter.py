import paddle
import torch

# convert torch.pth to paddle.pdparams
chkpt_path = 'checkpoint_best_sum_ar.pth'
paddle_chkpt_path = "checkpoint_best_sum_ar.pdparams"

paddle_dict = {}
params_dict = {}
torch_checkpoint = torch.load(chkpt_path)

start_epoch = torch_checkpoint['epoch']
paddle_dict['epoch'] = start_epoch
pretrained_dict = torch_checkpoint['model']

# fc layer, weight needs to be tansposed
fc_names = ["transformer.encoder.layers.0.weight","transformer.encoder.layers.1.weight",
            "transformer.encoder.layers.2.weight","transformer.decoder.layers.0.linear1.weight",
            "transformer.decoder.layers.0.linear2.weight","transformer.decoder.layers.1.linear1.weight",
            "transformer.decoder.layers.1.linear2.weight","transformer.decoder.layers.2.linear1.weight",
            "transformer.decoder.layers.2.linear2.weight","transformer.decoder.layers.3.linear1.weight",
            "transformer.decoder.layers.3.linear2.weight","transformer.decoder.layers.4.linear1.weight",
            "transformer.decoder.layers.4.linear2.weight","transformer.decoder.layers.5.linear1.weight",
            "transformer.decoder.layers.5.linear2.weight","class_embed.weight",
            "bbox_embed.layers.0.weight", "bbox_embed.layers.1.weight",
            "bbox_embed.layers.2.weight","iou_embed.layers.0.weight",
            "iou_embed.layers.1.weight","iou_embed.layers.2.weight"]
print("Total FC Layers: {}".format(len(fc_names)))

count = 0
for key in pretrained_dict:
    params = pretrained_dict[key].cpu().detach().numpy()
    flag = [item in key for item in fc_names]
    if any(flag):
        params = params.transpose()
        count = count + 1
    params_dict[key] = params
print("{} FC Layer Params Transposed".format(count))

paddle_dict['model'] = params_dict
paddle.save(paddle_dict,paddle_chkpt_path)

paddle_checkpoint = paddle.load(paddle_chkpt_path)

