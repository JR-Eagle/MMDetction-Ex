from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
In [2]:
#faster_rcnn
# config_file = '../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# ssd
config_file = '../configs/faster_rcnn/ssdlite_mobilenetv2_scratch_600e_coco.py"
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
# faster_rcnn
# checkpoint_file = '../checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
# SSD
checkpoint_file = '../checkpoints/ssd300_coco_20210803_015428-d231a06e.pth'
In [3]:
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
In [4]:
# test a single image
img = 'demo.jpg'
result = inference_detector(model, img)
In [5]:
# show the results
# show_result_pyplot(model, img, result)
model.show_result(img,result,out_file='img_rst.jpg')
# video = mmcv.VideoReader('demo.mp4')
# id = 0
# for frame in video:
# result = inference_detector(model,frame)
# model.show_result(frame,result,out_file=str(id)+".jpg")
# id = id + 1
