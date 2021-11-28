from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
#faster_rcnn
config_file = '../../configs/mask_rcnn/mask_rcnn_x101_64x4d_fpn_2x_coco.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# url: https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

checkpoint_file = '../../checkpoints/mask_rcnn_x101_64x4d_fpn_2x_coco_20200509_224208-39d6f70c.pth'
# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
# test a single image
img = 'demo.jpg'
result = inference_detector(model, img)
# show the results
# show_result_pyplot(model, img, result)
model.show_result(img,result,out_file='MASK_RCNN_Img.jpg')


# video = mmcv.VideoReader('demo.mp4')
# id = 0
# for frame in video:
# result = inference_detector(model,frame)
# model.show_result(frame,result,out_file=str(id)+".jpg")
# id = id + 1
