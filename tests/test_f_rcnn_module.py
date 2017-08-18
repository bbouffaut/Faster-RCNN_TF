import _init_paths
import faster_rcnn_tf as f_rcnn

f_rcnn.init_tf_network('/srv/workspace/Faster-RCNN_TF/data/pretrain_model/VGGnet_fast_rcnn_iter_70000.ckpt')
vis = f_rcnn.process_image('./data/demo/004545.jpg')

print([(v.class_name,v.score) for v in vis] )
