import _init_paths
import tensorflow as tf
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.config import cfg
from networks.factory import get_network
from faster_rcnn_tf_module import config as module_cfg
import numpy as np
import os, sys, cv2

class FastRCNNTf:

	def __init__(self, model):
		cfg.TEST.HAS_RPN = True  # Use RPN for proposals

		# init session
		self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		# load network
		self.net = get_network()	    
	    # load model
		self.saver = tf.train.Saver()
		self.saver.restore(sess, model)
	    #sess.run(tf.initialize_all_variables())

		if cfg.DEBUG:
			print('\n\nLoaded network {:s}'.format(args.model))

	def vis_detections(self, im, class_name, dets, thresh=0.5):
	    """Draw detected bounding boxes."""
	    inds = np.where(dets[:, -1] >= thresh)[0]
	    if len(inds) == 0:
	        return

	    vis = []

	    for i in inds:
	        bbox = dets[i, :4]
	        score = dets[i, -1]
	        vis.append()
		return (class_name,score)

	     
	def process_image(self, image_name):
	    """Detect object classes in an image using pre-computed object proposals."""

	    # Load the demo image
	    im_file = os.path.join(image_name)
	    #im_file = os.path.join('/home/corgi/Lab/label/pos_frame/ACCV/training/000001/',image_name)
	    im = cv2.imread(im_file)

	    # Detect all object classes and regress object bounds
	    scores, boxes = im_detect(self.sess, self.net, im)

	    if module_cfg.DEBUG:
	    	print ('Detection took {:.3f}s for '
	        	'{:d} object proposals').format(timer.total_time, boxes.shape[0])

	    for cls_ind, cls in enumerate(module_cfg.CLASSES[1:]):
	        cls_ind += 1 # because we skipped background
	        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
	        cls_scores = scores[:, cls_ind]
	        dets = np.hstack((cls_boxes,
	                          cls_scores[:, np.newaxis])).astype(np.float32)
	        keep = nms(dets, module_cfg.NMS_THRESH)
	        dets = dets[keep, :]
		
		return vis_detections(im,cls,dets, thresh=module_cfg.CONF_THRESH)

def init_tf_network(model):
	global fast_rcnn_tf_instance

	if not fast_rcnn_tf_instance:
		fast_rcnn_tf_instance = FastRCNNTf(model)

def process_image(image):
	if not fast_rcnn_tf_instance:
		raise Exception('Fast_Rcnn_tf shall be initialized first: call init_tf_network(model)')

	return fast_rcnn_tf_instance.process_image(image)
	
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--model', dest='model', help='Model path',
                        required=True)
    parser.add_argument('image_name')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

	args = parse_args()
	init_tf_network(args.model)

	print('\n\nLoaded network {:s}'.format(args.model))
	print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
	print('Demo for {}'.format(args.image_name))

	vis = process_image(args.image_name)
	print(vis)
