from module import _init_paths
import tensorflow as tf
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.config import cfg
from networks.factory import get_network
from module.config import cfg as module_cfg
from utils.timer import Timer
import numpy as np
import os, sys, cv2

global __INSTANCE__
__INSTANCE__ = None

class VisObject(object):
    pass

class ProcessedImage:

    def __init__(self, image_name):
        self.image_name = image_name
	# Load the demo image
	self.im_file = os.path.join(self.image_name)
	self.cv_im = cv2.imread(self.im_file)
        self.vis = []

    def get_cv_im(self):
        return self.cv_im

    def add_vis(self, vis_object):
	self.vis.append(vis_object)

    def get_vis(self):
	return self.vis

class FastRCNNTf:

	def __init__(self, model):
		cfg.TEST.HAS_RPN = True  # Use RPN for proposals

		# init session
		self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		# load network
		self.net = get_network(module_cfg.NET_TYPE)	    
	    # load model
		self.saver = tf.train.Saver()
		self.saver.restore(self.sess, model)
	    #sess.run(tf.initialize_all_variables())

		if module_cfg.DEBUG:
			print('\n\nLoaded network {:s}'.format(model))

	def vis_detections(self, image, class_name, dets, thresh=0.5):
	    """Draw detected bounding boxes."""
	    inds = np.where(dets[:, -1] >= thresh)[0]
	    if len(inds) == 0:
	        return

	    for i in inds:
		vis_object = VisObject()
	        vis_object.bbox = dets[i, :4]
	        vis_object.score = dets[i, -1]	
	        vis_object.class_name = class_name
		image.add_vis(vis_object)
	     
	def process_image(self, image_name):
	    """Detect object classes in an image using pre-computed object proposals."""
	    image = ProcessedImage(image_name)
	    # Detect all object classes and regress object bounds
	    timer = Timer()
	    timer.tic()
	    scores, boxes = im_detect(self.sess, self.net, image.get_cv_im())
	    timer.toc()
	    image.processing_time = timer.total_time

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
		self.vis_detections(image,cls,dets, thresh=module_cfg.CONF_THRESH)
	    
	    return image.get_vis()

def init_tf_network(model):
	global __INSTANCE__

	if __INSTANCE__ is None:
		__INSTANCE__ = FastRCNNTf(model)

def process_image(image):
	global __INSTANCE__

	if __INSTANCE__ is None:
		raise Exception('Fast_Rcnn_tf shall be initialized first: call init_tf_network(model)')

	return __INSTANCE__.process_image(image)
	

# #########################################################
# Below is the code on case of direct usage from command-line
# #########################################################

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
