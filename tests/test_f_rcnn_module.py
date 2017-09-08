import _init_paths
import faster_rcnn_tf as f_rcnn
import threading
import Queue
import os
from utils.timer import Timer
from handlers import EventsHandler

exit_flag = 0

class MyEventsHandler(EventsHandler):
    def __init__(self,thread_id):
       self.thread_id = thread_id 

    def on_detect_objects(self, image, processing_time, objects):
        print('{} processed image {} in {:.3f} for {:d} objects detected'.format(self.thread_id, image.image_name, processing_time, len(objects)))
	if (len(objects) > 0):
	    output_queue_lock.acquire()
	    output_queue.put(image)
	    output_queue_lock.release()


class ImageProcessingThread(threading.Thread):
	
	def __init__(self,thread_id,queue):
	    threading.Thread.__init__(self)
	    self.thread_id = thread_id
	    self.q = queue
	
	def run(self):
	    if ('Output' in self.thread_id):
		print('Start OutputThread processing {}'.format(self.thread_id))
	        process_output_queue(self.thread_id, self.q)
	    else:
		print('Start InputThread processing {}'.format(self.thread_id))
	        process_input_queue(self.thread_id, self.q)


def process_input_queue(thread_id, q):
	events_handler = MyEventsHandler(thread_id)
	while not exit_flag:
	    input_queue_lock.acquire()
	    if not q.empty():
	        try:
	            image_name = q.get()
		    input_queue_lock.release()
		    f_rcnn.process_image(image_name,events_handler)
		except Exception as ex:
		    print('Exception occured {}'.format(ex))
	            input_queue_lock.release()
	    else:
	        input_queue_lock.release()
	    
def process_output_queue(thread_id, q):
	while not exit_flag:
	    output_queue_lock.acquire()
	    if not q.empty():
	        try:
		    image = q.get()
		    output_queue_lock.release()
		    annotated_image = image.get_annotated_image()
		    print('OutputProcessing {}'.format(annotated_image))
       		except Exception as ex:
		    print('Exception occured {}'.format(ex))
		    output_queue_lock.release()
       	    else:
	        output_queue_lock.release()
		    	
input_queue_lock = threading.Lock()
input_queue = Queue.Queue()
threads = []
output_queue = Queue.Queue()
output_queue_lock = threading.Lock()
output_threads = []

nb_threads = 1
images_list = ['000456.jpg','000542.jpg','001150.jpg','001763.jpg','004545.jpg','pedestrian_cars.jpg','000456.jpg','000542.jpg','001150.jpg','001763.jpg','004545.jpg','pedestrian_cars.jpg']
nb_output_threads = 1

timer = Timer()

# initialize TF network
f_rcnn.init_tf_network('/srv/workspace/Faster-RCNN_TF/data/pretrain_model/VGGnet_fast_rcnn_iter_70000.ckpt')

# Create new threads
for t_name in [('Thread-' + str(i)) for i in range(nb_threads)]:
   thread = ImageProcessingThread(t_name, input_queue)
   thread.start()
   threads.append(thread)

# Create new OUTPUT threads
for t_name in [('OutputThread-' + str(i)) for i in range(nb_output_threads)]:
   thread = ImageProcessingThread(t_name, output_queue)
   thread.start()
   output_threads.append(thread)


# Fill the input queue with list of images => START Timer
timer.tic()
input_queue_lock.acquire()

for image in images_list:
    input_queue.put(os.path.join('/srv/workspace/Faster-RCNN_TF/data/demo/',image))

input_queue_lock.release()

# Wait for queue to empty
while not input_queue.empty():
    pass

# Notify threads it's time to exit when queue is empty
exit_flag = 1

# Wait for all threads to complete
for t in threads:
   t.join()

#Stop Timer
timer.toc()

print("Processing of {} images took {:.3f} sec".format(len(images_list),timer.total_time))
