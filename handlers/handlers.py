class EventsHandler():

    def on_detect_objects(self, image, processing_time, objects, image_annotated):
        """Called every time objects are detected in the image.
        "objects" is a list of objects returned by Fast_rcnn_tf algo.
        """
