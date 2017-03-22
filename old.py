'''

'''
import time
from picamera import PiCamera
from picamera.array import PiRGBArray
import tensorflow as tf
from my_ring_buf import RingBuffer
import numpy
from PIL import Image

CAM_TYPE = 'bgr'
CAM_RESOLUTION = (640,480)

class RingBuffer():
    # Provides a ring buffer of PiRGBArrays
    def __init__(self,length,resolution):
        self.cur = 0
        self.data = [PiRGBArray(camera, size=resolution) for i in range(length)]

def get_labels():
    """Get a list of labels so we can see if it's an ad or not."""
    with open('train/retrained_labels.txt', 'r') as fin:
        labels = [line.rstrip('\n') for line in fin]
    return labels

def array2PIL(arr, size):
    mode = 'RGBA'
    arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    if len(arr[0]) == 3:
        arr = numpy.c_[arr, 255*numpy.ones((len(arr),1), numpy.uint8)]
    return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)

def run_classification(labels):
    """Stream images off the camera and process them."""

    print("Initializing camera...")
    cam_width = 320
    cam_height = 240
    camera = PiCamera()
    camera.resolution = (cam_width, cam_height)
    camera.framerate = 2
    rawCapture = PiRGBArray(camera, size=(cam_width, cam_height))
    time.sleep(2) # Warmup...

    # Create frame ring buffer for temporary image storage
    print("Creating frame buffer...")
    #ringbuf = RingBuffer(300)

    # Unpersists graph from file
    print("Loading graph...")
    with tf.gfile.FastGFile("train/retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    print("Creating TF Session...")
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        print("Capturing images...")
        # count num of images classified
        for i, image in enumerate(
                camera.capture_continuous(
                    rawCapture, format=CAM_TYPE, use_video_port=True
                )
            ):

            start = time.time()
            # Get the numpy version of the image.
            decoded_image = image.array

            # Make the prediction. Big thanks to this SO answer:
            # http://stackoverflow.com/questions/34484148/feeding-image-data-in-tensorflow-for-transfer-learning
            predictions = sess.run(softmax_tensor, {'DecodeJpeg:0': decoded_image})
            prediction = predictions[0]

            # Get the highest confidence category.
            prediction = prediction.tolist()
            max_value = max(prediction)
            max_index = prediction.index(max_value)
            predicted_label = labels[max_index]

            end = time.time()
            print("%s (%.2f%%) t:%.2f sec" % (predicted_label, max_value * 100, end-start))

            # Reset the buffer so we're ready for the next one.
            rawCapture.truncate(0)

if __name__ == '__main__':
    print("Starting up WATCHMAN")
    run_classification(get_labels())
