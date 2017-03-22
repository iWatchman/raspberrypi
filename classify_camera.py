'''

'''
import io
import time
import picamera
import picamera.array
from picamera.array import PiRGBArray
import tensorflow as tf
import numpy as np

FILE_PATTERN = './vids/violence%02d.h264' # file pattern for recordings
FILE_BUFFER = 1048576                    # size of file buffer (bytes)

CAM_RESOLUTION = (640,480)  # recording resoluition
CAM_FRAMERATE = 24           # recording framerate
CAM_SECONDS = 10            # seconds stored in buffer
CAM_BITRATE = 1000000       # bitrate for encoder
CAM_FORMAT = 'bgr'          # format used to record

def get_labels():
    """Get a list of labels so we can see if it's violent or not."""
    with open('train/retrained_labels.txt', 'r') as fin:
        labels = [line.rstrip('\n') for line in fin]
    return labels


def run_classification(labels):
    """Stream images off the camera and process them."""

    print("Initializing camera...")
    with picamera.PiCamera() as camera:
        camera.resolution = CAM_RESOLUTION
        camera.framerate = CAM_FRAMERATE
        rawimage = PiRGBArray(camera, size=CAM_RESOLUTION)
        time.sleep(2)

        # Create frame ring buffer for temporary image storage
        print("Creating buffers...")
        #ringbuf = RingBuffer(300)
        camera.start_preview()
        ring_buffer = picamera.PiCameraCircularIO(
            camera, seconds=CAM_SECONDS, bitrate=CAM_BITRATE)
        file_number = 1
        file_output = io.open(
            FILE_PATTERN % file_number, 'wb', buffering=FILE_BUFFER)
        rawCapture = PiRGBArray(camera, size=CAM_RESOLUTION)

        #print("Creating Violence Detector...")
        #violence_detector = ViolenceDetector(camera)
        #violence_detector.initializeTF(labels)

        # Unpersists graph from file
        print("Loading graph...")
        with tf.gfile.FastGFile("train/retrained_graph.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

        print("Creating TF Session...")
        with tf.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

            print("Starting Camera...")
            camera.start_recording(
                ring_buffer, format='h264')
            try:
                while True:
                    print('Waiting for violence')
                    for i, image in enumerate(
                            camera.capture_continuous(
                                rawCapture, format='bgr', use_video_port=True
                            )
                        ):
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

                        print("%s (%.2f%%)" % (predicted_label, max_value * 100))

                        # Reset the buffer so we're ready for the next one.
                        rawCapture.truncate(0)
                        if predicted_label == 'violence' and max_value > 0.7:
                            break

                    # Violence Detected Mode
                    print('Violence detected, dumping to %s' % file_output.name)
                    with ring_buffer.lock:
                        for frame in ring_buffer.frames:
                            if frame.frame_type == picamera.PiVideoFrameType.sps_header:
                                ring_buffer.seek(frame.position)
                                break
                        while True:
                            buf = ring_buffer.read1()
                            if not buf:
                                break
                            file_output.write(buf)
                    camera.split_recording(file_output)
                    # Clear ring buffer by reconstructing it
                    # TODO: add a clear() method later...
                    ring_buffer = picamera.PiCameraCircularIO(
                        camera, seconds=CAM_SECONDS, bitrate=CAM_BITRATE)

                    # Wait CAM_SECONDS without classification to refill buffer
                    # TODO: setup some double-buffer system so this isn't necessary
                    check = time.time()
                    while check > time.time() - CAM_SECONDS:
                        camera.wait_recording(1)

                    # Reset back to Violence Not Detected mode
                    camera.split_recording(ring_buffer)
                    file_number += 1
                    file_output.close()
                    file_output = io.open(
                        FILE_PATTERN % file_number, 'wb', buffering=FILE_BUFFER)
            except KeyboardInterrupt:
                print('Keyboard Interrypt')
                exit()

            finally:
                camera.stop_recording()

if __name__ == '__main__':
    print("Starting up WATCHMAN")
    run_classification(get_labels())
