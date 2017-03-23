'''

'''
import io
import time
import picamera
import picamera.array
from picamera.array import PiRGBArray
import tensorflow as tf
import numpy as np
import threading
import requests
import datetime
from subprocess import call
import glob

FILE_PATTERN = './vids/violence%02d.h264' # file pattern for recordings
CONV_PATTERN = './vids/violence%02d.mp4'  # file patter to convert to

FILE_BUFFER = 1048576                     # size of file buffer (bytes)
REQUIRED_VIOLENCE = 0.8

CAM_NAME = 'Camera 1'
CAM_RESOLUTION = (640,480)  # recording resolution
CAM_FRAMERATE = 24          # recording framerate
CAM_SECONDS = 15            # seconds stored in buffer
CAM_BITRATE = 1000000       # bitrate for encoder
CAM_FORMAT = 'bgr'          # format used to record

BASE_URL = 'http://104.196.62.42:8080'
'https://test-project-156600.appspot.com'
SERVER_ADDR = BASE_URL + '/api/reportEvent'

def convert_push_file(file_number):
    old_file = FILE_PATTERN % file_number
    new_file = CONV_PATTERN % file_number
    print("converting %s to %s..." % (old_file, new_file))
    #do conversion

    call(['MP4Box','-fps','%i'%CAM_FRAMERATE,'-add','%s'%old_file,'%s'%new_file])
    push_file(new_file, file_number)

def push_file(filename, file_number):
    print("pushing %s..." % filename)

    with open('%s' % filename, 'rb') as f:
        bin_data = f.read()

    #now_date = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M')
    now_date = datetime.datetime.utcnow().isoformat()[:-7] + 'Z'

    these_header = {'content-disposition': 'form-data'}
    send_fname = 'violence%02d.mp4' % file_number
    these_files = {'videoClip': (send_fname, bin_data, 'video/mp4')}

    r = requests.post(
        SERVER_ADDR,data={'date': now_date, 'cameraName': CAM_NAME},
        headers = these_header,files=these_files)
    print(r.status_code, r.reason)

def get_labels():
    'Get a list of labels so we can see if it is violent or not.'
    with open('train/retrained_labels.txt', 'r') as fin:
        labels = [line.rstrip('\n') for line in fin]
    return labels


def run_classification(labels):
    'Stream images off the camera and process them.'

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
        with tf.gfile.FastGFile('train/retrained_graph.pb', 'rb') as f:
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
                threads = []
                while True:
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
                        if predicted_label == 'violence' and max_value > REQUIRED_VIOLENCE:
                            break

                    # Violence Detected Mode
                    print('Violence detected, dumping to %s' % file_output.name)
                    camera.wait_recording(5) # get 5 seconds after classification
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
                    ring_buffer = picamera.PiCameraCircularIO(
                        camera, seconds=CAM_SECONDS, bitrate=CAM_BITRATE)

                    #convert_push_file(file_number)

                    t = threading.Thread(
                        target=convert_push_file, args=(file_number,))
                    threads.append(t)
                    t.start()

                    # Reset back to Violence Not Detected mode
                    camera.split_recording(ring_buffer)
                    file_number += 1
                    file_output.close()
                    file_output = io.open(
                        FILE_PATTERN % file_number, 'wb', buffering=FILE_BUFFER)


            except KeyboardInterrupt:
                print("Keyboard Interrypt")
                exit()

            finally:
                camera.stop_recording()
                print("Emptying vids directory...")
                call(['rm'] + glob.glob('./vids/violence*'))
                print("Done")

if __name__ == '__main__':
    print("Starting up WATCHMAN")
    run_classification(get_labels())
