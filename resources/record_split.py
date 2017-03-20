import picamera

with picamera.PiCamera() as camera:
    camera.resolution = (640, 480)
    camera.start_recording('1.h264')
    for i in range(2, 10):
        camera.wait_recording(5)
        camera.split_recording('%d.h264' % i)
    camera.stop_recording()
