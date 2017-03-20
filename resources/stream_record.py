import io
import picamera

stream = io.BytesIO()
with picamera.PiCamera() as camera:
    camera.resolution = (640, 480)
    camera.start_recording(stream, quantization=23)
    camera.wait_recording(15)
    camera.stop_recording()
