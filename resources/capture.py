def capture_images(save_folder):
    """Stream images off the camera and save them."""
    camera = PiCamera()
    camera.resolution = (320, 240)
    camera.framerate = 10

    # Warmup...
    time.sleep(2)

    # And capture continuously forever.
    for i, frame in enumerate(camera.capture_continuous(
        save_folder + '{timestamp}.jpg',
        'jpeg', use_video_port=True
    )):
        pass
