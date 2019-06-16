#!/usr/local/bin/python3
""" Connects to the simulator and runs the model """
import base64
import io

import eventlet
import numpy as np
import socketio
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
from simple_pid import PID

# Handle to connect to the simulator
SIO = socketio.Server()
# ML handle to predict the steering
MODEL = None
# Speed control with PID controller
REGUL = PID(1, 0.7, 0, setpoint=30, output_limits=(0, 1))


@SIO.on('telemetry')
def telemetry(_, data):
    """ Event sent by the simulator. """
    def preprocess(image):
        """
        First, convert the image from a PIL object to a numpy
        array, then distribute the inputs between [-1.0;1.0].
        """
        img = img_to_array(image)
        return preprocess_input(img, mode='tf', data_format='channels_last')

    if data:
        # The current speed of the car
        speed = REGUL(float(data["speed"]))
        # The current image from the center camera of the car
        image = Image.open(io.BytesIO(base64.b64decode(data["image"])))
        # Use your model to compute steering and throttle
        feats = np.expand_dims(preprocess(image), axis=0)
        steer = MODEL.predict(feats).flatten()[0]
        print("steer: {:f} throttle: {:f}".format(steer, speed))
        # Response to the simulator with a steer angle and throttle
        send(steer, speed)
    else:
        # Edge case
        SIO.emit('manual', data={}, skip_sid=True)


@SIO.on('connect')
def connect(sid, _):
    """ Event fired when simulator connect. """
    print("connect ", sid)
    send(0, 0)


def send(steer, throttle):
    """ To send steer angle and throttle to the simulator. """
    SIO.emit("steer", data={'steering_angle': str(steer),
                            'throttle': str(throttle)},
             skip_sid=True)


if __name__ == '__main__':
    MODEL = load_model('model.h5')
    MODEL.summary()

    # wrap with a WSGI application
    APP = socketio.WSGIApp(SIO)
    # simulator will connect to localhost:4567
    eventlet.wsgi.server(eventlet.listen(('', 4567)), APP)
