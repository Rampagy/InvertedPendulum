import sys
sys.path.append("../Shared")

import NeuralNetwork as nn
import serial
from enum import Enum
import numpy as np

# create enumeration for communication with the controller
class Command(Enum):
    LEFT = 0
    RIGHT = 1
    RESET = 2

# create enumeration for communication with the controller
class Packet(Enum):
    POSITION = 0    # x
    VELOCITY = 1    # x_dot
    ANGLE = 2       # theta
    ANGLE_VEL = 3   # theta_dot


# create/load model
model = nn.Control_Model(input_len=4, output_len=2)


with serial.Serial('/dev/ttyUSB0', 115200, timeout=10) as ser:
    ser.flushInput()
    ser.flushOutput()

    # reset the system, before training
    ser.write(Command.RESET.value)

    # start training
    while True:
        # wait until observations are received (BLOCKING)
        dataString = ser.readline()
        print (dataString)
        observations = dataString.split(",")

        # strip out everything except the data needed to predict an action
        predict_obs = int(observations[Packet.POSITION.value : Packet.ANGLE_VEL.value])
        action = model.predict_move(np.asarray(predict_obs, size=(1,4)))

        ser.write(action)
