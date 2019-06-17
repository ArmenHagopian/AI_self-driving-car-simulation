# AI_self-driving-car-simulation

## Files and directories of the repository

The "train.py" file is used to create and/or train the model. Indeed, you can either create and train a new model by executing the python file using the following:

```
python train.py -c
```

or you can load an existing model file named "model.h5". In this case, the "train.py" will train the existing model once again.

A model generated using 200 epochs and also named "model.h5" is in this repository. It can be directly used in a simulation as explained in the next section.

When launched, this "train.py" file also generates a plot of the mean absolute error and the mean squared error of the validation and the test phases of the model. Moreover, the plot is based on a file named "model_history_log.csv" and generated during the training.

Furthermore, in order to train the model, a "driving_log.csv" file is needed. The latter must contain the paths to the left, center, and right cameras images as well as the measured steering angle, throttle, brake and speed related to the images. These data have been created in the training mode of the Udacity simulator.

Another set of data called the validation data is also mandatory. It needs to have the same structure as the "driving_log.csv" but in a file named "validation_log.csv" and containing data that will be used for the validation phase of the model.

Lastly, an important thing is that all these csv files must be in the same directory as the "train.py" file.

## How to run the simulation ?

Once the model has been created and trained, you need to execute the "drive.py" file to run the model. An important thing here is that the file containing the model must be named "model.h5" and be in the same directory as "drive.py". The python file can then be run:

```
python drive.py
```

By choosing the autonomous mode in the simulator, you will be able to get the car to drive by itself based on the aforementioned trained model.


## Algorithms and libraires used

The programming language used for this project is Python.
The neural network has been implemented using Keras API that runs on top of TensorFlow (we used 1.13.1 version).
In order to launch the simulator based on the created model, a socketIO server has to be created. Therefore, the python-socketio (4.0.2) library has been used.
The intelligence implemented controls the steering wheel angle but not the throttle nor the brake. Thus, a PID control has been used for the speed of the car with simple_pid library.
The pandas library has been used to manipulate the csv files containing the training and validation data.
The NumPy, Evenlet, Matplotlib and PIL (for image processing) libraries have also been used.
