# Self-Driving Car Training
Convolutional Neural Network model for the [Udacity Self-Driving car simulator](https://github.com/udacity/self-driving-car-sim).

Contains the model, the training script, and the launching script.


```
usage: train.py [-h] [-c]

Train a self-driving car.

optional arguments:
  -h, --help  show this help message and exit
  -c          Create a new model
```

The ``train.py`` file is used to create and/or train the model.
You can either train a new model using the `-c` argument or train an existing one.
File names such as model file or driving log cannot be passed as argument.

> train.py always looks for the model file "model.h5".

> If the file exists and -c is not used, the model will be retrained.

> If the file doesn't exist and -c is used, the file will be created.

> No warranty is provided otherwise.

## How to train a model
Two files are required:
- driving_log.csv
- validation_log.csv

Both are the same type of file and are generated from the simulator in training mode.
driving_log is the training set and validation_log.csv the test/validation set.
Both files must be in the same directory as the training script `train.py`.
Ensure the image files referenced by the logs are located in the subdirectory `IMG/` of the logs files.

Simply launch the training script
```
python3 train.py [-c]
```
The model file "model.h5" will be saved in the script directory and a
training history log file will be produced named model_history_log.csv

## How to run a model
Ensure the model file "model.h5" is located in the same directory as the running script drive.py

Simply launch the running script
```
python3 drive.py
```
Then launch the simulator and select "autonomous mode".

## Model provided
A pre-trained model is provided in the repo.

This model has been trained for 200 epochs of 8000 images.

It can be used to immediately evaluate the architecture.

## Requirements
Built on python 3.7.3

Use `pip install -r requirements.txt` for fast install

- eventlet >= 0.24.1
- Keras >= 2.2.4
- matplotlib >= 3.1.0
- numpy >= 1.16.3
- pandas >= 0.24.2
- Pillow >= 6.0.0
- scikit-learn >= 0.21.1
- scipy >= 1.3.0
- simple-pid >= 0.2.1
- tensorflow >= 1.13.1

## References
 - Bojarski, Mariusz, et al. "End to end learning for self-driving cars." arXiv preprint arXiv:1604.07316 (2016).
