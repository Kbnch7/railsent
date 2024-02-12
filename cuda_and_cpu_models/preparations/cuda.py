import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
tf.debugging.set_log_device_placement(True)

print("TensorFlow Version:", tf.__version__)
print("Is Built With CUDA:", tf.test.is_built_with_cuda())
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

if tf.config.list_physical_devices('GPU'):
    for device in tf.config.list_physical_devices('GPU'):
        print("Device Name:", device.name)
else:
    print("No GPU.")