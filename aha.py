import tensorflow as tf

if tf.test.is_gpu_available():
    print("Yup, gpu there")
else:
    print("got chopped")
