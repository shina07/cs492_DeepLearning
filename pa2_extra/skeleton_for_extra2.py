import sys
import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

TRAIN_DATA_PATH = "extra2-train.npy"
EVAL_DATA_PATH = "extra2-valid.npy"
TEST_DATA_PATH = "extra2-test_img.npy"

def custom_model_fn(features, labels, mode, params):
    """Model function for PA2 Extra2"""

    depth = params["depth"]
    units = params["units"]
    learning_rate = params["learning_rate"]
    dropout_rate = params["dropout_rate"]

    # If is_training is False, dropout is not applied
    is_training = False
    if mode == tf.estimator.ModeKeys.TRAIN and dropout_rate > 0:
        is_training = True

    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 32, 32, 3]) # cifar-10

    # Convolutional-Network-3
    if (depth == 3):
        conv_layer = tf.layers.conv2d (inputs = input_layer, filters = 32, kernel_size = [filter_size, filter_size], padding = "same", activation = tf.nn.relu)
        conv_layer = tf.layers.conv2d (inputs = conv_layer, filters = 32, kernel_size = [filter_size, filter_size], padding = "same", activation = tf.nn.relu)
        pooling = tf.layers.max_pooling2d (inputs = conv_layer, pool_size = [2, 2], strides = 2)
        pooling_flat = tf.reshape (pooling, [-1, 16 * 16 * 32])

    if (depth == 5):
        conv_layer = tf.layers.conv2d (inputs = input_layer, filters = 32, kernel_size = [filter_size, filter_size], padding = "same", activation = tf.nn.relu)
        conv_layer = tf.layers.conv2d (inputs = conv_layer, filters = 32, kernel_size = [filter_size, filter_size], padding = "same", activation = tf.nn.relu)
        pooling = tf.layers.max_pooling2d (inputs = conv_layer, pool_size = [2, 2], strides = 2)
        dropout = tf.layers.dropout (inputs = pooling, rate = dropout_rate, training = is_training)

        conv_layer = tf.layers.conv2d (inputs = pooling, filters = 64, kernel_size = [filter_size, filter_size], padding = "same", activation = tf.nn.relu)
        conv_layer = tf.layers.conv2d (inputs = conv_layer, filters = 64, kernel_size = [filter_size, filter_size], padding = "same", activation = tf.nn.relu)
        pooling = tf.layers.max_pooling2d (inputs = conv_layer, pool_size = [2, 2], strides = 2)
        pooling_flat = tf.reshape (pooling, [-1, 8 * 8 * 64])

    if (depth == 7):
        conv_layer = tf.layers.conv2d (inputs = input_layer, filters = 32, kernel_size = [filter_size, filter_size], padding = "same", activation = tf.nn.relu)
        conv_layer = tf.layers.conv2d (inputs = conv_layer, filters = 32, kernel_size = [filter_size, filter_size], padding = "same", activation = tf.nn.relu)
        pooling = tf.layers.max_pooling2d (inputs = conv_layer, pool_size = [2, 2], strides = 2)
        dropout = tf.layers.dropout (inputs = pooling, rate = dropout_rate, training = is_training)

        conv_layer = tf.layers.conv2d (inputs = dropout, filters = 64, kernel_size = [filter_size, filter_size], padding = "same", activation = tf.nn.relu)
        conv_layer = tf.layers.conv2d (inputs = conv_layer, filters = 64, kernel_size = [filter_size, filter_size], padding = "same", activation = tf.nn.relu)
        pooling = tf.layers.max_pooling2d (inputs = conv_layer, pool_size = [2, 2], strides = 2)
        dropout = tf.layers.dropout (inputs = pooling, rate = dropout_rate, training = is_training)

        conv_layer = tf.layers.conv2d (inputs = dropout, filters = 128, kernel_size = [3, 3], padding = "same", activation = tf.nn.relu)
        conv_layer = tf.layers.conv2d (inputs = conv_layer, filters = 128, kernel_size = [3, 3], padding = "same", activation = tf.nn.relu)
        pooling = tf.layers.max_pooling2d (inputs = conv_layer, pool_size = [2, 2], strides = 2)
        pooling_flat = tf.reshape (pooling, [-1, 4 * 4 * 128])
    
    # dense = tf.layers.dense (inputs = pooling_flat, units = units, activation = tf.nn.relu)
    dropout = tf.layers.dropout (inputs = pooling_flat, rate = dropout_rate, training = is_training)

    # Output logits Layer
    # logits = tf.layers.dense(inputs = hidden_layer, units = 10)
    logits = tf.layers.dense(inputs = dropout, units = 10)


    # Output logits Layer
    # logits = tf.layers.dense(inputs="your custom layer", units=10)

    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # In predictions, return the prediction value, do not modify
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Select your loss and optimizer from tensorflow API
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy (labels, logits) # Refer to tf.losses

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer (learning_rate) # Refer to tf.train
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def print_usage ():
    # print ("usage: $ python3 skeleton.py [depth] [units] [batch size] [learning rate] [steps] [dropout rate] [optimizer]")
    print ("usage: $ python3 skeleton.py [depth] [units] [batch size] [learning rate] [steps] [dropout rate]")
    print ("- depth         : 3 / 5 / 7")
    print ("- units         : number of units in hidden layer")
    print ("- batch_size    : batch size when training the model")
    print ("- learning rate : learning rate for optimizer")
    print ("- steps         : total steps when training the model")
    print ("- dropout rate  : rate when applying dropout. 0.0 for doing nothing")
    # print ("- optimizer     : D for GradientDescent, A for Adam")

def validate_depth (depth):
    if (depth == 3 or depth == 5 or depth == 7):
        return True
    return False

if __name__ == '__main__':
    argv = sys.argv[1:]

    if len (sys.argv) != 6:
        print_usage ()
        sys.exit ()
    
    depth = int (argv[0])
    # units = int (argv[1])
    units = 1024
    batch_size = int (argv[1])
    learning_rate = float (argv[2])
    steps = int (argv[3])
    # dropout_rate = float (argv[5])
    dropout_rate = 0.5
    filter_size = int (argv[4])

    if (validate_depth (depth) == False):
        print_usage ()
        sys.exit ()

    # Write your dataset path
    dataset_train = np.load(TRAIN_DATA_PATH)
    dataset_eval =  np.load(EVAL_DATA_PATH)
    test_data =  np.load(TEST_DATA_PATH)

    f_dim = dataset_train.shape[1] - 1
    train_data = dataset_train[:,:f_dim].astype(np.float32)
    train_labels = dataset_train[:,f_dim].astype(np.int32)
    eval_data = dataset_eval[:,:f_dim].astype(np.float32)
    eval_labels = dataset_eval[:,f_dim].astype(np.int32)
    test_data = test_data.astype(np.float32)

    # Save model and checkpoint
    model_params = {"depth" : depth, "units" : units, "learning_rate" : learning_rate, "dropout_rate" : dropout_rate, "filter_size" : filter_size} 
    mnist_classifier = tf.estimator.Estimator(model_fn=custom_model_fn, model_dir="./model", params = model_params)

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # Train the model. You can train your model with specific batch size and epoches
    train_input = tf.estimator.inputs.numpy_input_fn(x={"x": train_data},
        y=train_labels, batch_size=100, num_epochs=None, shuffle=True)
    mnist_classifier.train(input_fn=train_input, steps=20000, hooks=[logging_hook])

    # Eval the model. You can evaluate your trained model with validation data
    eval_input = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data},
        y=eval_labels, num_epochs=1, shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input)


    ## ----------- Do not modify!!! ------------ ##
    # Predict the test dataset
    pred_input = tf.estimator.inputs.numpy_input_fn(x={"x": test_data}, shuffle=False)
    pred_results = mnist_classifier.predict(input_fn=pred_input)
    pred_list = list (pred_results)
    result = np.asarray ([list (x.values())[1] for x in pred_list])
    ## ----------------------------------------- ##

    np.save('extra_20130538_network_%d.npy' %depth, result)
