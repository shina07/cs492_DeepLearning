import sys
import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

def custom_model_fn(features, labels, mode, params):
    """Model function for PA1"""

    depth = params["depth"]
    units = params["units"]
    learning_rate = params["learning_rate"]

    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 784]) # You also can use 1 x 784 vector

    first_hidden_layer = tf.layers.dense (inputs = input_layer, units = units, activation = tf.nn.relu)    
    second_hidden_layer = tf.layers.dense (inputs = first_hidden_layer, units = units, activation = tf.nn.relu)

    # Output logits Layer
    logits = tf.layers.dense(inputs = second_hidden_layer, units = 10)

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

    loss = tf.losses.sparse_softmax_cross_entropy (
        labels,
        logits
    )

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # optimizer = tf.train."custom optimizer" # Refer to tf.train
        optimizer = tf.train.GradientDescentOptimizer (learning_rate) # Learning Rate
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def print_usage ():
    print ("usage: $ python3 skeleton.py [depth] [units] [batch size] [learning rate] [steps]")

if __name__ == '__main__':
    argv = sys.argv[1:]

    if len (sys.argv) != 6:
        print_usage ()
        sys.exit ()

    # layers = 3
    # units = 100
    # batch_size = 100
    # learning_rate = 0.1
    # steps = 20000
    
    depth = int (argv[0])
    units = int (argv[1])
    batch_size = int (argv[2])
    learning_rate = float (argv[3])
    steps = int (argv[4])


    # Write your dataset path
    dataset_train = np.load('cs492c_assignment1_data/train.npy')
    dataset_eval =  np.load('cs492c_assignment1_data/valid.npy')
    test_data =  np.load('cs492c_assignment1_data/test.npy')

    train_data = dataset_train[:,:784]
    train_labels = dataset_train[:,784].astype(np.int32)
    eval_data = dataset_eval[:,:784]
    eval_labels = dataset_eval[:,784].astype(np.int32)

    # Save model and checkpoint
    model_params = {"depth" : depth, "units" : units, "learning_rate" : learning_rate}
    classifier = tf.estimator.Estimator(model_fn=custom_model_fn, model_dir="./model", params = model_params)

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # Train the model. You can train your model with specific batch size and epoches
    train_input = tf.estimator.inputs.numpy_input_fn(x={"x": train_data},
        y=train_labels, batch_size=batch_size, num_epochs=None, shuffle=True)
    classifier.train(input_fn=train_input, steps=steps, hooks=[logging_hook])

    # Eval the model. You can evaluate your trained model with validation data
    eval_input = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data},
        y=eval_labels, num_epochs=1, shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input)


    ## ----------- Do not modify!!! ------------ ##
    # Predict the test dataset
    pred_input = tf.estimator.inputs.numpy_input_fn(x={"x": test_data}, shuffle=False)
    pred_results = classifier.predict(input_fn=pred_input)
    # result = np.asarray([x.values()[1] for x in list(pred_results)])
    pred_list = list (pred_results)
    result = np.asarray ([list (x.values())[1] for x in pred_list])
    ## ----------------------------------------- ##

    np.save('20130538.npy', result)
