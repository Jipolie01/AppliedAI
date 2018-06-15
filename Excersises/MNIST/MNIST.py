import pickle, gzip, os
import numpy as np
from urllib import request
from pylab import imshow, show, cm

import tensorflow as tf
import tensorflow.contrib.eager as tfe

"""
url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
if not os.path.isfile("mnist.pkl.gz"):
   request.urlretrieve(url, "mnist.pkl.gz")

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='latin1') 
f.close()

def get_image(number): 
  (X, y) = [img[number] for img in train_set] 
  return (np.array(X), y) 

def view_image(number): 
  (X, y) = get_image(number) 
  print("Label: ", imshow(X.reshape(28,28), cmap=cm.gray))
  show()


Get_image already works with the trainset 

Gives np.array
[0, 0, 0, 0, 0, 0
....
0, 0, 0, 0, 0, 0]
Gives values of pixels in array form (first index of image)
Second index is the dtype. Which is mostly floats32


"""

def model(features, labels, mode):
    #making input layer
    input_layer = tf.reshape(features["x"], [-1,28*28])
    
    #Rest of layers
    secondLayer = tf.layers.dense(inputs=input_layer, units=10, activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=secondLayer, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    outputLayer = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=outputLayer, axis=1),
        # Add 'softmax_tensor' to the graph. It is used for PREDICT and by the 'logging_hook'
        "probabilities": tf.nn.softmax(outputLayer, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=outputLayer)

    # Configure the Training Op (for TRAIN)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"]
        )
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
    )









def main(unused_argv):
# Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    mnist_classifier = tf.estimator.Estimator(model_fn=model, model_dir="/tmp/mnist_model")


#image = get_image(1)
#image = np.array(image[0], image[1])
#image = np.array(image[0], np.array(image[1]))
#converting numpyarray to tensor usable

#newImage = tf.convert_to_tensor(newImage, tf.float32)


#print(train_set[1])
#print(len(train_set[1]))

#Running session
#sess = tf.Session()


#sess.run(outputLayer, feed_dict={input: image})


# Open the session

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50
    )

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True
    )
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook]
    )

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False
    )
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
    tf.app.run()