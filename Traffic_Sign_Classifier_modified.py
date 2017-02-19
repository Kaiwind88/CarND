
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 
# In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 
# 
# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 
# 
# In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.
# 
# The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.
# 
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ---
# ## Step 0: Load The Data

# In[2]:

# Load pickled data
import pickle
from sklearn.utils import shuffle
# TODO: Fill this in based on where you saved the training and testing data

training_file = './traffic-signs-data/train.p'
testing_file = './traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train_ori, y_train_ori = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']


# ---
# 
# ## Step 1: Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
# 
# Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

# ### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

# In[3]:

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
import numpy as np
n_train = y_train_ori.shape[0]


# TODO: Number of testing examples.
n_test = y_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_test.shape[1:3]

# TODO: How many unique classes/labels there are in the dataset.
labels = np.append(y_train_ori,y_test)
uni_labels = np.unique(labels)
n_classes = np.shape(uni_labels)[0]

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# ### Include an exploratory visualization of the dataset

# Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc.
# 
# The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.
# 
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections.

# In[4]:

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import random
import time
# Visualizations will be shown in the notebook.
get_ipython().magic('matplotlib inline')

index = random.randint(0, len(X_train_ori))
image = X_train_ori[index].squeeze()

plt.figure(figsize=(1,1))
plt.imshow(image)
print('label #:',y_train_ori[index])

plt.figure()

plt.hist(labels,bins = n_classes,rwidth = 0.8)
plt.xlabel('Label of Signs')
plt.ylabel('Count of Signs')


# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# There are various aspects to consider when thinking about this problem:
# 
# - Neural network architecture
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.
# 
# **NOTE:** The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play!

# ### Pre-process the Data Set (normalization, grayscale, etc.)

# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.

# In[5]:

### Preprocess the data here. Preprocessing steps could include normalization, converting to grayscale, etc.
### Feel free to use as many code cells as needed.

def normalization(X):
#    a = 0
#    b = 0.9
#    pixel_range = 255  # pixel values range between 0 and 255
#    X = X - 128 # zero-centering the data from range 0-255
    
#    return a + X * (b-a) / pixel_range # shrink data to (-1,1)
    return (X-128)/128

# ### Split Data into Training, Validation and Testing Sets

# In[6]:

### Split the data into training/validation/testing sets here.
### Feel free to use as many code cells as needed.
def split_validation(X,y):
     from sklearn.model_selection import train_test_split
     X_train,X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
     return X_train, X_valid, y_train, y_valid
   


# ### Model Architecture

# In[7]:

### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf
from tensorflow.contrib.layers import flatten



# In[]:
#==============================================================================
#     ### to remove previous variables
#     ### Run when want to restore variables
#tf.reset_default_graph()
#   
#==============================================================================
# In[]:
### store weights and bias
### graph input

x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)
keep_prob = tf.placeholder(tf.float32) # probability to keep units for dropout

mu = 0
sigma = 0.1
weights = {
    # conv, Input = 32x32x3. Output = 28x28x6.
    'conv1_W': tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma)),
    # conv, Input = 14x14x6. Output = 10x10x16.
    'conv2_W': tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma)),
    # fully connected, Input = 400(5*5*16). Output = 120.
    'fc1_W': tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma)),
    # Fully Connected. Input = 120. Output = 84.
    'fc2_W': tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma)),
    # (class prediction) 84 inputs,43 outputs 
    'fc3_W':  tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma))
}                 


biases = {
    'conv1_b': tf.Variable(tf.zeros(6)),
    'conv2_b': tf.Variable(tf.zeros(16)),
    'fc1_b': tf.Variable(tf.zeros(120)),
    'fc2_b': tf.Variable(tf.zeros(84)),
    'fc3_b': tf.Variable(tf.zeros(n_classes))
}
# In[]:
def LeNet(x,weights, biases, keep_prob):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer


    # SOLUTION: Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1   = tf.nn.conv2d(x, weights['conv1_W'], strides=[1, 1, 1, 1], padding='VALID') + biases['conv1_b']
                      
    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)
     
    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2   = tf.nn.conv2d(conv1, weights['conv2_W'], strides=[1, 1, 1, 1], padding='VALID') + biases['conv2_b']
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)
    
    
    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1   = tf.matmul(fc0, weights['fc1_W']) + biases['fc1_b']
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    #Dropout
    fc1 = tf.nn.dropout(fc1,keep_prob) 
    
    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2    = tf.matmul(fc1, weights['fc2_W']) + biases['fc2_b']
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)
    
    #Dropout
    fc2 = tf.nn.dropout(fc2,keep_prob)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
    logits = tf.matmul(fc2, weights['fc3_W']) + biases['fc3_b']
    
    return logits


# ### Train, Validate and Test the Model

# A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
# sets imply underfitting. A high accuracy on the test set but low accuracy on the validation set implies overfitting.

# In[8]:

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.


rate = 0.001
EPOCHS = 1
BATCH_SIZE = 128

# In[]:
logits = LeNet(x,weights,biases,keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
# In[]
# ## Model Evaluation
# Evaluate how well the loss and accuracy of the model for a given dataset.
# 
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))

accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y,keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

real_prediction = tf.argmax(logits,1)

def evaluate_error(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        valid_error = sess.run(loss_operation, feed_dict={x: batch_x, y: batch_y,keep_prob: 1.0})
        valid_error += valid_error
    return valid_error / BATCH_SIZE



def test_new(X):
    test_label = sess.run(real_prediction, feed_dict = {x: X,keep_prob: 1.0})
    return test_label


# In[9]:
RESTORE = False  # restore previous model, don't train?


# In[]:  Prepare validation set
X_train, X_valid, y_train, y_valid = split_validation(X_train_ori,y_train_ori)
X_train = normalization(X_train)
X_valid = normalization(X_valid)
num_examples = len(X_train)
    
# In[]:
save_file = './traffic_sign_train_f1.ckpt'
saver = tf.train.Saver()
with tf.Session() as sess:
    if RESTORE:
        print('Restoring previously trained model...')
            # Restore previously trained model
        saver.restore(sess, save_file)
        with open('./accuracy_history.p', 'rb') as f:
            accuracy_history = pickle.load(f)
#            return accuracy_history
    else:
        print('Training model from scratch...')
        # Variable initialization
        sess.run(tf.global_variables_initializer())
        
    accuracy_history = []

# Record time elapsed for performance check
    last_time = time.time()
    train_start_time = time.time()
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        
       
        X_train, y_train = shuffle(X_train, y_train)
        
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y,keep_prob:.5})
        valid_error = evaluate_error(X_valid,y_valid)
        train_error = evaluate_error(X_train,y_train)
        validation_accuracy = evaluate(X_valid, y_valid)
        accuracy_history.append((validation_accuracy,valid_error,train_error))
        total_time = time.time() - train_start_time
                              
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print("Validation Error = {:.5f}".format(valid_error))
        print('Total elapsed time: %.2f sec (%.2f min)' % (total_time, total_time/60))
        print()
        
#    saver.save(sess, './traffic_sign')
    ## how to save and reload: https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/b516a270-8600-4f93-a0a3-20dfeabe5da6/concepts/c22dbf36-7215-483a-a397-d5f4f757d2d1
    
    saver.save(sess, save_file)
    print("Model saved")
    
    # Also save accuracy history
    print('Accuracy history saved at accuracy_history.p')
    with open('accuracy_history.p', 'wb') as f:
        pickle.dump(accuracy_history, f)

# In[]:
hist = np.transpose(np.array(accuracy_history))
plt.plot(range(EPOCHS), hist[0], 'or-')  # validation accuracy
plt.xticks(range(0,EPOCHS+1,5))
plt.title('Validation Accuracy over Epochs')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

plt.figure()
plt.plot(range(EPOCHS), hist[1], 'ob-')  # validation error
plt.plot(range(EPOCHS), hist[2], 'or-')  # validation error
plt.xticks(range(0,EPOCHS+1,5))
#plt.title('Validation Error over Epochs')
plt.legend(['valid error','train error'])
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.show()
#==============================================================================
# # In[]:
#     ### run  tf.reset_default_graph(), and then run variable defining first
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     saver.restore(sess, save_file)
# #    sess.run(tf.global_variables_initializer())
#     print('Bias:')
#     bb = sess.run(biases)
#     print(bb['fc1_b'])
#     
#==============================================================================
# In[ ]:
## Evaluation of the Model
with tf.Session() as sess:
#    saver.restore(sess, tf.train.latest_checkpoint('.'))
#    saver.restore(sess,save_file)
    saver.restore(sess,save_file)
    X_test = normalization(X_test)
    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))


# In[18]:



# ---
# 
# ## Step 3: Test a Model on New Images
# 
# To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

# ### Load and Output the Images

# In[16]:

### Load the images and plot them here.
### Feel free to use as many code cells as needed.
#import PIL
from PIL import Image
import glob
import csv
X_new = []

dirt = './GTSRB_Final_Test_Images/GTSRB/Final_Test/Images/'
dirt_y = './GTSRB_Final_Test_Images/GTSRB/Final_Test/'
a = glob.glob(dirt + '*.ppm')
 ## read labels of new image from file
y_new_store = []
with open(dirt_y+'GT-final_test.csv', newline='') as f:
    reader = csv.reader(f, delimiter=';')
#    print(np.array(reader)[0])
    for row in reader:
        y_new_store.append(np.array(row)[7])
    del y_new_store[0]
# In[]
y_new = []
image_no = 5
test_no = np.array(random.sample(range(len(y_new_store)),image_no))


for ind in test_no:
   
    y_new.append(y_new_store[ind])
    image_new = Image.open(a[ind])
    image_new = image_new.resize((32, 32), Image.ANTIALIAS)
#    image_new = np.array(list(image_new.getdata()), dtype='uint8')
    X_new.append(np.reshape(image_new, (32, 32, 3)))
    plt.figure()
#    print('This image is:', type(X_new), 'with dimesions:', X_new.shape)
    plt.imshow(image_new)

X_new = np.array(X_new,dtype = float)


#y_new1 = np.array([16,1,38,33,11,38])
# ### Predict the Sign Type for Each Image

# In[15]:

### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.

#tf.reset_default_graph()
#saver = tf.train.Saver()
with tf.Session() as sess:

    saver.restore(sess,save_file)
    
    X_new = normalization(X_new)
    
    label_new= test_new(X_new)
    print("Predicted:", label_new, "True:", y_new)
    validation_accuracy = evaluate(X_new, y_new)

    print("Validation Accuracy = {:.3f}".format(validation_accuracy))

# ### Analyze Performance

# In[ ]:

### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.


# ### Output Top 5 Softmax Probabilities For Each Image Found on the Web

# For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 
# 
# The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.
# 
# `tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.
# 
# Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tk.nn.top_k` is used to choose the three classes with the highest probability:
# 
# ```
# # (5, 6) array
# a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
#          0.12789202],
#        [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
#          0.15899337],
#        [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
#          0.23892179],
#        [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
#          0.16505091],
#        [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
#          0.09155967]])
# ```
# 
# Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:
# 
# ```
# TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
#        [ 0.28086119,  0.27569815,  0.18063401],
#        [ 0.26076848,  0.23892179,  0.23664738],
#        [ 0.29198961,  0.26234032,  0.16505091],
#        [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
#        [0, 1, 4],
#        [0, 5, 1],
#        [1, 3, 5],
#        [1, 4, 3]], dtype=int32))
# ```
# 
# Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.

# In[ ]:

### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
#init_op = tf.global_variables_initializer() 
with tf.Session() as sess:
#    sess.run(init_op)
    saver.restore(sess,save_file)
    temp_logits = sess.run(logits,feed_dict = {x: X_new,keep_prob:1.0})
    prob_X = softmax(temp_logits)
    
    top_k = sess.run(tf.nn.top_k(prob_X, k=5))
    print(top_k)

# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the IPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run. You can then export the notebook by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 

# ### Project Writeup
# 
# Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 

