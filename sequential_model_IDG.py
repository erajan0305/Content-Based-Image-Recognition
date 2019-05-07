import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import merge,core
from keras.engine.topology import Input
#%%

PATH = os.getcwd()
print(PATH
# Define data path
data_path = PATH + '/Final Implementation'
data_dir_list = os.listdir(data_path)
print(data_path,data_dir_list)

img_rows=128
img_cols=128
num_channel=3
num_epoch=20

# Define the number of classes
num_classes = 5

img_data_list=[]
i=0
#Pre-Processing
for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('\nLoading the images of dataset-'+'{}'.format(dataset))
    for img in img_list:
        input_img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
        #input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

        input_img_resize=cv2.resize(input_img,(128,128))
        img_data_list.append(input_img_resize)
        i+=1
    print(dataset, i)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    i=0
        
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print ("-1",img_data.shape)

if num_channel==1:
	if K.image_dim_ordering()=='th':
		img_data= np.expand_dims(img_data, axis=1) 
		print ("-2",img_data.shape)
	else:
		img_data= np.expand_dims(img_data, axis=4) 
		print ("-3",img_data.shape)
		
else:
	if K.image_dim_ordering()=='th':
		img_data=np.rollaxis(img_data,3,1)
		print ("-4",img_data.shape)		
#%%
# Assigning Labels

# Define the number of classes
num_classes = 5

num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:1000]=0
labels[1000:1717]=1
labels[1717:2717]=2
labels[2717:3717]=3
labels[3717:]=4
	  
names = ['Aeroplanes','Bikes',"Boats","Cars","Dogs"]
	  
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

#%%
# Defining the model
input_shape=img_data[0].shape
#input_shape=Input((1, img_rows, img_cols))
					
model = Sequential()

#model.add(Convolution2D(16, 3,3,border_mode='same',input_shape=input_shape)) #1s
model.add(Convolution2D(16, 3,3,border_mode='same',input_shape=input_shape)) #1st
model.add(Activation('relu'))
model.add(BatchNormalization(axis=1))
model.add(Convolution2D(16, 3, 3)) # 2nd
model.add(Activation('relu'))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(32, 3,3,input_shape=input_shape)) #1st
model.add(Activation('relu'))
model.add(BatchNormalization(axis=1))
model.add(Convolution2D(32, 3, 3)) # 2nd
model.add(Activation('relu'))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3)) #3rd
model.add(Activation('relu'))
model.add(BatchNormalization(axis=1))
model.add(Convolution2D(64, 3, 3)) #4th
model.add(Activation('relu'))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.25))

model.add(Convolution2D(128, 3, 3)) #5th
model.add(Activation('relu'))
model.add(BatchNormalization(axis=1))
model.add(Convolution2D(128, 3, 3)) #6th
model.add(Activation('relu'))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=1))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

adam=Adam(lr=1e-4)
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])
model.compile(loss='categorical_crossentropy', optimizer=adam,metrics=["accuracy"])


# Viewing model_configuration
model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape			
model.layers[0].output_shape			
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
#print(model.layers[0].get_weights()[0])
model.layers[0].trainable

#%%
# Training
from keras import callbacks
#ImageDataGenerator Used.
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,horizontal_flip=True, fill_mode="nearest")

checkpointer = callbacks.ModelCheckpoint(filepath='_best_weights.h5', verbose=1, monitor='val_loss', mode='auto', save_best_only=True)
hist = model.fit_generator(aug.flow(X_train, y_train, batch_size=16), steps_per_epoch=len(X_train) // 16,epochs=20, verbose=1,  callbacks=[checkpointer], validation_data=(X_test, y_test))

# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(num_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

#%%

# Evaluating the model
score = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
print('Test Loss:', score[0])
print('Test accuracy:', score[1])

# Testing a new image
test_image = cv2.imread('C:/Users/PARVA SHAH/Pictures/11.jpg')
#test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
test_image=cv2.resize(test_image,(128,128))

test_image = np.array(test_image)
test_image = test_image.astype('float32')
test_image /= 255
print (test_image.shape)
   
if num_channel==1:
	if K.image_dim_ordering()=='th':
		test_image= np.expand_dims(test_image, axis=0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=3) 
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
		
else:
	if K.image_dim_ordering()=='th':
		test_image=np.rollaxis(test_image,2,0)
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
	else:
		test_image= np.expand_dims(test_image, axis=0)
		print (test_image.shape)
		
# Predicting the test
x=model.predict_classes(test_image)
if x==0 :
    print("5---Aeroplane")
elif x==1 :
    print("5---Bike")
elif x==2 :
    print("5---Boats")
elif x==3 :
    print("5---Cars")
elif x==4 :
    print("5---Dogs")
    
        

#%%

# Visualizing the intermediate layer

#
def get_featuremaps(model, layer_idx, X_batch):
	get_activations = K.function([model.layers[0].input, K.learning_phase()],[model.layers[layer_idx].output,])
	activations = get_activations([X_batch,0])
	return activations

layer_num=3
filter_num=0

activations = get_featuremaps(model, int(layer_num),test_image)

print ("6",np.shape(activations))
feature_maps = activations[0][0]      
print ("7",np.shape(feature_maps))

if K.image_dim_ordering()=='th':
	feature_maps=np.rollaxis((np.rollaxis(feature_maps,2,0)),2,0)
#print (feature_maps.shape)

fig=plt.figure(figsize=(16,16))
plt.imshow(feature_maps[:,:,filter_num],cmap='gray')
plt.savefig("featuremaps-layer-{}".format(layer_num) + "-filternum-{}".format(filter_num)+'.jpg')

num_of_featuremaps=feature_maps.shape[2]
fig=plt.figure(figsize=(16,16))	
plt.title("featuremaps-layer-{}".format(layer_num))
subplot_num=int(np.ceil(np.sqrt(num_of_featuremaps)))
for i in range(int(num_of_featuremaps)):
	ax = fig.add_subplot(subplot_num, subplot_num, i+1)
	#ax.imshow(output_image[0,:,:,i],interpolation='nearest' ) #to see the first filter
	ax.imshow(feature_maps[:,:,i],cmap='gray')
	plt.xticks([])
	plt.yticks([])
	plt.tight_layout()
plt.show()
fig.savefig("featuremaps-layer-{}".format(layer_num) + '.jpg')

#%%
# Printing the confusion matrix
from sklearn.metrics import classification_report,confusion_matrix
import itertools

Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)
y_pred = model.predict_classes(X_test)
target_names = ['Aeroplanes','Bikes',"Boats","Cars","Dogs"]
					
print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))

print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))


# Plotting the confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("8--Normalized confusion matrix")
    else:
        print('8--Confusion matrix, without normalization')

    print("9",cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = (confusion_matrix(np.argmax(y_test,axis=1), y_pred))

np.set_printoptions(precision=2)

plt.figure()

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix')
plt.show()

#%%
# Saving and loading model and weights
model.save('model.hdf5')
loaded_model=load_model('model.hdf5')
print(("4",loaded_model.predict(test_image)))
print("5",loaded_model.predict_classes(test_image))
