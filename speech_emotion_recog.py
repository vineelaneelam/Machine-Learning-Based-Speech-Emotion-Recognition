# Mounting Drive
"""

from google.colab import drive
drive.mount('/content/drive')

!mkdir SER

!cp "/content/drive/MyDrive/SER/SER_FOLDER.zip" "/content/SER"

cd /content/SER/

!unzip SER_FOLDER.zip

!ls "/content/SER/Emotion Speech Recognition"
!cd "/content/SER/Emotion Speech Recognition"
!ls

"""#  Installation of Dependencies

"""

# Commented out IPython magic to ensure Python compatibility.
# Provides a way of using operating system dependent functionality. 
import os

# LibROSA provides the audio analysis
import librosa
# Need to implictly import from librosa
import librosa.display

# Import the audio playback widget
import IPython.display as ipd
from IPython.display import Image

# Enable plot in the notebook
# %pylab inline
# %matplotlib inline
import matplotlib.pyplot as plt

# These are generally useful to have around
import numpy as np
import pandas as pd


# To build Neural Network and Create desired Model
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D #, AveragePooling1D
from keras.layers import Flatten, Dropout, Activation # Input, 
from keras.layers import Dense #, Embedding
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

"""# 3. Data Preparation

### Plotting the audio file's waveform and its spectrogram
"""

data, sampling_rate = librosa.load('/content/SER/Emotion Speech Recognition/Dataset/anger/anger016.wav')
# To play audio this in the jupyter notebook
ipd.Audio('/content/SER/Emotion Speech Recognition/Dataset/anger/anger016.wav')

len(data) , sampling_rate

plt.figure(figsize=(15, 5))
librosa.display.waveplot(data, sr=sampling_rate)

"""### Setup the Basic Paramter

### Converting Dataset in CSV format

it will cause easy operation on Dataset.
"""

dataset_path = os.path.abspath('./Dataset')
destination_path = os.path.abspath('./')
# To shuffle the dataset instances/records
randomize = True
# for spliting dataset into training and testing dataset
split = 0.8 #for training
# Number of sample per second e.g. 16KHz
sampling_rate = 20000 
emotions=["anger","disgust","fear","happy","neutral", "sad", "surprise"]\

!ls

# import required libraries
import os
import sys
import csv
import librosa
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
np.random.seed(42)

def create_meta_csv(dataset_path, destination_path):
        # Change dataset path accordingly
    DATASET_PATH = os.path.abspath(dataset_path)
    csv_path=os.path.join(destination_path, 'dataset_attr.csv')
    flist = []
    emotions=["anger","disgust","fear","happy","neutral", "sad", "surprise"]
    for root, dirs, files in os.walk(DATASET_PATH, topdown=False):
        for name in files:
            if (name.endswith('.wav')): 
                fullName = os.path.join(root, name)
                flist.append(fullName)

    split_format = str('/') if sys.platform=='linux' else str('\\')
    
    filenames=[]
    for idx,file in enumerate(flist):
        filenames.append(file.split(split_format)) 
        # print(filenames[idx])
    types=[]
    for idx,path in enumerate(filenames):
        types.append((flist[idx],emotions.index(path[-2]))) ##second last location has emotion name

    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows([("path","label")])
        writer.writerows(types)
    f.close()
    # change destination_path to DATASET_PATH if destination_path is None 
    if destination_path == None:
        destination_path = DATASET_PATH
        # write out as dataset_attr.csv in destination_path directory
        # if no error
    return True

def create_and_load_meta_csv_df(dataset_path, destination_path, randomize=True, split=None):

        if create_meta_csv(dataset_path, destination_path=destination_path):
          dframe = pd.read_csv(os.path.join('/content/SER/Emotion Speech Recognition', 'dataset_attr.csv'))

    # shuffle if randomize is True or if split specified and randomize is not specified 
    # so default behavior is split
        if randomize == True or (split != None and randomize == None):

        # shuffle the dataframe here
          dframe=dframe.sample(frac=1).reset_index(drop=True)
          pass

        if split != None:
          train_set, test_set = train_test_split(dframe, split)
          return dframe, train_set, test_set 
    
        return dframe

def train_test_split(dframe, split_ratio):
    # divide into train and test dataframes
    train_data= dframe.iloc[:int((split_ratio) * len(dframe)), :]
    test_data= dframe.iloc[int((split_ratio) * len(dframe)):,:]
    test_data=test_data.reset_index(drop=True) #reset index for test data
    return train_data, test_data

# loading dataframes using dataset module 


# To know more about "create_and_load_meta_csv_df" function and it's working, go to "./utils/dataset.py" script. 
df, train_df, test_df = create_and_load_meta_csv_df(dataset_path, destination_path, randomize, split)

print('Dataset samples  : ', len(df),"\nTraining Samples : ", len(train_df),"\ntesting Samples  : ", len(test_df))

"""# 4. Data Visualization

Let's understand what is our dataset.
"""

df.head()

print("Actual Audio : ", df['path'][0])
print("Labels       : ", df['label'][0])

"""
### Labels Assigned for emotions : 
- 0 : anger
- 1 : disgust
- 2 : fear
- 3 : happy
- 4 : neutral 
- 5 : sad
- 6 : surprise
"""

unique_labels = train_df.label.unique()
unique_labels.sort()
print("unique labels in Emtion dataset : ")
print(*unique_labels, sep=', ')
unique_labels_counts = train_df.label.value_counts(sort=False)
print("\n\nCount of unique labels in Emtion dataset : ")
print(*unique_labels_counts,sep=', ')

# Histogram of the classes
plt.bar(unique_labels, unique_labels_counts,align = 'center', width=0.6, color = 'c')
plt.xlabel('Number of labels', fontsize=16)
plt.xticks(unique_labels)
plt.ylabel('Count of each labels', fontsize=16)
plt.title('Histogram of the Labels', fontsize=16)
plt.show()

"""# 5. Data Pre-Processing

### Getting the features of audio files using librosa

Calculating MFCC, Pitch, magnitude, Chroma features.
"""

cd /content/SER/Emotion Speech Recognition/utils

from feature_extraction import get_features_dataframe
from feature_extraction import get_audio_features

cd /content/SER/Emotion Speech Recognition/

trainfeatures = pd.read_pickle('./features_dataframe/trainfeatures')
trainlabel = pd.read_pickle('./features_dataframe/trainlabel')
testfeatures = pd.read_pickle('./features_dataframe/testfeatures')
testlabel = pd.read_pickle('./features_dataframe/testlabel')

trainfeatures.shape

trainfeatures = trainfeatures.fillna(0)
testfeatures = testfeatures.fillna(0)

# By using .ravel() : Converting 2D to 1D e.g. (512,1) -> (512,). To prevent DataConversionWarning

X_train = np.array(trainfeatures)
y_train = np.array(trainlabel).ravel()
X_test = np.array(testfeatures)
y_test = np.array(testlabel).ravel()

y_train[:5]

# One-Hot Encoding
lb = LabelEncoder()

y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))

y_train[:5]

"""### Changing dimension for CNN model"""

x_traincnn =np.expand_dims(X_train, axis=2)
x_testcnn= np.expand_dims(X_test, axis=2)

x_traincnn.shape

"""# 6. Model Creation"""

model = Sequential()

model.add(Conv1D(256, 5,padding='same',
                 input_shape=(x_traincnn.shape[1],x_traincnn.shape[2])))
model.add(Activation('relu'))
model.add(Conv1D(128, 5,padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(y_train.shape[1]))
model.add(Activation('softmax'))
opt = keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

"""# 7. Training and Evaluation

### Removed the whole training part for avoiding unnecessary long epochs list
"""

cnnhistory=model.fit(x_traincnn, y_train, batch_size=16, epochs=400, validation_data=(x_testcnn, y_test))

"""### Loss and accuracy Vs Iterations"""

plt.plot(cnnhistory.history['accuracy'])
plt.plot(cnnhistory.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

"""### Saving the model"""

model_name = 'modser.h5'
save_dir = os.path.join(os.getcwd(), 'Trained_Models')
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

import json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

"""### Loading the model"""

# loading json and creating model
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/content/SER/Emotion Speech Recognition/Trained_Models/modser.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
score = loaded_model.evaluate(x_testcnn, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

"""# 8. Test Set Prediction

### Predicting emotions on the test data
"""

preds = loaded_model.predict(x_testcnn, 
                         batch_size=32, 
                         verbose=1)

preds

preds1=preds.argmax(axis=1)

preds1

abc = preds1.astype(int).flatten()

predictions = (lb.inverse_transform((abc)))

preddf = pd.DataFrame({'predictedvalues': predictions})
preddf[:10]

actual=y_test.argmax(axis=1)
abc123 = actual.astype(int).flatten()
actualvalues = (lb.inverse_transform((abc123)))

actualdf = pd.DataFrame({'actualvalues': actualvalues})
actualdf[:10]

finaldf = actualdf.join(preddf)

"""## Actual v/s Predicted emotions"""

finaldf[130:140]

finaldf.groupby('actualvalues').count()

finaldf.groupby('predictedvalues').count()

finaldf.to_csv('Predictions.csv', index=False)

"""# 9. Live Demonstration"""

demo_audio_path = '/content/SER/Emotion Speech Recognition/Dataset/disgust/disgust005.wav'
ipd.Audio(demo_audio_path)

demo_mfcc, demo_pitch, demo_mag, demo_chrom = get_audio_features(demo_audio_path,sampling_rate)

mfcc = pd.Series(demo_mfcc)
pit = pd.Series(demo_pitch)
mag = pd.Series(demo_mag)
C = pd.Series(demo_chrom)
demo_audio_features = pd.concat([mfcc,pit,mag,C],ignore_index=True)

demo_audio_features= np.expand_dims(demo_audio_features, axis=0)
demo_audio_features= np.expand_dims(demo_audio_features, axis=2)

demo_audio_features.shape

livepreds = loaded_model.predict(demo_audio_features, 
                         batch_size=32, 
                         verbose=1)

livepreds

# emotions=["anger","disgust","fear","happy","neutral", "sad", "surprise"]
index = livepreds.argmax(axis=1).item()
index

emotions[index]
