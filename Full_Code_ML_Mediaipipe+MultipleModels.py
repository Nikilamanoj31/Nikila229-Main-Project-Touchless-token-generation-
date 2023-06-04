# -*- coding: utf-8 -*-


# !pip install mediapipe

# !unzip /content/drive/MyDrive/Dataset_NOZJ.zip

import cv2
import mediapipe as mp
import pandas as pd  
import os
import numpy as np 

def image_processed(file_path):
    
    # reading the static image
    hand_img = cv2.imread(file_path)

    # Image processing
    # 1. Convert BGR to RGB
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

    # 2. Flip the img in Y-axis
    img_flip = cv2.flip(img_rgb, 1)

    # accessing MediaPipe solutions
    mp_hands = mp.solutions.hands

    # Initialize Hands
    hands = mp_hands.Hands(static_image_mode=True,
    max_num_hands=2, min_detection_confidence=0.7)

    # Results
    output = hands.process(img_flip)

    hands.close()

    try:
        data = output.multi_hand_landmarks[0]
        #print(data)

        data = str(data)

        data = data.strip().split('\n')

        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

        without_garbage = []

        for i in data:
            if i not in garbage:
                without_garbage.append(i)

        clean = []

        for i in without_garbage:
            i = i.strip()
            clean.append(i[2:])

        for i in range(0, len(clean)):
            clean[i] = float(clean[i])

        
        return(clean)

    except:
        return(np.zeros([1,63], dtype=int)[0]) # here 63 is the no of columns 

def make_csv():
    
    mypath = '/content/train'
    file_name = open('final_datasetBSL.csv', 'a')

    for each_folder in os.listdir(mypath):
        if '._' in each_folder:
            pass

        else:
            for each_number in os.listdir(mypath + '/' + each_folder):
                if '._' in each_number:
                    pass
                
                else:
                    label = each_folder

                    file_loc = mypath + '/' + each_folder + '/' + each_number
                    print(file_loc)
                    data = image_processed(file_loc)
                    
                    try:
                        for i in data:
                            file_name.write(str(i))
                            file_name.write(',')

                        file_name.write(label)
                        file_name.write('\n')
                    
                    except:
                        file_name.write('0')
                        file_name.write(',')

                        file_name.write('None')
                        file_name.write('\n')
       
    file_name.close()
    print('Data Created !!!')

if __name__ == "__main__":
    make_csv()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score, precision_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv('/content/drive/MyDrive/american_words_WithNumbersReplaced.csv',on_bad_lines='skip')
df.columns = [i for i in range(df.shape[1])]
df = df.rename(columns={63: 'Output'}) # here 63 is no of columns of input
df

print("Uncleaned dataset shape =", df.shape)
# removing null values from our dataset

all_null_values = df[df.iloc[:, 0] == 0]
print("Number of null values =", len(all_null_values.index))
# dropping those null values from our dataset

df.drop(all_null_values.index, inplace=True)

df

print("Cleaned dataset shape =", df.shape)
X = df.iloc[:, :-1]
print("Features shape =", X.shape)

Y = df.iloc[:, -1]
print("Labels shape =", Y.shape)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
clssifier = RandomForestClassifier( criterion = 'entropy', random_state = 0)
clssifier.fit(x_train, y_train)
print("Training score =", clssifier.score(x_train, y_train))
y_pred=clssifier.predict(x_test)
print("Testing score=",accuracy_score(y_test,y_pred))

from sklearn.tree import DecisionTreeClassifier
lassifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
lassifier.fit(x_train, y_train)
print("Training score =", lassifier.score(x_train, y_train))
y_pred=lassifier.predict(x_test)
print("Testing score=",accuracy_score(y_test,y_pred))

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifie = GaussianNB()
classifie.fit(x_train, y_train)
print("Training score =", classifie.score(x_train, y_train))
y_pred=classifie.predict(x_test)
print("Testing score=",accuracy_score(y_test,y_pred))

from sklearn.svm import SVC
svm = SVC(kernel = 'rbf', random_state = 0)
svm.fit(x_train, y_train)
print("Training score =", svm.score(x_train, y_train))
y_pred=svm.predict(x_test)
print("Testing score=",accuracy_score(y_test,y_pred))

f1, recall, precision

# SVM model...

svm = SVC(C=50, gamma=0.1, kernel='rbf')
svm.fit(x_train, y_train)
print("Training score =", svm.score(x_train, y_train))
y_pred = svm.predict(x_test)
print("Testing score =", accuracy_score(y_test, y_pred))

#KNN model ..
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)
print("Training score =", classifier.score(x_train, y_train))
y_pred=classifier.predict(x_test)
print("Testing score=",accuracy_score(y_test,y_pred))

cf_matrix = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
precision = precision_score(y_test, y_pred, average='micro')

f1, recall, precision

labels = sorted(list(set(df['Output'])))
labels = [x.upper() for x in labels]
print(labels)
fig, ax = plt.subplots(figsize=(9, 9))

ax.set_title("Confusion Matrix - Indian Sign Language")

maping = sns.heatmap(cf_matrix, 
                     annot=True,
                     cmap = plt.cm.Blues, 
                     linewidths=.2,
                     xticklabels=labels,
                     yticklabels=labels, vmax=8,
                     ax=ax
                    )
maping

maping.figure.savefig("output.png")

import joblib
# Save the model as a pickle in a file
joblib.dump(classifier, 'mediapipe_trial_NOZJ_WITH_KNN_k=3.pkl')

file='MediaPipe_Model_Vernacular_WITH_KNN.h5' # Give any name with .h5 extension to save the model and use it anytime 
classifier.save(file)
from google.colab import files
files.download("/content/MediaPipe_Model_Vernacular_WITH_KNN.h5") # Download the .h5 file which contains the saved model weights information

