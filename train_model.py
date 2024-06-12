import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.2)

#Memuat Dataset dan Menyiapkan Data untuk Training
def load_data(data_dir):
    X = []
    y = []
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        if os.path.isdir(folder_path):
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                image = cv2.imread(image_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Konversi ke RGB
                results = hands.process(image_rgb)
                if results.multi_hand_landmarks:  # Jika tangan terdeteksi
                    hand_landmarks = results.multi_hand_landmarks[0]  # Ambil landmark tangan pertama
                    hand_features = np.array([[lmk.x, lmk.y, lmk.z] for lmk in hand_landmarks.landmark]).flatten()
                    X.append(hand_features)
                    y.append(folder_name)
    return np.array(X), np.array(y)

data_dir = "data"
X, y = load_data(data_dir)

#Pembuatan Model SVM
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = svm.SVC(kernel='linear')  # Model SVM dengan kernel linear
clf.fit(X_train, y_train)

# Simpan model menggunakan pickle
with open('svm_model.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)

#Evaluasi Model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
