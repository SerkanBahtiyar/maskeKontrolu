from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

#başlangıç ​​öğrenme oranını eğitilecek dönem sayısını belirlemek için parametreler
EPOCHS = 13
#epochs sayısını yüksek seçmek kesinliği artırır!
#tüm giriş verilerinin işlenmesindeki tek bir iterasyon.
VarsayilanDeger = 1e-4 
#ağı eğitmek için kullanacağımız optimize edici olan Adam optimizer için varsayılan değerdir
SB = 32
#Eğitim için ağımıza toplu görüntü aktaracağız(parti boyutunu kontrol eder)
#datasetin bulunduğu konumu belirttik
DIRECTORY = r"C:\Users\serka\OneDrive\Masaüstü\190508020_Serkan_Bahtiyar\dataset"
CATEGORIES = ["with_mask", "without_mask"]
#veri kümesi dizinimizdeki görüntülerin listesini alma, ardından veri listesini (yani görüntüler) ve sınıf görüntülerini başlatma
print("bilgi-> resimler yukleniyor...")
etiketler = []
veri = []
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)

    	veri.append(image)
    	etiketler.append(category)

# etiketlerde tek sıcak kodlama gerçekleştirin
lb = LabelBinarizer()
etiketler = lb.fit_transform(etiketler)
etiketler = to_categorical(etiketler)

veri = np.array(veri, dtype="float32")
etiketler = np.array(etiketler)

(trainX, testX, trainY, testY) = train_test_split(veri, etiketler,
	test_size=0.20, stratify=etiketler, random_state=42)

# veri büyütme için eğitim görüntü oluşturucusunu oluşturma işlemi
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# MobileNetV2 ağını yükle ve baş fc katman setlerinin pasif kalmasını sağlamak için kullanıldı
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# temel modelin üstüne yerleştirilecek modelin başını oluşturmak için kullanıldı
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

#baş fC modelini temel modelin üstüne yerleştirilir bu eğitilen gerçek model olacaktır
model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
	layer.trainable = False

# modelimizi derlemek için kullanıldı

print("bilgi-> model derlemesi...")
opt = Adam(lr=VarsayilanDeger, decay=VarsayilanDeger / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# ağın başını eğitmek

print("bilgi-> ağ başı eğitimi...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=SB),
	steps_per_epoch=len(trainX) // SB,
	validation_data=(testX, testY),
	validation_steps=len(testX) // SB,
	epochs=EPOCHS)
# test setiyle ilgili tahminlerde bulunma
print("bilgi->  ağ değerlendirmesi...")
predIdxs = model.predict(testX, batch_size=SB)

# test setindeki her bir görüntü için karşılık gelen en büyük tahmin edilen olasılığa sahip etiketi bulmamız gerekir
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

print("bilgi-> saving mask detector model...")
model.save("mask_detector.model")
