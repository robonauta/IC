from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical

from extractor import tidy

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(41, 41, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

model.summary()

(train_images, train_labels) = tidy('00000085.jpg', '00000085_label.png')
(test_images, test_labels) = tidy('00000088.jpg', '00000088_label.png')

train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)

'''
test_loss, test_acc = model.evaluate(test_images, test_labels)

print(test_acc)
'''