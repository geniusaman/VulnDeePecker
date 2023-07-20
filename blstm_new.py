from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, LeakyReLU, Dropout
from keras.optimizers import Adamax
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix


class BLSTM:
    def __init__(self, data, name="", batch_size=64):
        vectors = np.stack(data["vector"].values)
        labels = data["val"].values

        # Perform a stratified train-test split
        X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.2, stratify=labels)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = to_categorical(y_train)
        self.y_test = to_categorical(y_test)
        self.name = name
        self.batch_size = batch_size
        self.class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)

        model = Sequential()
        model.add(Bidirectional(LSTM(300), input_shape=(vectors.shape[1], vectors.shape[2])))
        model.add(Dense(300))
        model.add(LeakyReLU())
        model.add(Dropout(0.5))
        model.add(Dense(300))
        model.add(LeakyReLU())
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

        # Lower learning rate to prevent divergence
        adamax = Adamax(lr=0.002)
        model.compile(adamax, 'categorical_crossentropy', metrics=['accuracy'])
        self.model = model


    
    def train(self):
      # Compute class weights for imbalanced classes
      y_integers = np.argmax(self.y_train, axis=1)
      class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_integers), y=y_integers)
      class_weight_dict = dict(enumerate(class_weights))

      self.model.fit(self.X_train, self.y_train, batch_size=self.batch_size, epochs=4, class_weight=class_weight_dict)
      self.model.save_weights(self.name + "_model.h5")


    def test(self):
        self.model.load_weights(self.name + "_model.h5")
        values = self.model.evaluate(self.X_test, self.y_test, batch_size=self.batch_size)
        print("Accuracy is...", values[1])
        predictions = (self.model.predict(self.X_test, batch_size=self.batch_size)).round()

        tn, fp, fn, tp = confusion_matrix(np.argmax(self.y_test, axis=1), np.argmax(predictions, axis=1)).ravel()
        print('False positive rate is...', fp / (fp + tn))
        print('False negative rate is...', fn / (fn + tp))
        recall = tp / (tp + fn)
        print('True positive rate is...', recall)
        precision = tp / (tp + fp)
        print('Precision is...', precision)
        print('F1 score is...', (2 * precision * recall) / (precision + recall))
