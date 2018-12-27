from keras.utils import to_categorical
import numpy as np
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

label2emotion = {0: "sad", 1: "happy", 2: "angry"}

class MicroF1ModelCheckpoint(ModelCheckpoint):

    def __init__(self, filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super().__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)

    def on_epoch_end(self, epoch, logs={}):

        val_predict = (np.asarray(self.model.predict([np.array(self.validation_data[0]), np.array(self.validation_data[1])])))
        val_targ = self.validation_data[2]

        discretePredictions = to_categorical(val_predict.argmax(axis=1), num_classes=len(label2emotion))
        #discretePredictions = np.around(val_predict)

        truePositives = np.sum(discretePredictions * val_targ, axis=0)
        falsePositives = np.sum(np.clip(discretePredictions - val_targ, 0, 1), axis=0)
        falseNegatives = np.sum(np.clip(val_targ - discretePredictions, 0, 1), axis=0)

        truePositives = truePositives[1:].sum()
        falsePositives = falsePositives[1:].sum()
        falseNegatives = falseNegatives[1:].sum()

        microPrecision = truePositives / (truePositives + falsePositives)
        microRecall = truePositives / (truePositives + falseNegatives)

        # microF1 = (2 * microRecall * microPrecision) / (microPrecision + microRecall) if (microPrecision + microRecall) > 0 else 0
        microF1 = 2 / (1.0 / microPrecision + 1.0 / microRecall)

        print("- microF1: %f - true_positives: %d - false_positives %d - false_negatives: %d" % (
        microF1, int(truePositives), int(falsePositives), int(falseNegatives)))

        logs['f1_micro_score'] = microF1
        super().on_epoch_end(epoch, logs=logs)
        return