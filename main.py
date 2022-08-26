from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import Callback
np.random.seed(42)

# data_url = "https://raw.githubusercontent.com/ashavish/name-nationality/master/data/name_data.csv"
name_data = pd.read_csv('name_data.csv')
model = load_model("model.h5")
check_new_names = ['lalitha','tyson','shailaja','shyamala','vishwanathan','ramanujam','conan','kryslovsky',
'ratnani','diego','kakoli','shreyas','brayden','shanon']

from sklearn.model_selection import train_test_split
X = name_data['name'].astype(str)
Y = name_data['label']
train_names,test_names,train_labels,test_labels = train_test_split(X,Y,test_size=0.2,random_state =42,stratify=Y)

def char_encoded_representation(data,tokenizer,vocab_size,max_len):
	char_index_sentences = tokenizer.texts_to_sequences(data)
	sequences = [to_categorical(x, num_classes=vocab_size) for x in char_index_sentences]
	X = sequence.pad_sequences(sequences, maxlen=max_len)
	return X

max_len = max([len(str(each)) for each in train_names])
# mapping = get_char_mapping(train_names)
# vocab_size = len(mapping)

tok = Tokenizer(char_level=True)
tok.fit_on_texts(train_names)
import pickle

# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tok, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# # loading
# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)
vocab_size = len(tok.word_index) + 1
X_train = char_encoded_representation(train_names,tok,vocab_size,max_len)
X_train.shape

X_test = char_encoded_representation(test_names,tok,vocab_size,max_len)
X_test.shape

le = LabelEncoder()
le.fit(train_labels)
y_train = le.transform(train_labels)
y_test = le.transform(test_labels)

# Model Specification


# def build_model(hidden_units,max_len,vocab_size):
# 	model = Sequential()
# 	# model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
# 	model.add(LSTM(hidden_units,input_shape=(max_len,vocab_size)))
# 	model.add(Dense(1, activation='sigmoid'))
# 	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 	print(model.summary())
# 	return model
#
# class myCallback(Callback):
# 	def __init__(self,X_test,y_test):
# 		self.X_test = X_test
# 		self.y_test = y_test
# 	def on_epoch_end(self, epoch, logs={}):
# 		loss,acc = model.evaluate(self.X_test, self.y_test, verbose=0)
# 		print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
#
# model = build_model(100,max_len,vocab_size)
# model.fit(X_train, y_train, epochs=5, batch_size=64,callbacks=myCallback(X_test,y_test))
vocab_size = 53
max_len = 36
X_predict = char_encoded_representation(check_new_names,tok,vocab_size,max_len)

predictions_prob = model.predict(X_predict)
predictions = np.array(predictions_prob)
predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0
predictions = np.squeeze(predictions)
predictions_lstm_char = le.inverse_transform(list(predictions.astype(int)))
test = pd.DataFrame({'names':check_new_names,'predictions_lstm_char':predictions_lstm_char})
print(test)