# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing import sequence
# from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import Embedding
from tensorflow.keras.utils import to_categorical
import numpy as np
import pickle
import streamlit as st
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from keras.callbacks import Callback
m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: rgb(30, 200, 49);font-size:20px;height:2em;width:10em;position:relative;left:30%;}

</style>""", unsafe_allow_html=True)

st.header('Country Origin')
st.image('earth.jpg', width=2, use_column_width=True)
name = st.text_input(label='Name', placeholder='enter name')
name = [name]
np.random.seed(42)
model1 = load_model("model.h5")

max_len = 36
vocab_size = 53

def char_encoded_representation(data,tokenizer,vocab_size,max_len):
	char_index_sentences = tokenizer.texts_to_sequences(data)
	sequences = [to_categorical(x, num_classes=vocab_size) for x in char_index_sentences]
	X = sequence.pad_sequences(sequences, maxlen=max_len)
	return X

# loading
with open('tokenizer.pickle', 'rb') as handle:
	tok = pickle.load(handle)

# # Model Specification
# check_new_names = ['lalitha','tyson','shailaja','shyamala','vishwanathan','ramanujam','conan','kryslovsky',
# 'ratnani','diego','kakoli','shreyas','brayden','shanon']

if st.button('Find'):

	X_predict = char_encoded_representation(name,tok,vocab_size,max_len)
	# print(X_predict.shape)
	predictions_prob = model1.predict(X_predict)
	predictions = np.array(predictions_prob)
	predictions[predictions > 0.5] = 1
	predictions[predictions <= 0.5] = 0
	predictions = np.squeeze(predictions)
	prediction = np.where(predictions==0,'Indian', 'Non Indian')
	# predictions_lstm_char = le.inverse_transform(list(predictions.astype(int)))
	# test = pd.DataFrame({'names':check_new_names,'predictions_lstm_char':predictions_lstm_char})
	# st.write(prediction)
	print(type(prediction))
	prediction = str(prediction)
	print(type(prediction))
	st.title(prediction)


