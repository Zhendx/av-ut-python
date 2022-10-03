### Librerias  de python ###
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import pymongo

# Conexion con la base de datos
client = pymongo.MongoClient("mongodb+srv://bot:bot@cluster0.yixc3.mongodb.net/Bot?retryWrites=true&w=majority")
db = client.Bot
col = db["intents"]

# Docs generados a partir del entrenamiento del chat Bot
model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

#--- Tokeniza y lematiza el mensaje ---#
def clean_up_sentence(sentence):
    #Divide una frase en tokens o palabras.
    sentence_words = nltk.word_tokenize(sentence) 
    #Limpia los tokens o palabras de los caracteres especiales
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

#--- Devuelve la matriz de palabras de la bolsa: 0 o 1 para cada palabra de la bolsa que existe en la frase ---#
def bow(sentence, words, show_details=True):  
    #Limpia el mensaje de entrada
    sentence_words = clean_up_sentence(sentence) 
    #Bolsa de palabras - matriz de N palabras - matriz de vocabulario
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                #Asigna 1 si la palabra actual está en la posición del vocabulario  
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

#--- Devuelve una lista de probabilidades ---#
def predict_class(sentence, model):
    #Devuelve la matriz de palabras de la bolsa: 0 o 1 para cada palabra de la bolsa que existe en la frase
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    #Ordenar por probabilidad
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

#--- Devuelve la respuesta ---#
def getResponse(ints):
    #Asigna en tag, todos los tags de la lista de probabilidades 
    tag = ints[0]['intent'] 
    #Compara los tags de las probabilidades con los tags de la base de datos
    for i in col.find():
        print (i['tag'])
        if(i['tag'] == tag):
            #Escoge una respuesta al azar
            result = random.choice(i['responses'])
            break
    return result

#--- LLama a las funciones necesarias para determinar una respuesta ---#
def chatbot_response(msg):
    #Devuelve una lista de probabilidades
    ints = predict_class(msg, model)
    #Devuelve la respuesta
    res = getResponse(ints)
    return res

#--- Llama a las dunciones necesarias para determinar una respuesta ---#
def audio_response(msg):
    #Devuelve una lista de probabilidades
    ints = predict_class(msg, model)
    #Devuelve la respuesta
    res = getResponse(ints)
    return res