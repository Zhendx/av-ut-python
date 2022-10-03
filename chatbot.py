### Libreria para la creacion del servicio ###
#from flask import Flask, render_template, jsonify, request

### Librerias ###
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random
import pymongo

### Preparacion de los datos ###
# Declaracion de variables
words=[]
classes = []
documents = []
ignore_words = ['?', '!']
URI="--Aqui va la URI de MongoDB--"
# Conexion con la base de datos
client = pymongo.MongoClient(URI)
db = client.Bot
col = db["intents"]
# Bucle para tokenizar los datos
for intents in col.find():
    print (intents)
    for pattern in intents['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intents['tag']))
        if intents['tag'] not in classes:
            classes.append(intents['tag'])
# Lematizar los datos tokenizados
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Print de las clases y patrones tokenizados
print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique lemmatized words", words)
# Creacion de los pickle de los patrones y los tags
pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

# Codigo de entrenamiento
training = []
output_empty = [0] * len(classes)
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])
random.shuffle(training)
training = np.array(training)
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Datos de entrenamiento creados") 

# Crear modelo - 3 capas. Primera capa 128 neuronas, segunda capa 64 neuronas y 
# la tercera capa de salida contiene un número de neuronas
# igual al número de intenciones para predecir la intención de salida
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
# Optimizador SGD del modelo
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# Save del modelo
hist = model.fit(np.array(train_x), np.array(train_y), epochs=1000, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print("Modelo creado")

### Flask para la creacion del servicio ###
#app = Flask(__name__)
#Crea una ruta GET y POST del html
#@app.route('/', methods=["GET", "POST"])
#def index():
#    return 'Chatbot'
#Determina la direccion y el puerto en el que nuestro app va a ser ejecutado
#if __name__ == '__main__':
#    app.run(host='0.0.0.0', port='8888', debug=True)
