from flask import Flask, render_template, jsonify, request
    
#Importa las funciones o procedimientos 
import function

app = Flask(__name__)
app.config['SECRET_KEY'] = 'enter-a-very-secretive-key-190622'

#Crea una ruta GET y POST del html
@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html', **locals())

#Crea una ruta GET y POST del chatbot
@app.route('/chatbot', methods=["GET", "POST"])
def chatbotResponse():
    if request.method == 'POST':
        #Extrae la entrada
        the_question = request.form['question']
        #LLama a la funcion del procedimiento 
        response = function.chatbot_response(the_question.lower())
    return jsonify({"response": response })

#Crea una ruta GET y POST del audiobot
@app.route('/audiobot', methods=["GET", "POST"])
def audbotResponse():
    if request.method == 'POST':
        #Extrae la entrada
        the_question = request.form['audio']
        #LLama a la funcion del procedimiento 
        response = function.audiobot_response(the_question.lower())
    return jsonify({"response": response })

#Determina la direccion y el puerto en el que nuestro app va a ser ejecutado
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8888', debug=True)