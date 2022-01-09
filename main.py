from flask import Flask, request, jsonify,render_template
import os
import numpy as np
from flask_cors import CORS, cross_origin
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
# from paths import Paths

# paths = Paths()

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    txt = request.json['data']
    tokenizer = Tokenizer(num_words=8000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    seq = tokenizer.texts_to_sequences(txt)
    padded = pad_sequences(seq, maxlen=130)


    # Restore the weights

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    pred = loaded_model.predict(padded)
    labels = ['entertainment', 'bussiness', 'science/tech', 'health']
    print(pred, labels[np.argmax(pred)])
    return jsonify({"PREDICTED CLASS" : labels[np.argmax(pred)]})



if __name__ ==  "__main__":
    # Flask Server
    app.run(host='0.0.0.0', port=5000, debug=True)
