#import svm_prediction
import os
import csv
import json
import sys
sys.path.append('../../../source/modeling/')
import svm_constructiveness_predictor
from svm_constructiveness_predictor import ConstructivenessSVM
import bilstm_constructiveness_predictor
from bilstm_constructiveness_predictor import Constructiveness_biLSTM
#ROOT = '/home/ling-discourse-lab/Varada/'
ROOT = '/Users/vkolhatk/Data/Constructiveness_public/'

svm_model_path = ROOT + 'output/intermediate_output/models/svm_model.pkl'
bilstm_model_path = ROOT + 'output/intermediate_output/models/NYT_picks_train_SFU_test.tflearn'

# Load models
#bilstm = Constructiveness_biLSTM(bilstm_model_path)
csvm = ConstructivenessSVM(svm_model_path)

from flask import Flask
from flask import render_template, request, jsonify

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_prediction", methods=["GET", "POST"])
def get_prediction():
    text = request.args.get("result")
    selected_model = request.args.get("model")
    label = 'Default'
    print("SELECTED MODEL: ", selected_model)
    if selected_model == "svm":
        print("You selected SVM")
        #prediction = svm_prediction.predict(text, model_path)[0]
        label = csvm.predict_svm(text)
    elif selected_model == "lstm":
        print("You selected lstm")
        label = bilstm.predict_bilstm(text)
        # do whatever needs to be done for lstm
    else:
        print("Please select a model first.")
        return jsonify(predicted_label="Please select a model first")

    #label = "Constructive"

    print(text)
    return jsonify(predicted_label="According to our " + selected_model + " model the comment is likely to be perceived as " + label.upper() + ".")

@app.route("/select_model", methods=["GET", "POST"])
def select_model():
    model = request.args.get('result')

    print("THIS IS THE MODEL: ", model)
    return jsonify(resp='You selected: ' + model)


@app.route("/get_feedback", methods=["GET", "POST"])
def feedback():
    text = request.args.get('result')
    label = request.args.get('label')
    comments = request.args.get('comments')

    file_exists = os.path.isfile('feedback.csv')
    with open('feedback.csv', 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['Text', 'Label', 'Comments'])
        writer.writerow([text, label, comments])
        return jsonify(feedback='Thank you for your feedback!')


if __name__ == '__main__':
    app.run(host='localhost', port=9999)

