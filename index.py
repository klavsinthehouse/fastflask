from flask import Flask, request, jsonify
from fastai.vision.all import *

app = Flask(__name__)
learn = load_learner('./appliancesModel.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(true)
    image_url = data['sessionInfo']['parameters']['image_url']
    img_data = requests.get(image_url).content
    img = PILImage.create(img_data)
    prediction = learn.predict(img)[0]
   
    json_response = {
        "sessionInfo": {
            "parameters": {
                "appliance": prediction,
            },
        },
    } 
    return jsonify(json_response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

