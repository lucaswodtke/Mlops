from flask import Flask, request, jsonify
import joblib
import numpy as np
import mlflow

app = Flask(__name__)

# Carregar o scaler e o modelo
scaler = joblib.load('scaler.pkl')
model = mlflow.sklearn.load_model("models:/CaliforniaHousingModel/Production")  # Substitua pela URI correta

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Extrair features do JSON
        features = [
            data['MedInc'],
            data['HouseAge'],
            data['AveRooms'],
            data['AveBedrms'],
            data['Population'],
            data['AveOccup'],
            data['Latitude'],
            data['Longitude']
        ]
        
        # Pré-processar
        input_data = np.array(features).reshape(1, -1)
        scaled_data = scaler.transform(input_data)
        
        # Fazer previsão
        prediction = model.predict(scaled_data)
        
        return jsonify({"predicted_price": float(prediction[0])})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)