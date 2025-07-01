from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

modelo = joblib.load("modelo_postres.pkl")
encoders = joblib.load("encoders.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/resultado', methods=['POST'])
def resultado():
    info = {
        'tipo': request.form['tipo'],
        'hambre': int(request.form['hambre']),
        'cansancio': int(request.form['cansancio']),
        'restaurante': int(request.form['restaurante']),
        'celebracion': int(request.form['celebracion']),
        'comido_platano': int(request.form['comido_platano']),
        'estacion': request.form['estacion']}
    
    datos = pd.DataFrame([info])
    datos['tipo'] = encoders['tipo'].transform(datos['tipo'])
    datos['estacion'] = encoders['estacion'].transform(datos['estacion'])

    y_pred = modelo.predict(datos)
    postre = encoders['postre'].inverse_transform(y_pred)[0]
    imagen = postre.lower().replace(" ", "_") + ".jpg"
    if postre != "mini magnum":
        postre = postre.capitalize()

    return render_template('resultado.html', postre=postre, imagen=imagen)

if __name__ == "__main__":
    app.run(debug=True)