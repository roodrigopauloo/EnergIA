from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model, scaler = joblib.load("model/modelo_regressao.pkl")

def classificar_consumo(valor):
    if valor < 200:
        return "Baixo"
    elif valor < 400:
        return "Médio"
    else:
        return "Alto"


@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    classe = None

    if request.method == "POST":
        try:
            num_moradores = float(request.form["num_moradores"])
            area_m2 = float(request.form["area_m2"])
            temperatura_media = float(request.form["temperatura_media"])
            renda_familiar = float(request.form["renda_familiar"])
            uso_ar_condicionado = int(request.form["uso_ar_condicionado"])
            tipo_construcao = int(request.form["tipo_construcao"])
            equipamentos_eletro = float(request.form["equipamentos_eletro"])
            potencia_total_equipamentos = float(request.form["potencia_total_equipamentos"])
            
            densidade_habitacional = num_moradores / area_m2

            dados = np.array([[
                num_moradores,
                area_m2,
                temperatura_media,
                renda_familiar,
                uso_ar_condicionado,
                tipo_construcao,
                equipamentos_eletro,
                potencia_total_equipamentos,
                densidade_habitacional
            ]])

            dados_scaled = scaler.transform(dados)

            previsao = model.predict(dados_scaled)[0]

            resultado = round(previsao, 2)
            classe = classificar_consumo(resultado)

        except Exception as e:
            resultado = f"Erro: {e}"

    return render_template("index.html", resultado=resultado, classe=classe)


if __name__ == "__main__":
    app.run(debug=True)