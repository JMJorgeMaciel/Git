from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
# 1. Datos para entrenar(texto, 1=positivo, 0 =egativo)

textos =[
    "me encarta este curso", "que bueno esta esto", "excelente trabajo",
    "odio esto", "que feo", "esto es orrible"
]
etiquetas = [1, 1, 1, 0, 0, 0]
# 2. Convertir textos a numeros

vectorizador = TfidfVectorizer()
x = vectorizador.fit_transform(textos)

# 3. Entrenar la IA
modelo = MultinomialNB()
modelo.fit(x, etiquetas)
# 4. Probarla

while True:
    frase = input("\nEscrbir algo")
    x_test = vectorizador.transform([frase])
    pred = modelo.predict(x_test)
    print("-> Es positivo :)" if pred[0]==1 else "-> Es NEGATIVO :(")