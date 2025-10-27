import streamlit as st
from joblib import load

model = load('data/iris_model.joblib')
st.title("IRIS FLOWER PREDICTOR APP")

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length", value=None, placeholder="Example: 2, 3, 4,."
    )
    
    sepal_width = st.number_input(
        "Sepal Width", value=None, placeholder="Example: 2, 3, 4,."
    )

with col2:
    petal_length = st.number_input(
        "Petal Length", value=None, placeholder="Example: 2, 3, 4,."
    )
    petal_width = st.number_input(
        "Petal Width", value=None, placeholder="Example: 2, 3, 4,."
    )

c1, c2, c3 = st.columns(3)
with c2:
    if st.button("Predict"):
        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        if prediction[0] == 0:
            st.write("Flower predicted: Setosa")
            st.image("images/iris_setosa.png")
        elif prediction[0] == 1:
            st.write("Flower predicted: Versicolour")
            st.image("images/iris_versicolor.png")
        else:
            st.write("Flower predicted: Virginica")
            st.image("images/iris_virginica.png")
    # - 0: Iris Setosa
  # - 1: Iris Versicolour
  # - 2: Iris Virginica




