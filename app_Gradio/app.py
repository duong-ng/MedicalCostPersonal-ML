import gradio as gr
import pickle, os, joblib
import numpy as np
import pandas as pd
import dill

with gr.Blocks() as demo:
    age = gr.Slider(minimum=0, maximum=100, step=1, label="Age")
    sex = gr.Radio(
        choices=["female", "male"],
        label='Select an option'    
    )
    bmi = gr.Textbox(label="BMI")
    children = gr.Textbox(label="Number of children")
    smoker = gr.Radio(
        choices=["yes", "no"],
        label='Smoker?'
    )
    region = gr.Dropdown(
        choices=["northeast", "northwest", "southeast", "southwest"],
        label='Choose a region'
    )
def predict(age, sex, bmi, children, smoker, region):
    # load the model and preprocessor
    model = 'D:\VietAI_FinalAssignment\model\xgbmodel.pkl'
    preprocessor = "D:\VietAI_FinalAssignment\preprocessor\preprocessor.pkl"
    if not os.path.exists(model) or not os.path.exists(preprocessor):
        raise FileNotFoundError("Model or preprocessor file not found.")
    model = joblib.load(model)
    preprocessor = joblib.load(preprocessor)
    # preprocess the input data
    input_data = pd.DataFrame({
            "age" : [age],
            "sex" : [sex],
            "bmi" : [bmi],
            "children" : [children],
            "smoker" : [smoker],
            "region" : [region]
        })
    input_data = preprocessor.transform(input_data)
    # make prediction
    prediction = model.predict(input_data)
    # return the prediction
    return f"Predicted medical charges cost is: ${prediction[0]:.2f}"
    

demo = gr.Interface(fn=predict, inputs=[age, sex, bmi, children, smoker, region], outputs="text")       
demo.launch(share=True, live=True, show_api=True)