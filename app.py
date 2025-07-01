import pickle 
import gradio as gr

vectorizer = pickle.load(open('vectorizer.pkl','rb'))
model_category = pickle.load(open('model_category.pkl','rb'))
model_transactionType = pickle.load(open('model_transactionType.pkl','rb'))


def predict(text):
    vector = vectorizer.transform([text])
    category = model_category.predict(vector)[0]
    transaction_type = model_transactionType.predict(vector)[0]
    return category, transaction_type


print(predict("paid rs 20 to amazon"))

iface = gr.Interface(fn=predict,inputs="text", outputs="label",title="Transaction Classifier")
iface.launch()