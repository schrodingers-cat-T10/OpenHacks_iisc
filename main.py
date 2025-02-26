import re
import PyPDF2 as pdf
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_nomic import NomicEmbeddings
from tensorflow.keras.models import load_model
import numpy as np
import pickle

def pdf_to_text(pdf_path: str):
    text = ""
    try:
        reader = pdf.PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

def text_to_chunks(text: str):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks

def embedding_vector(chunks: list):
    embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

def create_conversation_chain(vectorstore):
    llm = Ollama(model="deepseek-r1:1.5b") 
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=vectorstore.as_retriever()
    )
    return conversation_chain

def extract_blood_report_values(conversation_chain):
    blood_report_keys = [
        "Cholesterol", "Hemoglobin", "Platelets", "White Blood Cells (WBC)", "Red Blood Cells (RBC)",
        "Hematocrit", "Mean Corpuscular Volume (MCV)", "Mean Corpuscular Hemoglobin (MCH)",
        "Mean Corpuscular Hemoglobin Concentration (MCHC)", "Insulin", "BMI (Body Mass Index)",
        "Systolic Blood Pressure (SBP)", "Diastolic Blood Pressure (DBP)", "Triglycerides",
        "HbA1c (Glycated Hemoglobin)", "LDL (Low-Density Lipoprotein) Cholesterol",
        "HDL (High-Density Lipoprotein) Cholesterol", "ALT (Alanine Aminotransferase)",
        "AST (Aspartate Aminotransferase)", "Heart Rate", "Creatinine", "Troponin",
        "C-reactive Protein (CRP)", "Disease"
    ]
    blood_report_values = {key: 0 for key in blood_report_keys}

    for key in blood_report_keys:
        query = f"What is the value of {key} in the blood report?"
        try:
            response = conversation_chain.invoke(query)
            if response and "answer" in response:
                answer = response["answer"]
                match = re.search(r"[\d,.]+", answer)
                if match:
                    value = float(match.group(0).replace(",", ""))
                    blood_report_values[key] = value
        except Exception as e:
            print(f"Error extracting value for {key}: {e}")

    return [blood_report_values[key] for key in blood_report_keys]

def prediction(model, scaler, encoder, data):
    try:
        data = np.array(data).reshape(1, -1)
        
        if not hasattr(scaler, "transform"):
            print("Scaler object is invalid.")
            return None
        
        scaled_data = scaler.transform(data)
        reshaped_data = scaled_data.reshape(-1, 6, 4, 1)  
        
        prediction = model.predict(reshaped_data)
        return encoder.inverse_transform(prediction.argmax(axis=1))
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def main():
    try:
        model = load_model("E:\\jahan\\iisc\\hello.keras")
        with open("E:\\jahan\\iisc\\scaler.pkl", "rb") as scaling:
            scaler = pickle.load(scaling)
        with open("E:\\jahan\\iisc\\encoder.pkl", "rb") as encodings:
            encoder = pickle.load(encodings)
    except Exception as e:
        print(f"Error loading model or preprocessing files: {e}")
        return

    pdf_path = "E:\\jahan\\iisc\\lzQ3Phvl24.pdf"
    text = pdf_to_text(pdf_path)
    if not text:
        print("No text extracted from PDF. Exiting.")
        return

    chunks = text_to_chunks(text)
    vectorstore = embedding_vector(chunks)
    conversation_chain = create_conversation_chain(vectorstore)
    blood_report_data = extract_blood_report_values(conversation_chain)
    print("Extracted Blood Report Data:", blood_report_data)
    result = prediction(model, scaler, encoder, blood_report_data)
    print("Prediction Result:", result)

if __name__ == "__main__":
    main()