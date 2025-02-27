# Disease Prediction from Blood Reports via Deepseek

## 📌 Overview
This project builds a **hybrid LSTM-CNN ** to predict diseases based on blood report values extracted from PDF files. It leverages **RAG (Retrieval-Augmented Generation)** to retrieve relevant data and integrates **FAISS vector embeddings** for efficient retrieval. 

## 🚀 Features
- **Extracts blood report values** from PDFs using an AI-based RAG model.
- **Preprocesses and normalizes** data for deep learning.
- **Combines Bidirectional LSTM & CNN layers** for sequential & spatial feature extraction.
- **Predicts possible diseases** based on blood parameters.
- **Uses FAISS embeddings** for efficient text search & retrieval.

## 🏗 Tech Stack
- **Python** (TensorFlow, NumPy, Pandas, Scikit-Learn, Matplotlib)
- **Deep Learning** (Bidirectional LSTM, CNN, FAISS, Ollama LLMs)
- **PDF Processing** (PyPDF2)
- **LangChain & FAISS** (for RAG-based document understanding)

## 🔧 Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```

## 🎯 Usage
1. Place the **blood report PDF** in the `data/` folder.
2. Run the main script:
   ```sh
   python main.py
   ```
3. The extracted blood parameters will be used to predict diseases.

## 📊 Model Architecture
The model consists of:
1. **Bidirectional LSTMs** for sequential feature learning.
2. **CNN layers** for spatial feature extraction.
3. **FAISS vector embeddings** to retrieve contextual medical data.
4. **Fully Connected Dense layers** for classification.

## 📌 Example Output
```sh
Extracted Blood Report Data: [130, 15.2, 200000, 6000, ...]
Predicted Disease: Diabetes
```

## 🔍 Future Enhancements
- Improve accuracy with more **pretrained embeddings**.
- Expand to **multimodal analysis** (e.g., MRI scans + blood reports).
- Using BlockChain smart contract for secure data EHR

## 📜 License
This project is licensed under the **MIT License**.

---
✅ *AI-powered disease prediction for better healthcare!* 🚀
