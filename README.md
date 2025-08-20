# 🧠 Fact-Checking with LLaMA + BM25

This project is an **AI-powered fact-checking system** that uses **BM25** for evidence retrieval and **LLaMA 3.2 (via Ollama)** for reasoning to verify political claims. It processes a dataset of claims with supporting evidence, ranks the most relevant passages, generates reasoning with a fact-checking verdict, and evaluates predictions using accuracy, precision, recall, and F1-score. The system also provides visual insights through confusion matrices, prediction distributions, and BM25 score analysis, making it a complete pipeline for automated claim verification and performance evaluation.

---

## 🔑 Key Features
- **BM25-based evidence retrieval** for ranking relevant passages  
- **LLaMA 3.2 reasoning** via Ollama for fact-checking labels  
- **Evaluation metrics**: Accuracy, Precision, Recall, F1-score  
- **Visualizations**: Confusion matrix, prediction distribution, label comparison, BM25 score analysis  
- **JSON dataset support** for claim–evidence pairs  
- **Complete pipeline** from claim input → reasoning → evaluation → visualization  

---

## 📂 Project Structure
```
fact-checking-llm/
│── final_working_code.py     # Main pipeline
│── requirements.txt          # Dependencies
│── README.md                 # Documentation
│── data/
│    └── train.json            # Training dataset
│    └── test.json             # Testing dataset
│── results/
│    └── pred vs act 2.png/              # Stores output plots
│    └── bm25 2.png/
│    └── pred final.png/
│    └── confusion final.png/
│    └── predictions/
│    └── bm25scores/
```

---

## ⚙️ Installation
```bash
git clone https://github.com/GaganaBhaskar/TripleR-Fact-check.git
cd TripleR-Fact-check
pip install -r requirements.txt
```

---

## ▶️ Usage
1. Make sure **Ollama** is installed and running locally:
   ```bash
   ollama run llama3.2
   ```
2. Place your dataset in `data/test.json`.
3. Run the pipeline:
   ```bash
   python final_working_code.py
   ```

---

## 📊 Output
- **Evaluation Metrics** (Accuracy, Precision, Recall, F1)  
- **Reasoning & Predictions** for each claim  
- **Plots** saved in `results/graphs/`:  
  - Confusion Matrix  
  - Prediction Accuracy Distribution (Pie Chart)  
  - Predicted vs Actual Labels (Line Plot)  
  - BM25 Score Scatter Plot  

---

## 🛠️ Tech Stack
- Python 3.12  
- [Ollama](https://ollama.ai/) – for LLaMA 3.2 inference  
- BM25 (rank-bm25)  
- scikit-learn  
- matplotlib, seaborn  
- json, os

---
##🔗 APIs

Ollama API – http://localhost:11434/api/generate    #for claim verification reasoning

---
##🙏 Acknowledgement

I would like to extend my sincere gratitude to my mentors, Leena Chandrashekar and Pavan Kumar C, for their invaluable guidance, encouragement, and continuous support throughout this project. Their mentorship and insights were fundamental to the successful completion of this work.
