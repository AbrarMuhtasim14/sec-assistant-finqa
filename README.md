# 🏦 SEC-Assistant: Citation-Grounded Financial Q&A

Fine-tuned **Phi-3-Mini (3.8B)** for answering questions about SEC filings with built-in hallucination resistance.

## 🔗 Links

| Resource | Link |
|----------|------|
| 📊 Live Dashboard | [Streamlit App](https://sec-assistant-finappgit-fafbnrjvqiczunwyrgh5ln.streamlit.app/) |
| 🤗 Model | [HuggingFace](https://huggingface.co/spaces/Abrar144/sec-assistant-demo) |
| 📓 Notebook | [Training Pipeline](03-complete-pipeline.ipynb) |

## 🎯 Results

| Metric | Base Model | Fine-tuned | Change |
|--------|-----------|------------|--------|
| Citation Rate | 80% | 100% | +20% |
| Answer Rate | 45% | 100% | +55% |
| IDK Capability | ❌ | ✅ | New |
| General Reasoning | ✅ | ✅ | Preserved |

## 💡 What It Does

**Answers with citations:**


Q: What was Apple's revenue in 2022?
A: Based on the filing (AAPL/2022), the answer is: 394.3 billion.
Derived from: "Apple reported total net sales of $394.3 billion"

text


**Refuses when info missing:**
Q: What was Apple's employee count in 2022?
A: The provided context does not contain this information.



## 🛠️ Tech Stack

| Component | Detail |
|-----------|--------|
| Base Model | Phi-3-Mini (3.8B) |
| Method | QLoRA (r=16, α=32) |
| Dataset | FinQA (SEC filing Q&A) |
| Training | ~600 examples, 25 min on Kaggle T4 |
| Adapter Size | 22.5 MB |

## 📊 Approach

- **70% Normal examples**: Q&A with cited answers
- **30% IDK examples**: Mismatched context → model learns to refuse
- **Result**: Model cites sources AND knows its limits

## 🚀 Load the Model

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("microsoft/phi-3-mini-4k-instruct")
model = PeftModel.from_pretrained(base, "Abrar144/sec-assistant-phi3-finqa")

📁 Files
File	Description
03-complete-pipeline.ipynb	Complete training + evaluation notebook
app.py	Streamlit dashboard code
all_results.json	All evaluation results
training_history.json	Training loss curve data
metrics_comparison.csv	Before vs after metrics
⚠️ Limitations
Numeric accuracy limited (needs math reasoning)
~600 training examples (more = better)
English only, single-turn Q&A
Not financial advice
📄 License
MIT

