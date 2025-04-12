# 🔍 SPARK: Step-by-step Proof Assistant for Reasoning and Knowledge

> A lightweight framework for enhancing mathematical reasoning in Small Language Models (SLMs) using preference-based optimization methods like GRPO and DPO.

---

## 📚 Overview

SPARK (Step-by-step Proof Assistant for Reasoning and Knowledge) is a research initiative aimed at boosting the mathematical problem-solving capabilities of small language models through:

- Step-by-step reasoning strategies
- Reward-guided fine-tuning (GRPO)
- Evaluation on GSM8K and MATH datasets
- Structured XML reasoning outputs (Phi-mini)

> 📄 See the full [Report.pdf](./Report.pdf) for all methodology, experiments, and analysis.

---

## 🗂️ Repository Structure

```bash
.
├── ALM_Report         # LaTeX source and plots for the research report
├── Baseline           # Baseline evaluations for GPT-4, Gemini, Qwen, Phi
├── Dataset            # Processed GSM8K and MATH datasets + regex extraction
├── GRPO               # GRPO training code, graphs, checkpoints for Qwen and Phi
├── Mid-Eval Targets   # Mid-eval materials and checkpoints
├── Papers             # Reference papers (e.g., arXiv preprints)
├── Presentation       # Project presentations at different stages
├── Report.pdf         # Final PDF of the research report
├── LICENSE            # Licensing details
└── README.md          # You're here!
```

---

## 🚀 Key Features

- ✅ **Fine-Tuning with GRPO**: Direct policy optimization using group-relative preferences.
- 🔬 **Error Analysis & Reward Engineering**: Tailored rewards for correctness, length, and structured format.
- 📊 **Training Insights**: Includes logs, metrics, and visualizations of performance curves.
- 💡 **Format-Specific Fine-Tuning**: XML-based reasoning generation with structure-aware rewards.

---

## 🧠 Models & Techniques

### Models Evaluated
- `Qwen2-1.5B-Instruct`
- `Qwen2.5-MATH-1.5B` (4-bit)
- `Phi-mini-4k-Instruct` (4-bit)
- `Gemini-2.0-Flash` (Benchmark)

### Techniques Used
- Direct Preference Optimization (DPO)
- Group Relative Policy Optimization (GRPO)
- Chain-of-Thought prompting (CoT)
- XML-based reasoning generation (Phi)
- LoRA + QLoRA for efficient fine-tuning

---

## 📊 Datasets

### GSM8K
- Grade-school math word problems
- Focused on multi-step arithmetic reasoning

### MATH & MATH-500
- High-school competition math problems
- Filtered and verified using Gemini zero-shot answers

---

## 🧪 How to Run Experiments

1. Clone the repo:
```bash
git clone https://github.com/your-username/SPARK-math-reasoning.git
cd SPARK-math-reasoning
```

2. Install dependencies (use a virtual environment):
```bash
pip install -r requirements.txt
```

3. Start from baseline:
```bash
cd Baseline/Qwen-Math/Final_Code
python baseline_gsm8k_base.py  # or baseline_math_base.py
```

4. Run GRPO fine-tuning (Qwen or Phi):
```bash
cd GRPO/Qwen
python qwen_grpo.py
```

---

## 📈 Performance Summary

| Model                     | GSM8K (Baseline) | GSM8K (GRPO) | MATH (Baseline) | MATH (GRPO) |
|--------------------------|------------------|--------------|------------------|-------------|
| Qwen2.5-MATH-1.5B (4-bit)| 76.80%           | 78.60%       | 24.43%           | 29.00%      |
| Phi-mini-4k-instruct     | 78.04%           | **83.33%**   | 18.85%           | **33.50%**  |
| Gemini-2.0-Flash         | 72.78%           | N/A          | 33.69%           | N/A         |

---

## 📷 Visualizations

See the `ALM_Report/` and `GRPO/*/Graphs/` directories for plots such as:

- 📈 Train loss over time
- ✅ Correctness reward trends
- 🔍 Reasoning length reward
- 📦 Output format compliance (XML/LaTeX)

---

## 🎯 Goals

- Democratize mathematical reasoning through SLMs
- Maximize reasoning quality within hardware constraints
- Enable structured outputs for explainability (XML tagging)
- Use reward shaping instead of large-scale supervised fine-tuning

---

## 📜 License

This repository is licensed under the terms of the [MIT License](./LICENSE).

---

## 🏫 Acknowledgements

Developed at IIIT Hyderabad as part of the course/research on **LLM Alignment & Mathematical Reasoning**.  
Thanks to the open-source community and everyone who contributed to discussions and tools used.

---

## 📎 Resources

- [SPARK Report (PDF)](./Report.pdf)
- [Presentation Slides](./Presentation/)
- [Related Paper: arXiv:2402.03300v3](./Papers/2402.03300v3.pdf)
