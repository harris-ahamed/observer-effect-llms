# The Observer Effect in LLMs

**Universität Koblenz · Research Lab 2026**

Team: Harris Ahamed Anwerdeen · Nisarga Doddapalya Boregowda · Irfan Ahmed Jalaludeen Ahmed · Syed Abdul Kader  
Supervisor: Prof. Marina Ernst

## About

This research investigates whether Large Language Models (LLMs) change their behaviour when they know they are being observed — analogous to the Hawthorne Effect in human psychology.

We tested 4 AI models (ChatGPT GPT-4o mini, Gemini 2.5 Flash Lite, Claude Haiku 4.5, Llama 3.1 8B Instant) across 4 observer contexts (Normal, Developer Watching, Casual, Research Evaluation) using 3 peer-reviewed datasets (TruthfulQA, MMLU, Do-Not-Answer), collecting 3,168 responses in total.

We measured 6 behavioural parameters and found that 3 — Word Count, Disclaimers, and Position Clarity — changed significantly (ANOVA + Bonferroni correction, α = 0.0028).

**Live Dashboard:** [observer-effect-llms.streamlit.app](https://observer-effect-llms.streamlit.app)

## Repository Structure

- `app.py` — Streamlit dashboard for interactive exploration of results
- `observer_effect_collect.py` — Main data collection script (MMLU + Do-Not-Answer)
- `stage1_collect.py` — TruthfulQA data collection script
- `data.xlsx` — Combined dataset of all 3,168 responses
- `requirements.txt` — Python dependencies
- `config.toml` — Streamlit configuration

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Key Findings

- Word Count dropped 13% under Developer Watching context (p = 0.0021 ★)
- Disclaimers increased 64% in Casual context (p = 0.0015 ★)
- Position Clarity increased in Casual context (p = 0.0013 ★)
- Gemini showed the largest change (−25%), Claude was the only model to increase (+7%)
