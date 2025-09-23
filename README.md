# PeaceAgent
Based on [agiresearch/WarAgent](https://github.com/agiresearch/WarAgent) but implemented in CrewAI with HITL for role play targeting for peace 🕊️🇺🇳

## 📦 Setup
### Python version
Python 3.9+
### Python virtual environment (Optional)
```bash
# Virtual environment if you need
python -m venv peace-simulator
source peace-simulator/bin/activate  # Linux/Mac
# peace-simulator\Scripts\activate.bat  # Windows
```
### Python Packages
Open terminal and install packages with commands
```bash
# Core dependencies
pip install crewai crewai-tools
pip install langchain-openai langchain-anthropic langchain-groq
pip install fastapi uvicorn websockets
pip install pydantic python-multipart

# Optional
pip install pandas numpy matplotlib plotly
```
### API Key (Setup any one of these)
```bash
export GROQ_API_KEY="your_groq_api_key"     # Economy
export OPENAI_API_KEY="your_openai_api_key" # Qulity
export ANTHROPIC_API_KEY="your_claude_key"  # Performance
export GEMINI_API_KEY="your_gemini_key"     # Balanced
```
### Start Web app
```
python peace_simulator_api.py --model gpt-4

# Browser goto http://localhost:8000
```

## Based on works
```
@article{hua2023war,
      title={War and Peace (WarAgent): Large Language Model-based Multi-Agent Simulation of World Wars}, 
      author={Wenyue Hua and Lizhou Fan and Lingyao Li and Kai Mei and Jianchao Ji and Yingqiang Ge and Libby Hemphill and Yongfeng Zhang},
      year={2023},
      eprint={2311.17227},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
```
@ARTICLE{Bueno_de_Mesquita2011-tu,
  title     = "A new model for predicting policy choices",
  author    = "Bueno de Mesquita, Bruce",
  abstract  = "A new forecasting model, solved for Bayesian Perfect Equilibria,
               is introduced. It, along with several alternative models, is
               tested on data from the European Union. The new model, which
               allows for contingent forecasts and for generating confidence
               intervals around predictions, outperforms competing models in
               most tests despite the absence of variance on a critical
               variable in all but nine cases. The more proximate the political
               setting of the issues is to the new model{\^a}€™s underlying
               theory of competitive and potentially coercive politics, the
               better the new model does relative to other models tested in the
               European Union context.",
  journal   = "Confl. Manag. Peace Sci.",
  publisher = "SAGE Publications",
  volume    =  28,
  number    =  1,
  pages     = "65--87",
  month     =  feb,
  year      =  2011,
  language  = "en"
}
```
