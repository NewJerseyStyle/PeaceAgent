# 🕊️ Interactive Peace Simulator - 完整使用指南

## 專案概述

Interactive Peace Simulator 是一個基於 CrewAI 的交互式歷史模擬器，允許用戶扮演1937年中日危機中的關鍵決策者（昭和天皇或蔣中正），通過外交智慧嘗試阻止第二次中日戰爭的全面爆發。

### 🌟 核心特色

- **真實歷史背景**：基於詳細的1930年代軍國主義社會網絡分析
- **交互式決策**：使用 CrewAI 的 `human_input=True` 功能實現人機協作
- **雙重視角**：可選擇扮演日本天皇或中國領袖
- **模式選擇**：默認模式（有限信息）vs 開發者模式（全透明）
- **Web界面**：現代化的響應式Web UI
- **實時通信**：WebSocket 實現即時反饋
- **和平導向**：以防止戰爭為目標，而非軍事勝利

## 🏗️ 系統架構

```
Frontend (HTML/CSS/JS)
        ↕️
FastAPI Backend
        ↕️
CrewAI Simulation Engine
        ↕️
LLM Models (Groq/OpenAI/Anthropic)
```

### 組件說明

1. **Web前端** (`peace_simulator_web_ui.html`)
   - 響應式HTML/CSS/JS界面
   - 角色選擇、設定配置
   - 實時指標顯示
   - 決策輸入界面

2. **API後端** (`peace_simulator_api.py`)
   - FastAPI REST API
   - WebSocket 實時通信
   - 會話管理
   - 人機輸入隊列管理

3. **模擬引擎** (`peace_simulator_interface.py`)
   - 基於 CrewAI 的交互式模擬
   - 人工輸入集成
   - 和平指標計算
   - 詳細歷史建模

## 📦 安裝與配置

### 環境需求

```bash
# Python 版本
Python 3.9+

# 系統依賴
pip install --upgrade pip
```

### 1. 克隆或下載檔案

將以下檔案儲存到同一目錄：
- `peace_simulator_interface.py`
- `peace_simulator_api.py`
- `peace_simulator_web_ui.html`
- `sino_japanese_war_simulation.py`
- `waragent_crewai.py`

### 2. 安裝Python依賴

```bash
# 創建虛擬環境
python -m venv peace-simulator
source peace-simulator/bin/activate  # Linux/Mac
# peace-simulator\Scripts\activate.bat  # Windows

# 安裝核心依賴
pip install crewai crewai-tools
pip install langchain-openai langchain-anthropic langchain-groq
pip install fastapi uvicorn websockets
pip install pydantic python-multipart

# 可選：資料分析工具
pip install pandas numpy matplotlib plotly
```

### 3. API Key 設定

根據要使用的 LLM 模型設定對應的 API Key：

#### Groq（推薦，快速且經濟）
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```
- 註冊：https://console.groq.com
- 每分鐘請求限制高，成本低
- 適合高頻次互動模擬

#### OpenAI GPT-4
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```
- 最高品質但成本較高
- 適合最終版本或重要演示

#### Anthropic Claude
```bash
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
```

### 🚀 立即開始使用
```bash
# 1. 安裝依賴
pip install crewai crewai-tools fastapi uvicorn websockets
pip install langchain-openai langchain-anthropic langchain-groq

# 2. 設定 API Key (選擇一個)
export GROQ_API_KEY="your_groq_api_key"     # 推薦：快速且經濟
export OPENAI_API_KEY="your_openai_api_key" # 最高品質
export ANTHROPIC_API_KEY="your_claude_key"  # 優秀理解能力

# 3. 啟動 Web 應用
python peace_simulator_api.py

# 4. 打開瀏覽器
# 訪問 http://localhost:8000
```

#### 和平導向目標

不是征服，而是通過智慧避免戰爭
多維度和平指標：外交成功、國際聲望、內部穩定
6種策略選擇：外交倡議、軍事克制、國際調解等

#### 教育與研究價值

歷史教學：體驗決策複雜性
外交訓練：學習危機管理
學術研究：反事實歷史探索

## 🔄 核心工作流程

選擇角色 → 天皇或蔣中正
配置設定 → 模式、模型、回合數
接收情報 → AI 生成當前局勢分析
做出決策 → 6種策略 + 詳細理由說明
觀察反應 → AI 代理實時回應您的決策
評估結果 → 和平分數、外交成效等指標
下回合 → 基於前回合結果的新挑戰