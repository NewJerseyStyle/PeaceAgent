# Enhanced WarAgent CrewAI Implementation - Complete Setup Guide

## 新功能概述 (New Features Overview)

本增強版本在原有 WarAgent 基礎上新增：

### 1. Groq LLM 支持
- 支持高性能 Groq 模型：`llama3-70b-8192`、`llama3-8b-8192`、`mixtral-8x7b-32768`
- 快速推理速度，適合高頻次 agent 交互
- 成本效益高，適合長時間模擬

### 2. 第二次中日戰爭詳細模擬
基於歷史資料，實現了更高粒度的1930年代中日衝突模擬：

#### 日本方面細化角色：
- **天皇與皇室顧問**：神權統治與政治平衡
- **統制派 (Tosei-ha)**：現代化軍國主義官僚
- **皇道派 (Kōdō-ha)**：精神主義激進派
- **關東軍**：自主性強的滿洲駐軍
- **中國方面軍**：侵華主力部隊
- **韓國駐軍**：殖民地控制力量
- **海軍派系**：艦隊派 vs 條約派
- **財閥 (Zaibatsu)**：軍工複合體
- **帝國議會與內閣**：式微的文官政治

#### 中國方面細化角色：
- **蔣介石中央政府**：國民黨正統政權
- **中央軍**：德式訓練的現代化部隊
- **汪精衛左派**：親日合作政府
- **各地軍閥**：北方 (張氏)、中央 (馮系)、南方 (廣西) 派系
- **中國共產黨**：革命武裝與群眾動員
- **共產黨游擊隊**：敵後武裝鬥爭
- **農民群眾**：戰爭受害者與抵抗基礎
- **殖民地人民**：朝鮮、台灣被壓迫群體

### 3. 複雜社會網絡動力學
- **派系關係追蹤**：內部聯盟與敵對關係
- **資源流動分析**：軍事、經濟、政治、社會資源
- **衝突強度階梯**：從外交摩擦到全面戰爭的漸進升級
- **下克上 (Gekokujo) 機制**：日軍基層突破上級限制的策略
- **統一戰線動力學**：中國各派系的合作與競爭

## 安裝指南

### 環境需求
```bash
# Python 3.9+
python --version

# 創建虛擬環境
conda create --name waragent-enhanced python=3.9
conda activate waragent-enhanced

# 或使用 venv
python -m venv waragent-enhanced
source waragent-enhanced/bin/activate  # Linux/Mac
# waragent-enhanced\Scripts\activate.bat  # Windows
```

### 依賴安裝
```bash
# 核心框架
pip install crewai crewai-tools

# LLM 支持
pip install langchain-openai langchain-anthropic langchain-groq

# 基礎依賴
pip install langchain pandas numpy matplotlib seaborn

# 可選：數據分析和可視化
pip install networkx plotly jupyter
```

### API Key 設置

#### OpenAI (GPT-4)
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

#### Anthropic (Claude-2)
```bash
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
```

#### Groq (推薦用於高頻模擬)
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

獲取 Groq API Key：
1. 訪問 [Groq Console](https://console.groq.com)
2. 註冊並創建 API Key
3. Groq 提供快速推理和較低成本

## 使用指南

### 基本使用

#### 1. 運行標準第二次中日戰爭模擬
```bash
python sino_japanese_war_simulation.py --model groq/llama3-70b-8192 --max_turns 15
```

#### 2. 使用自定義觸發事件
```bash
python sino_japanese_war_simulation.py \
  --model groq/llama3-70b-8192 \
  --trigger "盧溝橋事變後日軍全面侵華" \
  --max_turns 20 \
  --verbose
```

#### 3. 導出詳細分析
```bash
python sino_japanese_war_simulation.py \
  --model groq/mixtral-8x7b-32768 \
  --max_turns 25 \
  --export_analysis \
  --output detailed_analysis_$(date +%Y%m%d).json
```

### 模型選擇建議

| 模型 | 優勢 | 適用場景 | 成本 |
|------|------|----------|------|
| `groq/llama3-70b-8192` | 最高推理質量，長上下文 | 複雜歷史模擬，詳細分析 | 中等 |
| `groq/llama3-8b-8192` | 快速響應，成本低 | 快速原型，大量實驗 | 低 |
| `groq/mixtral-8x7b-32768` | 平衡性能與成本 | 中等複雜度模擬 | 中低 |
| `gpt-4` | 最高整體質量 | 最終發布版本 | 高 |
| `claude-2` | 長文本理解佳 | 歷史文獻分析 | 中高 |

### 進階配置

#### 自定義參數配置
```python
# 創建自定義模擬實例
from sino_japanese_war_simulation import SinoJapaneseWarSimulation

simulation = SinoJapaneseWarSimulation(llm_model="groq/llama3-70b-8192")

# 修改衝突強度
simulation.conflict_intensity = ConflictIntensity.LIMITED_WAR

# 調整派系關係
simulation.faction_relationships["Tosei_Ha"]["Kodo_Ha"] = -0.9

# 設置資源初始值
simulation.resource_flows["Kwantung_Army"]["military_strength"] = 120
simulation.resource_flows["Chinese_Communists"]["popular_support"] = 80

# 運行模擬
results = simulation.run_detailed_simulation(max_turns=30)
```

#### 批量實驗
```bash
#!/bin/bash
# 運行多重情境實驗

models=("groq/llama3-70b-8192" "groq/mixtral-8x7b-32768" "groq/llama3-8b-8192")
triggers=("Marco Polo Bridge Incident" "Full Japanese Invasion" "Communist United Front")

for model in "${models[@]}"; do
  for trigger in "${triggers[@]}"; do
    echo "Running simulation: $model with trigger: $trigger"
    python sino_japanese_war_simulation.py \
      --model "$model" \
      --trigger "$trigger" \
      --max_turns 20 \
      --output "results_${model//\//_}_${trigger// /_}.json" \
      --export_analysis
  done
done
```

## 輸出分析

### 標準輸出結果
```json
{
  "scenario": "Second Sino-Japanese War - Detailed",
  "trigger_event": "Marco Polo Bridge Incident - July 7, 1937",
  "initial_conflict_intensity": "diplomatic",
  "final_state": {
    "final_conflict_intensity": "full_war",
    "outcome": "Chinese Communist Victory - Popular revolution successful",
    "total_turns": 18,
    "total_events": 156
  },
  "network_analysis": {
    "faction_influence_scores": {
      "Chinese_Communists": 245.6,
      "Tosei_Ha": 198.3,
      "Chiang_Kai_Shek": 167.8
    }
  }
}
```

### 關鍵指標解釋

#### 1. 衝突強度階梯 (Conflict Intensity)
- `diplomatic`: 外交摩擦階段
- `skirmish`: 局部衝突
- `limited_war`: 有限戰爭  
- `full_war`: 全面戰爭
- `total_war`: 總體戰爭

#### 2. 派系影響力分數 (Faction Influence)
綜合計算：
- 軍事實力 (30%)
- 經濟資源 (25%) 
- 政治影響 (25%)
- 民眾支持 (20%)

#### 3. 資源流動模式
追蹤各派系的：
- 軍事實力變化
- 經濟資源調動
- 政治影響力升降
- 民眾支持度波動

## 歷史準確性與學術價值

### 基於史實的設計
- 使用1930年代軍國主義多個社會集團的社會網絡為基礎
- 整合皇道派與統制派的內部鬥爭動力學
- 反映關東軍下克上傳統與自主行動模式
- 體現中國各派系的複雜聯盟關係

### 可用於學術研究
- 量化歷史決策過程
- 分析複雜社會網絡動力學
- 探索反事實歷史情境
- 研究衝突升級機制

### 教育應用價值
- 互動式歷史教學
- 戰略決策訓練
- 國際關係理論實證
- 複雜系統分析練習

## 故障排除

### 常見問題

#### 1. Groq API 連接失敗
```bash
# 檢查 API Key
echo $GROQ_API_KEY

# 測試連接
curl -H "Authorization: Bearer $GROQ_API_KEY" \
     https://api.groq.com/openai/v1/models
```

#### 2. 內存不足
```bash
# 減少並發 agent 數量
export CREWAI_MAX_AGENTS=3

# 或使用較小模型
python sino_japanese_war_simulation.py --model groq/llama3-8b-8192
```

#### 3. 長時間運行超時
```bash
# 設置較短的 turn 數
python sino_japanese_war_simulation.py --max_turns 10

# 或分批運行
python sino_japanese_war_simulation.py --max_turns 5 --output batch_1.json
```

### 性能優化建議

1. **模型選擇**：開發測試用 `llama3-8b`，正式分析用 `llama3-70b`
2. **批量處理**：使用腳本運行多個情境，避免單次長時間運行
3. **結果緩存**：保存中間結果，支持斷點續跑
4. **並行實驗**：在不同終端運行不同參數的模擬

## 擴展開發

### 添加新派系
```python
# 在 _load_second_sino_japanese_war_data() 中添加
"New_Faction": CountryProfile(
    name="New Faction Name",
    resources={"custom_resource": 50},
    military_strength=60,
    # ... 其他屬性
)
```

### 自定義事件類型
```python
class CustomEventType(Enum):
    DIPLOMATIC_CRISIS = "diplomatic_crisis"
    ECONOMIC_WARFARE = "economic_warfare"
    # 添加更多事件類型
```

### 擴展分析功能
```python
def custom_analysis(self, results):
    # 添加自定義分析邏輯
    network_metrics = self.calculate_network_centrality()
    results["custom_analysis"] = network_metrics
    return results
```

這個增強版本提供了前所未有的歷史模擬精度和分析深度，特別適合研究1930年代東亞複雜的政治軍事動力學。
