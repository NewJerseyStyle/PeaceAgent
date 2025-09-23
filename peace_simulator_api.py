#!/usr/bin/env python3
"""
Peace Simulator FastAPI Backend
===============================

This module provides a REST API backend for the Interactive Peace Simulator,
connecting the web frontend with the CrewAI simulation engine.

Features:
- RESTful API endpoints for simulation control
- Real-time WebSocket updates for simulation progress
- Human input integration with CrewAI agents
- Session management and result persistence
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our simulation modules
try:
    from peace_simulator_interface import (
        InteractivePeaceSimulator, PlayerRole, SimulationMode, PeaceMetrics
    )
    from sino_japanese_war_simulation import ConflictIntensity
except ImportError as e:
    logging.error(f"Failed to import simulation modules: {e}")
    sys.exit(1)


# Pydantic models for API requests/responses
class SimulationStartRequest(BaseModel):
    player_role: str  # "emperor" or "chiang"
    simulation_mode: str = "default"  # "default" or "dev"
    llm_model: str = "gemini/gemini-2.0-flash"
    max_turns: int = 15
    custom_trigger: Optional[str] = None


class PlayerDecisionRequest(BaseModel):
    decision_type: str
    reasoning: str
    target_group: Optional[str] = None
    diplomatic_message: Optional[str] = None
    internal_directive: Optional[str] = None


class SimulationStatus(BaseModel):
    session_id: str
    is_running: bool
    current_turn: int
    max_turns: int
    conflict_intensity: str
    peace_metrics: Dict[str, float]
    awaiting_human_input: bool


class TurnUpdate(BaseModel):
    turn_number: int
    situation_description: str
    decision_options: List[Dict[str, str]]
    intelligence_report: str
    dev_mode_info: Optional[Dict[str, Any]] = None


# Global state management
class SessionManager:
    def __init__(self):
        self.active_sessions: Dict[str, InteractivePeaceSimulator] = {}
        self.websocket_connections: Dict[str, WebSocket] = {}
        self.human_input_queues: Dict[str, asyncio.Queue] = {}
        self.session_results: Dict[str, Dict[str, Any]] = {}
    
    def create_session(self, session_id: str, config: SimulationStartRequest) -> InteractivePeaceSimulator:
        """Create a new simulation session."""
        player_role = PlayerRole(config.player_role)
        simulation_mode = SimulationMode(config.simulation_mode)
        
        simulator = InteractivePeaceSimulator(
            player_role=player_role,
            simulation_mode=simulation_mode,
            llm_model=config.llm_model
        )
        
        if config.custom_trigger:
            simulator.game_state.trigger_event = config.custom_trigger
        
        self.active_sessions[session_id] = simulator
        self.human_input_queues[session_id] = asyncio.Queue()
        
        return simulator
    
    def get_session(self, session_id: str) -> Optional[InteractivePeaceSimulator]:
        """Get an active simulation session."""
        return self.active_sessions.get(session_id)
    
    def remove_session(self, session_id: str):
        """Clean up a simulation session."""
        self.active_sessions.pop(session_id, None)
        self.websocket_connections.pop(session_id, None)
        self.human_input_queues.pop(session_id, None)
    
    async def add_websocket(self, session_id: str, websocket: WebSocket):
        """Add WebSocket connection for a session."""
        await websocket.accept()
        self.websocket_connections[session_id] = websocket
    
    async def send_update(self, session_id: str, message: Dict[str, Any]):
        """Send update to client via WebSocket."""
        websocket = self.websocket_connections.get(session_id)
        if websocket:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logging.error(f"Failed to send WebSocket message to {session_id}: {e}")
    
    async def queue_human_input(self, session_id: str, decision: PlayerDecisionRequest):
        """Queue human input for processing by CrewAI."""
        queue = self.human_input_queues.get(session_id)
        if queue:
            await queue.put(decision)


# Initialize FastAPI app and session manager
app = FastAPI(title="Peace Simulator API", version="1.0.0")
session_manager = SessionManager()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (the HTML interface)
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main HTML interface."""
    # In production, you'd serve this from the peace_simulator_web_ui.html file
    return HTMLResponse(open("peace_simulator_web_ui.html", "r", encoding="utf-8").read())


@app.post("/api/simulation/start")
async def start_simulation(config: SimulationStartRequest, background_tasks: BackgroundTasks):
    """Start a new peace simulation."""
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config.player_role}"
    
    try:
        # Validate API keys based on model selection
        if config.llm_model.startswith("groq/") and not os.getenv("GROQ_API_KEY"):
            raise HTTPException(status_code=400, detail="GROQ_API_KEY not configured")
        elif config.llm_model == "gpt-4" and not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(status_code=400, detail="OPENAI_API_KEY not configured")
        elif config.llm_model == "claude-2" and not os.getenv("ANTHROPIC_API_KEY"):
            raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY not configured")
        
        # Create simulation session
        simulator = session_manager.create_session(session_id, config)
        
        # Start simulation in background
        background_tasks.add_task(run_simulation_background, session_id)
        
        return {
            "session_id": session_id,
            "status": "started",
            "player_role": config.player_role,
            "simulation_mode": config.simulation_mode,
            "message": "Simulation started successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start simulation: {str(e)}")


@app.get("/api/simulation/{session_id}/status")
async def get_simulation_status(session_id: str):
    """Get current simulation status."""
    simulator = session_manager.get_session(session_id)
    if not simulator:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SimulationStatus(
        session_id=session_id,
        is_running=True,  # Simplified for demo
        current_turn=simulator.game_state.current_turn,
        max_turns=15,  # Would be stored in session config
        conflict_intensity=simulator.conflict_intensity.value,
        peace_metrics={
            "peace_score": simulator.peace_metrics.conflict_prevention,
            "diplomatic_success": simulator.peace_metrics.diplomatic_success,
            "international_reputation": simulator.peace_metrics.international_reputation,
            "internal_stability": simulator.peace_metrics.internal_stability
        },
        awaiting_human_input=True  # Simplified for demo
    )


@app.post("/api/simulation/{session_id}/decision")
async def submit_decision(session_id: str, decision: PlayerDecisionRequest):
    """Submit human player decision."""
    simulator = session_manager.get_session(session_id)
    if not simulator:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Queue the human input for processing
        await session_manager.queue_human_input(session_id, decision)
        
        # Send acknowledgment via WebSocket
        await session_manager.send_update(session_id, {
            "type": "decision_received",
            "message": "Decision received and processing..."
        })
        
        return {"status": "decision_received", "message": "Decision is being processed"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process decision: {str(e)}")


@app.get("/api/simulation/{session_id}/results")
async def get_simulation_results(session_id: str):
    """Get final simulation results."""
    results = session_manager.session_results.get(session_id)
    if not results:
        raise HTTPException(status_code=404, detail="Results not found")
    
    return results


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time updates."""
    await session_manager.add_websocket(session_id, websocket)
    
    try:
        while True:
            # Keep connection alive and handle any client messages
            message = await websocket.receive_text()
            # Echo back for debugging
            await websocket.send_json({
                "type": "echo",
                "message": f"Received: {message}"
            })
            
    except WebSocketDisconnect:
        logging.info(f"WebSocket disconnected for session {session_id}")
        session_manager.websocket_connections.pop(session_id, None)


async def run_simulation_background(session_id: str):
    """Run the simulation in background with human input integration."""
    simulator = session_manager.get_session(session_id)
    if not simulator:
        return
    
    try:
        await session_manager.send_update(session_id, {
            "type": "simulation_started",
            "message": "Simulation started, generating initial situation..."
        })
        
        # Run simulation turns
        for turn in range(1, 16):  # Max 15 turns
            simulator.game_state.current_turn = turn
            
            # Send turn start notification
            await session_manager.send_update(session_id, {
                "type": "turn_started",
                "turn": turn,
                "message": f"Turn {turn} beginning..."
            })
            
            # Generate situation and options
            turn_data = await generate_turn_data(simulator)
            
            # Send turn data to client
            await session_manager.send_update(session_id, {
                "type": "turn_data",
                "data": turn_data
            })
            
            # Wait for human input
            queue = session_manager.human_input_queues.get(session_id)
            if queue:
                try:
                    # Wait for human decision with timeout
                    human_decision = await asyncio.wait_for(queue.get(), timeout=300.0)  # 5 minutes timeout
                    
                    # Process human decision
                    await process_human_decision(simulator, human_decision, session_id)
                    
                    # Check end conditions
                    if check_simulation_end(simulator):
                        break
                        
                except asyncio.TimeoutError:
                    await session_manager.send_update(session_id, {
                        "type": "timeout",
                        "message": "Decision timeout - simulation paused"
                    })
                    break
        
        # Generate final results
        final_results = generate_final_results(simulator)
        session_manager.session_results[session_id] = final_results
        
        await session_manager.send_update(session_id, {
            "type": "simulation_ended",
            "results": final_results
        })
        
    except Exception as e:
        logging.error(f"Simulation error for {session_id}: {e}")
        await session_manager.send_update(session_id, {
            "type": "error", 
            "message": f"Simulation error: {str(e)}"
        })
    
    finally:
        # Clean up session
        session_manager.remove_session(session_id)


async def generate_turn_data(simulator: InteractivePeaceSimulator) -> Dict[str, Any]:
    """Generate turn data including situation and decision options."""
    
    # Generate situation description
    situations = {
        PlayerRole.EMPEROR: [
            "軍部將領要求您批准對華北的進一步軍事行動，聲稱這是維護帝國尊嚴的必要措施。同時，國際觀察家警告此舉可能引發全面戰爭。",
            "關東軍報告中國軍隊在華北集結，要求預先批准自衛反擊。外務省建議透過外交途徑解決，但軍部認為外交手段過於軟弱。",
            "國際聯盟呼籲雙方克制，美國表達關切。軍部認為這是西方勢力干涉，但也有顧問建議這是展現日本和平意願的機會。"
        ],
        PlayerRole.CHIANG: [
            "日軍在華北調動頻繁，地方指揮官請求中央政府明確指示。共產黨方面表示願意暫時合作抗日，但條件是停止剿共。",
            "各地軍閥對是否抵抗日軍意見分歧，有些主張妥協以保存實力，有些要求堅決抵抗。國際社會的支持仍然有限。",
            "日本方面釋出可能談判的信號，但條件苛刻。國內輿論要求強硬回應，但軍事實力差距明顯。"
        ]
    }
    
    situation_list = situations[simulator.player_role]
    situation_index = (simulator.game_state.current_turn - 1) % len(situation_list)
    situation_description = situation_list[situation_index]
    
    # Generate decision options
    decision_options = [
        {"id": "diplomatic", "title": "外交倡議", "description": "提議直接談判，尋求和平解決方案"},
        {"id": "restraint", "title": "軍事克制", "description": "下令避免軍事升級，保持防禦態勢"},
        {"id": "mediation", "title": "國際調解", "description": "尋求第三方國際力量介入調停"},
        {"id": "unity", "title": "內部團結", "description": "團結國內各派系支持和平政策"},
        {"id": "concession", "title": "策略讓步", "description": "提出妥協方案以防止戰爭爆發"},
        {"id": "appeal", "title": "公開呼籲", "description": "向國際社會發表和平聲明"}
    ]
    
    # Generate intelligence report
    intelligence_report = generate_intelligence_report(simulator)
    
    # Dev mode information
    dev_info = None
    if simulator.simulation_mode == SimulationMode.DEV:
        dev_info = {
            "ai_agent_thoughts": [
                "Tosei-ha Agent: 分析當前局勢，建議採取漸進式控制策略以避免國際制裁...",
                "Chinese Communist Agent: 評估統一戰線可能性，準備在日軍威脅下暫時與國民黨合作...",
                "Zaibatsu Agent: 計算戰爭經濟收益vs國際貿易損失，建議謹慎評估軍事行動規模...",
                "International Observer: 西方列強正密切關注局勢發展，準備可能的調停介入..."
            ],
            "faction_relationships": {
                "Tosei_Ha_vs_Kodo_Ha": -0.7,
                "Emperor_vs_Military": 0.3,
                "Chinese_Communists_vs_KMT": -0.6,
                "International_Pressure": 0.4
            },
            "resource_status": {
                "Japanese_Military_Readiness": 85,
                "Chinese_Defense_Capability": 45,
                "International_Support_China": 30,
                "Economic_Pressure_Japan": 25
            }
        }
    
    return TurnUpdate(
        turn_number=simulator.game_state.current_turn,
        situation_description=situation_description,
        decision_options=decision_options,
        intelligence_report=intelligence_report,
        dev_mode_info=dev_info
    ).dict()


def generate_intelligence_report(simulator: InteractivePeaceSimulator) -> str:
    """Generate intelligence report based on player role and simulation mode."""
    
    if simulator.player_role == PlayerRole.EMPEROR:
        if simulator.simulation_mode == SimulationMode.DEV:
            return """🔍 完整情報 (開發者模式):
- 統制派正計劃系統性擴張，預估需要3個月完成華北控制
- 關東軍已獲得部分自主行動授權，正在測試中央政府反應
- 財閥對戰爭經濟持謹慎樂觀態度，但擔心國際制裁
- 中國內部：蔣介石與共產黨談判破裂機率70%，軍閥分化嚴重
- 國際社會：美國國會反戰情緒高漲，英國專注歐洲局勢"""
        else:
            return """📋 可用情報:
- 軍事顧問報告陸軍部內部對行動規模存在分歧
- 外務省確認國際社會正密切關注事態發展
- 宮內省建議考慮帝國長遠利益與國際形象
- 海軍方面對陸軍獨立行動表達關切"""
    
    else:  # Chiang Kai-shek
        if simulator.simulation_mode == SimulationMode.DEV:
            return """🔍 完整情報 (開發者模式):
- 日本天皇派與軍部存在分歧，有40%機會接受有條件談判
- 關東軍補給線延長，冬季作戰能力將受限
- 蘇聯對華軍援意願低，但願意提供情報支持
- 美國商界對華同情，但政府不願直接介入
- 共產黨軍力評估：正規軍3萬，民兵8萬，士氣高昂"""
        else:
            return """📋 可用情報:
- 日軍調動規模較大，但後勤補給線較長
- 地方軍閥態度分歧，需要中央政府明確指導
- 國際輿論同情中國，但實質支持有限
- 共產黨釋出合作信號，但要求政治讓步"""
    
    return "情報收集中..."


async def process_human_decision(simulator: InteractivePeaceSimulator, decision: PlayerDecisionRequest, session_id: str):
    """Process human decision and update simulation state."""
    
    # Send processing notification
    await session_manager.send_update(session_id, {
        "type": "processing_decision",
        "message": "正在處理您的決策，AI代理正在分析並回應..."
    })
    
    # Simulate decision processing time
    await asyncio.sleep(2)
    
    # Calculate decision impact
    impact = calculate_decision_impact(decision, simulator.player_role)
    
    # Update peace metrics
    simulator.peace_metrics.conflict_prevention = max(0, min(100, 
        simulator.peace_metrics.conflict_prevention + impact["peace"]))
    simulator.peace_metrics.diplomatic_success += impact["diplomacy"]
    simulator.peace_metrics.international_reputation = max(0, min(100,
        simulator.peace_metrics.international_reputation + impact["reputation"]))
    simulator.peace_metrics.internal_stability = max(0, min(100,
        simulator.peace_metrics.internal_stability + impact["stability"]))
    
    # Update conflict intensity
    if impact["peace"] > 10:
        if simulator.conflict_intensity == ConflictIntensity.FULL_WAR:
            simulator.conflict_intensity = ConflictIntensity.LIMITED_WAR
        elif simulator.conflict_intensity == ConflictIntensity.LIMITED_WAR:
            simulator.conflict_intensity = ConflictIntensity.SKIRMISH
        elif simulator.conflict_intensity == ConflictIntensity.SKIRMISH:
            simulator.conflict_intensity = ConflictIntensity.DIPLOMATIC
    elif impact["peace"] < -10:
        if simulator.conflict_intensity == ConflictIntensity.DIPLOMATIC:
            simulator.conflict_intensity = ConflictIntensity.SKIRMISH
        elif simulator.conflict_intensity == ConflictIntensity.SKIRMISH:
            simulator.conflict_intensity = ConflictIntensity.LIMITED_WAR
    
    # Generate AI responses
    ai_responses = generate_ai_responses(decision, simulator.player_role, simulator.simulation_mode)
    
    # Send decision results
    await session_manager.send_update(session_id, {
        "type": "decision_processed",
        "impact": impact,
        "ai_responses": ai_responses,
        "updated_metrics": {
            "peace_score": simulator.peace_metrics.conflict_prevention,
            "diplomatic_success": simulator.peace_metrics.diplomatic_success,
            "international_reputation": simulator.peace_metrics.international_reputation,
            "internal_stability": simulator.peace_metrics.internal_stability
        },
        "conflict_intensity": simulator.conflict_intensity.value
    })


def calculate_decision_impact(decision: PlayerDecisionRequest, player_role: PlayerRole) -> Dict[str, float]:
    """Calculate the impact of a human decision on various metrics."""
    
    base_impacts = {
        "diplomatic": {"peace": 15, "diplomacy": 20, "reputation": 10, "stability": 5},
        "restraint": {"peace": 20, "diplomacy": 10, "reputation": 15, "stability": 10},
        "mediation": {"peace": 10, "diplomacy": 25, "reputation": 20, "stability": 5},
        "unity": {"peace": 5, "diplomacy": 15, "reputation": 5, "stability": 25},
        "concession": {"peace": 25, "diplomacy": 15, "reputation": -5, "stability": -10},
        "appeal": {"peace": 10, "diplomacy": 20, "reputation": 15, "stability": 0}
    }
    
    impact = base_impacts.get(decision.decision_type, {"peace": 0, "diplomacy": 0, "reputation": 0, "stability": 0})
    
    # Add role-specific modifiers
    if player_role == PlayerRole.EMPEROR:
        # Emperor has more authority, stronger impact
        for key in impact:
            impact[key] *= 1.2
    else:
        # Chiang faces more constraints, but diplomatic moves are more credible internationally
        if decision.decision_type in ["diplomatic", "mediation", "appeal"]:
            impact["reputation"] *= 1.3
    
    # Add reasoning quality bonus
    reasoning_length = len(decision.reasoning)
    quality_bonus = min(5, reasoning_length / 100)  # Up to 5 points for detailed reasoning
    
    for key in impact:
        if key != "stability":  # Stability less affected by reasoning quality
            impact[key] += quality_bonus
    
    # Add some randomness
    import random
    for key in impact:
        impact[key] += random.uniform(-3, 3)
    
    return impact


def generate_ai_responses(decision: PlayerDecisionRequest, player_role: PlayerRole, mode: SimulationMode) -> List[str]:
    """Generate AI agent responses to human decisions."""
    
    responses = []
    
    if player_role == PlayerRole.EMPEROR:
        # Japanese side responses
        if decision.decision_type == "diplomatic":
            responses.append("🇨🇳 中國方面：對天皇陛下的和平倡議表示謹慎歡迎，願意在平等基礎上進行談判。")
            responses.append("⚔️ 軍部反應：統制派表達不滿，認為外交讓步顯示軟弱。關東軍要求保留自衛反擊權。")
            responses.append("🌐 國際社會：各國讚揚日本的和平姿態，美國表示願意提供調停協助。")
        elif decision.decision_type == "restraint":
            responses.append("🇨🇳 中國方面：中國軍隊停止進一步動員，蔣介石發表聲明感謝天皇陛下的理智決定。")
            responses.append("⚔️ 軍部反應：勉強執行天皇命令，但警告錯失戰略機會將影響帝國未來安全。")
            responses.append("🌐 國際社會：國際輿論高度讚揚，英美表示日本展現了負責任大國風範。")
    
    else:  # Chiang Kai-shek
        if decision.decision_type == "diplomatic":
            responses.append("🇯🇵 日本方面：天皇及溫和派對談判提議表示可以考慮，但軍部要求先停止反日活動。")
            responses.append("🇨🇳 國內反應：部分將領支持談判，但學生團體抗議對日妥協。共產黨態度微妙。")
            responses.append("🌐 國際社會：列強支持中國的和平努力，但提醒需要務實的談判立場。")
        elif decision.decision_type == "unity":
            responses.append("🇯🇵 日本方面：日軍暫停進攻，觀察中國內部統一程度。軍部對中國團結表達警惕。")
            responses.append("🇨🇳 國內反應：多數軍閥響應統一號召，共產黨同意暫時停止內戰。民眾士氣大振。")
            responses.append("🌐 國際社會：中國的團結決心獲得國際讚賞，蘇聯暗示可能提供更多支持。")
    
    # Add dev mode detailed responses
    if mode == SimulationMode.DEV:
        responses.append(f"🔍 AI分析: {decision.decision_type}決策的成功機率約為65-75%，主要風險在於...")
        responses.append("⚙️ 系統評估: 當前決策符合和平導向目標，預期可降低衝突升級風險12-18%")
    
    return responses


def check_simulation_end(simulator: InteractivePeaceSimulator) -> bool:
    """Check if simulation should end based on current state."""
    
    # Peace achieved
    if (simulator.peace_metrics.conflict_prevention >= 80 and 
        simulator.peace_metrics.diplomatic_success >= 50):
        return True
    
    # Peace failed
    if simulator.peace_metrics.conflict_prevention <= 20:
        return True
    
    # Maximum turns reached
    if simulator.game_state.current_turn >= 15:
        return True
    
    return False


def generate_final_results(simulator: InteractivePeaceSimulator) -> Dict[str, Any]:
    """Generate final simulation results."""
    
    # Determine outcome
    outcome_type = ""
    outcome_message = ""
    
    if simulator.peace_metrics.conflict_prevention >= 80:
        outcome_type = "peace_achieved"
        outcome_message = "🕊️ 和平達成！您成功阻止了戰爭爆發，外交智慧拯救了無數生命。"
    elif simulator.peace_metrics.conflict_prevention <= 20:
        outcome_type = "peace_failed"
        outcome_message = "💥 和平失敗。儘管努力嘗試，但衝突仍然無法避免。"
    elif simulator.peace_metrics.conflict_prevention >= 60:
        outcome_type = "partial_success"
        outcome_message = "🤝 部分成功。雖未完全阻止衝突，但顯著降低了戰爭規模。"
    else:
        outcome_type = "mixed_results"
        outcome_message = "⚖️ 結果複雜。局勢得到一定控制，但仍存在不穩定因素。"
    
    # Performance rating
    peace_score = simulator.peace_metrics.conflict_prevention
    if peace_score >= 80:
        rating = "outstanding"
        rating_message = "⭐⭐⭐ 卓越外交家"
    elif peace_score >= 60:
        rating = "good"
        rating_message = "⭐⭐ 優秀談判者"
    elif peace_score >= 40:
        rating = "fair"
        rating_message = "⭐ 合格外交官"
    else:
        rating = "needs_improvement"
        rating_message = "💭 需要改進"
    
    return {
        "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "player_role": simulator.player_role.value,
        "simulation_mode": simulator.simulation_mode.value,
        "turns_completed": simulator.game_state.current_turn,
        "final_metrics": {
            "peace_score": simulator.peace_metrics.conflict_prevention,
            "diplomatic_success": simulator.peace_metrics.diplomatic_success,
            "international_reputation": simulator.peace_metrics.international_reputation,
            "internal_stability": simulator.peace_metrics.internal_stability
        },
        "conflict_intensity": simulator.conflict_intensity.value,
        "outcome": {
            "type": outcome_type,
            "message": outcome_message
        },
        "performance": {
            "rating": rating,
            "message": rating_message
        },
        "timestamp": datetime.now().isoformat(),
        "total_events": len(simulator.detailed_events) if hasattr(simulator, 'detailed_events') else 0
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(session_manager.active_sessions)
    }


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the API server
    uvicorn.run(
        "peace_simulator_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
