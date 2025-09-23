#!/usr/bin/env python3
"""
Interactive Peace Simulator - 1937 Sino-Japanese Crisis
=======================================================

This module allows users to play as Emperor Hirohito or Chiang Kai-shek in an interactive
simulation attempting to prevent the Second Sino-Japanese War from escalating into full conflict.

Features:
- Human-in-the-loop decision making using CrewAI's human_input functionality
- Two player perspectives: Japanese Emperor or Chinese Nationalist Leader
- Default mode (limited information) vs Dev mode (full transparency)
- Real-time diplomatic negotiation simulation
- Peace-oriented objectives and scoring
"""

import os
import sys
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import argparse

from crewai import Agent, Task, Crew, Process
from crewai.tools.base_tool import Tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

# Import base classes
from sino_japanese_war_simulation import SinoJapaneseWarSimulation, ConflictIntensity, DetailedEvent
from historical_events_system import HistoricalEventsManager, HistoricalEvent, EventSeverity
from multi_agent_coalition_system import MultiAgentCoalitionSystem, Topic, Actor, Coalition
from bdm_peace_calculator import PeaceWarCalculator, EMPEROR_ACTIONS, CHIANG_ACTIONS


class PlayerRole(Enum):
    EMPEROR = "emperor"
    CHIANG = "chiang"
    OBSERVER = "observer"


class SimulationMode(Enum):
    DEFAULT = "default"  # Limited information, realistic perspective
    DEV = "dev"         # Full transparency, see all agent thoughts


@dataclass
class PeaceMetrics:
    """Metrics for measuring peace achievement."""
    diplomatic_success: float = 0.0
    conflict_prevention: float = 80.0  # Starts at 80, decreases with escalation
    civilian_casualties_avoided: int = 0
    international_reputation: float = 30.0
    internal_stability: float = 20.0
    negotiation_rounds: int = 0
    peace_agreements_reached: int = 0


class InteractivePeaceSimulator(SinoJapaneseWarSimulation):
    """Interactive peace simulation allowing human players to prevent war."""

    def __init__(self, player_role: PlayerRole, simulation_mode: SimulationMode = SimulationMode.DEFAULT,
                 llm_model: str = "groq/llama3-70b-8192", start_year: int = 1930):
        super().__init__(llm_model)
        self.player_role = player_role
        self.simulation_mode = simulation_mode
        self.peace_metrics = PeaceMetrics()
        self.player_decisions: List[Dict[str, Any]] = []
        self.diplomatic_proposals: List[Dict[str, Any]] = []
        self.negotiation_history: List[str] = []

        # Initialize historical events system
        self.events_manager = HistoricalEventsManager(start_year)
        self.start_year = start_year

        # Initialize multi-agent coalition system
        self.coalition_system = MultiAgentCoalitionSystem()
        self.peace_calculator = PeaceWarCalculator()

        # Initialize coalitions
        self.coalition_system.coalitions["KMT_Government"] = Coalition("KMT_Government", "Chiang_KMT")
        self.coalition_system.coalitions["United_Front"] = Coalition("United_Front", "CCP")
        self.coalition_system.coalitions["Northern_Alliance"] = Coalition("Northern_Alliance", "Yan_Xishan")

        # Override initial conflict intensity for peace-focused simulation
        self.conflict_intensity = ConflictIntensity.DIPLOMATIC

        # Setup interactive logging
        self._setup_interactive_logging()

        # Initialize peace-oriented objectives
        self._initialize_peace_objectives()

    def _setup_interactive_logging(self):
        """Setup logging for interactive session."""
        self.session_logger = logging.getLogger("PeaceSimulator")
        
        # Create console handler for user feedback
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter('🕊️ %(message)s')
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        
        # Create file handler for detailed logs
        file_handler = logging.FileHandler(
            f"peace_simulation_{self.player_role.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        
        self.session_logger.addHandler(console_handler)
        self.session_logger.addHandler(file_handler)
        self.session_logger.setLevel(logging.DEBUG)

    def _initialize_peace_objectives(self):
        """Initialize objectives focused on peace rather than victory."""
        # Override some country profiles to be more peace-oriented
        if "Emperor" in self.game_state.countries:
            emperor_profile = self.game_state.countries["Emperor"]
            emperor_profile.domestic_policies.append("Peace preservation")
            emperor_profile.domestic_policies.append("International harmony")
        
        if "Chiang_Kai_Shek" in self.game_state.countries:
            chiang_profile = self.game_state.countries["Chiang_Kai_Shek"]
            chiang_profile.domestic_policies.append("Diplomatic resolution")
            chiang_profile.domestic_policies.append("National sovereignty through negotiation")
        
        self.session_logger.info("Peace objectives initialized - simulation goal is conflict prevention")

    def display_welcome_message(self):
        """Display welcome message and simulation setup."""
        print("\n" + "="*80)
        print("🕊️  INTERACTIVE PEACE SIMULATOR - 1937 SINO-JAPANESE CRISIS  🕊️")
        print("="*80)
        
        if self.player_role == PlayerRole.EMPEROR:
            print(f"""
📜 You are Emperor Hirohito of Japan
🗓️ Date: {self.start_year}, during economic crisis and rising militarism
🎯 Your Goal: Use your divine authority to prevent military escalation and maintain peace

Your Position:
- You have ultimate moral authority but work through military and civilian advisors
- The military factions (Tosei-ha and remaining Kodo-ha) are pushing for expansion
- International opinion matters for Japan's long-term interests
- Your decisions will determine if diplomacy can triumph over militarism

Key Challenges:
- Balance military pressure with diplomatic solutions
- Manage internal faction disputes
- Maintain imperial dignity while pursuing peace
- Consider international consequences of Japanese actions
""")
        
        elif self.player_role == PlayerRole.CHIANG:
            print(f"""
📜 You are Generalissimo Chiang Kai-shek of the Republic of China
🗓️ Date: {self.start_year}, facing Japanese militarism and internal challenges
🎯 Your Goal: Protect Chinese sovereignty while avoiding devastating full-scale war

Your Position:
- You lead the Nationalist government but face internal divisions
- Japanese forces are technologically superior
- You must balance resistance with pragmatic diplomacy
- Communist forces remain a long-term threat

Key Challenges:
- Defend Chinese territory without triggering total war
- Unite fractured Chinese factions for negotiation
- Secure international support for peaceful resolution
- Maintain leadership credibility while seeking compromise
""")
        
        mode_description = "🔍 DEV MODE: You can see all agent thoughts and plans" if self.simulation_mode == SimulationMode.DEV else "👁️ DEFAULT MODE: You see only information available to your character"
        
        print(f"""
{mode_description}

🎮 How to Play:
- You will be presented with situations requiring decisions
- Type your responses when prompted for human input
- Your goal is to prevent war escalation and achieve lasting peace
- Success is measured by diplomatic breakthroughs, not military victories

📊 Peace Metrics Tracked:
- Conflict Prevention Score (starts at 100)
- Diplomatic Success Rate
- International Reputation
- Internal Stability
""")
        print("="*80)
        input("\n🚀 Press Enter to begin the simulation...")

    def create_human_player_agent(self) -> Agent:
        """Create the human player agent with appropriate tools and context."""
        if self.player_role == PlayerRole.EMPEROR:
            return Agent(
                role="Human Player - Emperor Hirohito",
                goal="Prevent war escalation through wise imperial decisions and diplomatic leadership",
                backstory=f"""You are the human player controlling Emperor Hirohito. You have divine authority 
                in Japanese society but must work through various factions and advisors. Your goal is to use this 
                unique position to prevent the China incident from escalating into full-scale war.
                
                You can:
                - Issue imperial edicts and guidance
                - Influence military and civilian leaders
                - Engage in diplomatic initiatives
                - Balance competing faction interests
                - Make moral appeals for peace
                
                Remember: Your ultimate goal is peace, not conquest. Use your authority wisely.""",
                verbose=self.simulation_mode == SimulationMode.DEV,
                allow_delegation=True,
                llm=self.llm,
                tools=[self._create_human_emperor_tools()]
            )
        
        elif self.player_role == PlayerRole.CHIANG:
            return Agent(
                role="Human Player - Chiang Kai-shek",
                goal="Protect Chinese sovereignty through diplomatic means while preventing devastating war",
                backstory=f"""You are the human player controlling Chiang Kai-shek. You lead the Republic of China 
                but face the challenge of Japanese aggression. Your goal is to find diplomatic solutions that preserve 
                Chinese independence without triggering a war that could destroy your nation.
                
                You can:
                - Negotiate directly with Japanese representatives
                - Rally Chinese factions for unified response
                - Seek international mediation and support
                - Make strategic concessions to preserve peace
                - Build coalitions for diplomatic pressure
                
                Remember: Your ultimate goal is preserving China through peace, not martyrdom through war.""",
                verbose=self.simulation_mode == SimulationMode.DEV,
                allow_delegation=True,
                llm=self.llm,
                tools=[self._create_human_chiang_tools()]
            )

    def _create_human_emperor_tools(self) -> Tool:
        """Create tools for human player as Emperor."""
        def imperial_decision(decision_type: str, target_group: str = "", diplomatic_message: str = "", 
                            internal_directive: str = "") -> str:
            """Make imperial decisions as the human player."""
            try:
                # Record player decision
                decision_record = {
                    "timestamp": datetime.now().isoformat(),
                    "player": "Emperor",
                    "decision_type": decision_type,
                    "target_group": target_group,
                    "diplomatic_message": diplomatic_message,
                    "internal_directive": internal_directive
                }
                self.player_decisions.append(decision_record)
                
                # Calculate peace impact
                peace_impact = self._calculate_peace_impact("imperial", decision_type, diplomatic_message)
                self._update_peace_metrics(peace_impact)
                
                # Create detailed event
                event = DetailedEvent(
                    timestamp=datetime.now().isoformat(),
                    event_type="Human Imperial Decision",
                    primary_actor="Emperor",
                    secondary_actors=[target_group] if target_group else [],
                    action_description=f"Imperial {decision_type}: {diplomatic_message or internal_directive}",
                    resources_involved={"imperial_authority": 20, "diplomatic_influence": 15},
                    consequences=[f"Peace impact: {peace_impact:.1f}"],
                    international_reaction={},
                    conflict_intensity_change=-abs(peace_impact) * 0.1  # Peace decisions reduce conflict
                )
                
                self.detailed_events.append(event)
                
                # Update conflict intensity based on peace-oriented decisions
                if peace_impact > 0:
                    self._update_conflict_intensity(-0.2)  # Peace decisions reduce conflict
                
                self.session_logger.info(f"Imperial Decision Made: {decision_type} - Peace Impact: {peace_impact:.1f}")
                
                return f"Imperial decision executed successfully. Peace impact: {peace_impact:.1f}"
                
            except Exception as e:
                return f"Imperial decision failed: {str(e)}"
        
        return Tool(
            name="imperial_peace_decision",
            description="Make imperial decisions focused on peace and conflict prevention",
            func=imperial_decision
        )

    def _create_human_chiang_tools(self) -> Tool:
        """Create tools for human player as Chiang Kai-shek."""
        def nationalist_decision(strategy_type: str, negotiation_position: str = "", unity_appeal: str = "", 
                               international_request: str = "") -> str:
            """Make nationalist decisions as the human player."""
            try:
                # Record player decision
                decision_record = {
                    "timestamp": datetime.now().isoformat(),
                    "player": "Chiang_Kai_Shek",
                    "strategy_type": strategy_type,
                    "negotiation_position": negotiation_position,
                    "unity_appeal": unity_appeal,
                    "international_request": international_request
                }
                self.player_decisions.append(decision_record)
                
                # Calculate peace impact
                peace_impact = self._calculate_peace_impact("nationalist", strategy_type, negotiation_position)
                self._update_peace_metrics(peace_impact)
                
                # Create detailed event
                event = DetailedEvent(
                    timestamp=datetime.now().isoformat(),
                    event_type="Human Nationalist Decision",
                    primary_actor="Chiang_Kai_Shek",
                    secondary_actors=["Chinese_Factions", "International_Community"],
                    action_description=f"Nationalist {strategy_type}: {negotiation_position or unity_appeal}",
                    resources_involved={"political_authority": 15, "diplomatic_influence": 20},
                    consequences=[f"Peace impact: {peace_impact:.1f}"],
                    international_reaction={},
                    conflict_intensity_change=-abs(peace_impact) * 0.1
                )
                
                self.detailed_events.append(event)
                
                # Update conflict intensity based on peace-oriented decisions
                if peace_impact > 0:
                    self._update_conflict_intensity(-0.15)
                
                self.session_logger.info(f"Nationalist Decision Made: {strategy_type} - Peace Impact: {peace_impact:.1f}")
                
                return f"Nationalist strategy executed successfully. Peace impact: {peace_impact:.1f}"
                
            except Exception as e:
                return f"Nationalist decision failed: {str(e)}"
        
        return Tool(
            name="nationalist_peace_decision", 
            description="Make nationalist decisions focused on diplomatic solutions and peace",
            func=nationalist_decision
        )

    def _calculate_peace_impact(self, player_type: str, decision_type: str, message: str) -> float:
        """Calculate the peace impact of a human player decision."""
        base_impact = 0.0
        
        # Positive peace keywords
        peace_keywords = ["negotiate", "diplomacy", "peace", "compromise", "dialogue", "mediation", 
                         "cooperation", "understanding", "restraint", "withdrawal", "ceasefire"]
        
        # Negative conflict keywords
        conflict_keywords = ["attack", "invade", "retaliate", "ultimatum", "force", "military", 
                           "escalate", "strike", "advance", "occupy"]
        
        message_lower = message.lower()
        
        # Score based on keywords
        peace_score = sum(1 for keyword in peace_keywords if keyword in message_lower)
        conflict_score = sum(1 for keyword in conflict_keywords if keyword in message_lower)
        
        base_impact = (peace_score - conflict_score) * 10
        
        # Adjust based on decision type
        if decision_type in ["diplomatic_initiative", "peace_proposal", "mediation_request"]:
            base_impact += 15
        elif decision_type in ["military_restraint", "troop_withdrawal", "ceasefire_order"]:
            base_impact += 20
        elif decision_type in ["defensive_only", "negotiate_first"]:
            base_impact += 10
        elif decision_type in ["military_action", "retaliation", "escalation"]:
            base_impact -= 20
        
        return max(-50, min(50, base_impact))

    def _update_peace_metrics(self, peace_impact: float):
        """Update peace metrics based on decisions."""
        if peace_impact > 0:
            self.peace_metrics.diplomatic_success += peace_impact * 0.5
            self.peace_metrics.international_reputation += peace_impact * 0.3
            self.peace_metrics.conflict_prevention = min(100, self.peace_metrics.conflict_prevention + peace_impact * 0.2)
        else:
            self.peace_metrics.conflict_prevention += peace_impact * 0.5  # Negative impact reduces peace score
            self.peace_metrics.international_reputation += peace_impact * 0.2
        
        # Ensure metrics stay in valid ranges
        self.peace_metrics.diplomatic_success = max(0, self.peace_metrics.diplomatic_success)
        self.peace_metrics.international_reputation = max(0, min(100, self.peace_metrics.international_reputation))
        self.peace_metrics.conflict_prevention = max(0, min(100, self.peace_metrics.conflict_prevention))

    def create_interactive_tasks(self) -> List[Task]:
        """Create tasks with human input for interactive simulation."""
        tasks = []

        # Get coalition and faction status
        coalition_status = self._get_coalition_status()
        faction_dynamics = self._get_faction_dynamics()

        # Human player decision task
        human_player_agent = self.create_human_player_agent()

        player_task = Task(
            description=f"""
🎯 PEACE SIMULATION - Turn {self.game_state.current_turn}
{'='*60}

Current Situation:
- Conflict Intensity: {self.conflict_intensity.value.upper()}
- Your Peace Score: {self.peace_metrics.conflict_prevention:.1f}/100
- International Reputation: {self.peace_metrics.international_reputation:.1f}/100
- Diplomatic Success Rate: {self.peace_metrics.diplomatic_success:.1f}

Faction Dynamics:
{faction_dynamics}

Coalition Status:
{coalition_status}

Recent Developments:
{self._format_recent_events()}

Intelligence Report:
{self._generate_intelligence_report()}

🤔 YOUR DECISION REQUIRED:
As {'Emperor Hirohito' if self.player_role == PlayerRole.EMPEROR else 'Chiang Kai-shek'}, you must decide how to respond to the current crisis.

Consider these options:
1. Diplomatic Initiative - Propose direct negotiations
2. Military Restraint - Order forces to avoid escalation
3. International Mediation - Seek third-party intervention
4. Internal Consultation - Rally domestic support for peace
5. Strategic Concession - Offer compromise to prevent war
6. Public Appeal - Make statement to international community
7. Coalition Management - Influence faction alignments
8. Resource Reallocation - Shift focus between different policy areas

Your decision will significantly impact the peace process. Choose wisely!

Use your decision tool to implement your choice with detailed reasoning.
            """,
            expected_output=f"Peace-oriented decision by {'Emperor' if self.player_role == PlayerRole.EMPEROR else 'Chiang Kai-shek'} with reasoning and implementation plan",
            agent=human_player_agent,
            human_input=True  # This enables human input!
        )
        tasks.append(player_task)
        
        # Create responsive AI tasks based on player role
        ai_response_tasks = self._create_ai_response_tasks(human_player_agent)
        tasks.extend(ai_response_tasks)
        
        return tasks

    def _get_coalition_status(self) -> str:
        """Get current coalition status for display."""
        status_lines = []
        for coalition_name, coalition in self.coalition_system.coalitions.items():
            members = ", ".join(coalition.members)
            total_power = sum(
                self.coalition_system.actors[m].get_total_power()
                for m in coalition.members
                if m in self.coalition_system.actors
            )
            status_lines.append(f"- {coalition_name}: {members} (Power: {total_power:.1f})")

        if not status_lines:
            return "- No active coalitions"

        return "\n".join(status_lines)

    def _get_faction_dynamics(self) -> str:
        """Get faction dynamics based on BDM calculations."""
        if self.player_role == PlayerRole.EMPEROR:
            japan_position, japan_details = self.peace_calculator.calculate_country_intention("japan")
            lines = [f"Japan War Intention: {japan_position:.1f}/100"]

            # Show top 3 most influential factions
            sorted_factions = sorted(japan_details.items(),
                                   key=lambda x: x[1]['influence'], reverse=True)[:3]
            for faction_name, details in sorted_factions:
                lines.append(f"- {faction_name}: {details['stance']} (Influence: {details['influence']:.1f})")
        else:
            china_position, china_details = self.peace_calculator.calculate_country_intention("china")
            lines = [f"China War Readiness: {china_position:.1f}/100"]

            # Show Chinese coalition dynamics
            for actor_name, actor in list(self.coalition_system.actors.items())[:3]:
                if actor.faction_type != "military":  # Focus on Chinese actors
                    focus = max(actor.resource_allocation.get_weights().items(),
                              key=lambda x: x[1])[0].value if actor.resource_allocation.get_weights() else "balanced"
                    lines.append(f"- {actor_name}: Focus on {focus} (Coalition: {actor.coalition or 'Independent'})")

        return "\n".join(lines)

    def _format_recent_events(self) -> str:
        """Format recent events for player display."""
        if not self.detailed_events:
            return "- Marco Polo Bridge Incident has occurred\n- Tensions are rising between Japanese and Chinese forces"

        recent = self.detailed_events[-3:]
        formatted = []
        for event in recent:
            formatted.append(f"- {event.primary_actor}: {event.action_description}")

        return "\n".join(formatted)

    def _generate_intelligence_report(self) -> str:
        """Generate intelligence report based on simulation mode."""
        if self.simulation_mode == SimulationMode.DEV:
            return self._generate_full_intelligence()
        else:
            return self._generate_limited_intelligence()

    def _generate_full_intelligence(self) -> str:
        """Generate full intelligence report for dev mode."""
        report = "🔍 FULL INTELLIGENCE (DEV MODE):\n"
        
        # Show AI agent intentions
        key_factions = ["Tosei_Ha", "Kwantung_Army", "Chinese_Communists", "Zaibatsu"]
        for faction in key_factions:
            if faction in self.game_state.countries:
                profile = self.game_state.countries[faction]
                report += f"- {faction}: {profile.current_status}\n"
        
        # Show relationship tensions
        report += "\nFACTION RELATIONSHIPS:\n"
        if self.player_role == PlayerRole.EMPEROR:
            key_relations = [("Tosei_Ha", "Kodo_Ha"), ("Kwantung_Army", "Army_General_Staff"), ("Zaibatsu", "Military")]
        else:
            key_relations = [("Chinese_Communists", "Chiang_Kai_Shek"), ("Northern_Warlords", "Central_Government")]
        
        for rel in key_relations:
            if rel[0] in self.game_state.international_relations and rel[1] in self.game_state.international_relations[rel[0]]:
                value = self.game_state.international_relations[rel[0]][rel[1]]
                status = "Allied" if value > 0.3 else "Hostile" if value < -0.3 else "Neutral"
                report += f"- {rel[0]} ↔ {rel[1]}: {status} ({value:.1f})\n"
        
        return report

    def _generate_limited_intelligence(self) -> str:
        """Generate limited intelligence report for default mode."""
        report = "📋 AVAILABLE INTELLIGENCE:\n"
        
        if self.player_role == PlayerRole.EMPEROR:
            report += "- Military advisors report increasing pressure for action\n"
            report += "- Diplomatic channels with China remain open\n" 
            report += "- International observers are monitoring the situation\n"
            report += "- Domestic opinion shows mixed support for expansion vs. peace\n"
        else:
            report += "- Japanese forces are mobilizing but haven't launched full attack\n"
            report += "- Regional warlords are awaiting central government guidance\n"
            report += "- International community has expressed concern\n"
            report += "- Communist forces claim willingness to cooperate against Japan\n"
        
        return report

    def _create_ai_response_tasks(self, human_player_agent: Agent) -> List[Task]:
        """Create AI agent response tasks based on human player decisions."""
        tasks = []
        
        # Create counter-party agent (Japanese AI if player is Chiang, Chinese AI if player is Emperor)
        if self.player_role == PlayerRole.EMPEROR:
            # Player is Emperor, so create Chinese response agent
            counterpart_agent = Agent(
                role="Chinese Diplomatic Representative",
                goal="Respond to Japanese imperial initiatives while protecting Chinese interests",
                backstory="""You represent Chinese interests in negotiations with the Japanese Emperor.
                Your goal is to find peaceful solutions that preserve Chinese sovereignty and dignity.
                You should respond constructively to genuine peace initiatives while firmly rejecting
                unacceptable demands. Look for creative compromises that could prevent full-scale war.""",
                verbose=self.simulation_mode == SimulationMode.DEV,
                llm=self.llm
            )
            
            counterpart_task = Task(
                description=f"""Respond to the Japanese Emperor's latest decision and initiative.
                
                The Emperor has made a decision regarding the current crisis. Your role is to:
                1. Evaluate the sincerity and acceptability of the Japanese position
                2. Propose Chinese counter-proposals or acceptance where appropriate
                3. Identify areas of potential compromise
                4. Maintain Chinese dignity while seeking peaceful resolution
                
                Consider both immediate tactical responses and long-term strategic implications.
                Your response should be constructive if the Emperor shows genuine peace intent.""",
                expected_output="Chinese diplomatic response with evaluation of Japanese position and counter-proposals",
                agent=counterpart_agent
            )
        else:
            # Player is Chiang, so create Japanese response agent  
            counterpart_agent = Agent(
                role="Japanese Imperial Advisor",
                goal="Advise on response to Chinese initiatives while balancing military and diplomatic pressures",
                backstory="""You are an Imperial advisor trying to balance the Emperor's desire for peace
                with military faction pressure for action. Your goal is to find honorable solutions that
                satisfy Japanese strategic interests while avoiding unnecessary war. You should respond
                positively to genuine Chinese concessions while maintaining Japanese dignity.""",
                verbose=self.simulation_mode == SimulationMode.DEV,
                llm=self.llm
            )
            
            counterpart_task = Task(
                description=f"""Advise the Emperor on how to respond to Chiang Kai-shek's latest position.
                
                The Chinese leader has made a decision regarding the current crisis. Your role is to:
                1. Assess whether Chinese proposals are acceptable to Japanese interests
                2. Recommend Imperial responses that could advance peace negotiations
                3. Balance military faction expectations with diplomatic opportunities  
                4. Suggest face-saving solutions for both sides
                
                Consider how to maintain Japanese honor while pursuing peace if China shows flexibility.""",
                expected_output="Imperial advisory response with analysis of Chinese position and recommended Japanese response",
                agent=counterpart_agent
            )
        
        tasks.append(counterpart_task)
        
        # Add military faction pressure task (always present as challenge)
        if self.simulation_mode == SimulationMode.DEV:
            military_pressure_agent = Agent(
                role="Military Faction Pressure",
                goal="Represent military pressure for more aggressive action",
                backstory="""You represent the military factions pushing for decisive action.
                While you're not necessarily opposed to peace, you believe that strength and
                resolve are necessary to achieve favorable terms. You pressure for backup
                military preparations and warn against appearing weak.""",
                verbose=True,
                llm=self.llm
            )
            
            military_task = Task(
                description=f"""Express military concerns about the current diplomatic approach.
                
                Military factions are watching the peace negotiations with concern about:
                1. Whether diplomatic concessions will be seen as weakness
                2. The need to maintain military readiness as negotiation backup
                3. Domestic and international perception of military strength
                4. The risk of missing strategic opportunities
                
                Present military perspective while not directly undermining peace efforts.""",
                expected_output="Military faction concerns and pressure points regarding current diplomatic strategy",
                agent=military_pressure_agent
            )
            tasks.append(military_task)
        
        return tasks

    def run_interactive_simulation(self, max_turns: int = 15) -> Dict[str, Any]:
        """Run the interactive peace simulation."""
        self.display_welcome_message()
        
        simulation_results = {
            "simulation_type": "Interactive Peace Simulation",
            "player_role": self.player_role.value,
            "simulation_mode": self.simulation_mode.value,
            "scenario": "Second Sino-Japanese War Prevention",
            "trigger_event": self.game_state.trigger_event,
            "turns": [],
            "player_decisions": [],
            "peace_metrics_history": [],
            "final_peace_outcome": None
        }
        
        self.session_logger.info(f"Starting interactive peace simulation as {self.player_role.value}")
        
        for turn in range(max_turns):
            self.game_state.current_turn = turn + 1
            
            print(f"\n🔄 TURN {self.game_state.current_turn} - PEACE NEGOTIATION ROUND")
            print("-" * 60)
            
            try:
                # Create tasks for this turn
                tasks = self.create_interactive_tasks()
                
                # Create crew
                agents = [task.agent for task in tasks if task.agent]
                crew = Crew(
                    agents=agents,
                    tasks=tasks,
                    process=Process.sequential,
                    verbose=self.simulation_mode == SimulationMode.DEV
                )
                
                # Execute turn
                turn_result = crew.kickoff()
                
                # Record turn data
                turn_data = {
                    "turn_number": self.game_state.current_turn,
                    "conflict_intensity": self.conflict_intensity.value,
                    "peace_metrics": {
                        "conflict_prevention": self.peace_metrics.conflict_prevention,
                        "diplomatic_success": self.peace_metrics.diplomatic_success,
                        "international_reputation": self.peace_metrics.international_reputation
                    },
                    "events": [event.action_description for event in self.detailed_events[-5:]],
                    "turn_result": str(turn_result)
                }
                
                simulation_results["turns"].append(turn_data)
                simulation_results["peace_metrics_history"].append(turn_data["peace_metrics"])
                
                # Display turn summary
                self._display_turn_summary(turn_data)
                
                # Check for peace achievement or failure
                outcome = self._check_peace_outcome()
                if outcome:
                    simulation_results["final_peace_outcome"] = outcome
                    self.session_logger.info(f"Simulation ended: {outcome}")
                    break
                
                # Brief pause for readability
                input("\n⏸️  Press Enter to continue to next turn...")
                
            except KeyboardInterrupt:
                print("\n❌ Simulation interrupted by user")
                simulation_results["final_peace_outcome"] = "User Interrupted"
                break
            except Exception as e:
                self.session_logger.error(f"Turn {self.game_state.current_turn} failed: {str(e)}")
                print(f"❌ Error in turn execution: {str(e)}")
                break
        
        # Record final results
        simulation_results["player_decisions"] = self.player_decisions
        simulation_results["total_turns"] = self.game_state.current_turn
        simulation_results["final_peace_score"] = self.peace_metrics.conflict_prevention
        
        if not simulation_results["final_peace_outcome"]:
            simulation_results["final_peace_outcome"] = self._determine_final_peace_outcome()
        
        # Display final results
        self._display_final_results(simulation_results)
        
        return simulation_results

    def _display_turn_summary(self, turn_data: Dict[str, Any]):
        """Display summary of the current turn."""
        print(f"\n📊 TURN {turn_data['turn_number']} SUMMARY")
        print("-" * 40)
        print(f"Conflict Level: {turn_data['conflict_intensity'].upper()}")
        print(f"Peace Score: {turn_data['peace_metrics']['conflict_prevention']:.1f}/100")
        print(f"Diplomatic Success: {turn_data['peace_metrics']['diplomatic_success']:.1f}")
        print(f"International Reputation: {turn_data['peace_metrics']['international_reputation']:.1f}/100")

        # Show war intentions from both sides
        japan_intention, _ = self.peace_calculator.calculate_country_intention("japan")
        china_intention, _ = self.peace_calculator.calculate_country_intention("china")
        print(f"\n⚔️ War Intentions:")
        print(f"  Japan: {japan_intention:.1f}/100")
        print(f"  China: {china_intention:.1f}/100")

        current_year = self.start_year + (turn_data['turn_number'] // 4)
        years_to_goal = 1940 - current_year
        if years_to_goal > 0:
            print(f"\n📅 Years until 1940: {years_to_goal}")

        if turn_data['events']:
            print(f"\nRecent Events:")
            for event in turn_data['events']:
                print(f"  • {event}")

    def _check_peace_outcome(self) -> Optional[str]:
        """Check if simulation has reached a peace outcome (success or failure)."""
        # Get current war intentions from both sides
        japan_intention, _ = self.peace_calculator.calculate_country_intention("japan")
        china_intention, _ = self.peace_calculator.calculate_country_intention("china")

        # Determine opponent's war intention based on player role
        if self.player_role == PlayerRole.EMPEROR:
            opponent_intention = china_intention
            player_intention = japan_intention
        else:
            opponent_intention = japan_intention
            player_intention = china_intention

        # Check if we've reached 1940 (victory condition)
        current_year = self.start_year + (self.game_state.current_turn // 4)  # Assuming 4 turns per year

        if current_year >= 1940:
            # Victory condition assessment at 1940
            if opponent_intention >= 90:
                if self.peace_metrics.conflict_prevention >= 80 and player_intention <= 30:
                    return "⚠️ PYRRHIC VICTORY - Peace maintained but opponent ready for total war!"
                else:
                    return "💥 DEFEAT - Opponent achieved total war readiness despite your efforts"
            elif opponent_intention >= 70:
                if self.peace_metrics.conflict_prevention >= 70 and player_intention <= 40:
                    return "🤝 PARTIAL VICTORY - Peace maintained but high tensions remain"
                else:
                    return "⚖️ STALEMATE - Neither full peace nor war achieved"
            else:
                if self.peace_metrics.conflict_prevention >= 80 and player_intention <= 30:
                    return "🕊️ COMPLETE VICTORY - Lasting peace achieved through 1940!"
                elif self.peace_metrics.conflict_prevention >= 60 and player_intention <= 50:
                    return "✅ MINOR VICTORY - War avoided but some tensions persist"
                else:
                    return "📊 DRAW - Status quo maintained to 1940"

        # Early failure conditions (before 1940)
        if self.peace_metrics.conflict_prevention <= 20:
            return "⚔️ PEACE FAILED - Conflict escalated beyond diplomatic resolution"

        if self.conflict_intensity == ConflictIntensity.TOTAL_WAR:
            return "💥 TOTAL WAR - All diplomatic efforts have failed"

        if opponent_intention >= 95 and self.peace_metrics.conflict_prevention < 50:
            return "🚨 IMMINENT WAR - Opponent preparing for total war, peace hanging by thread"

        # Mid-game assessment (not final)
        if self.game_state.current_turn >= 12 and current_year < 1940:
            if self.peace_metrics.conflict_prevention >= 60:
                return None  # Continue playing
            elif self.peace_metrics.conflict_prevention >= 40:
                return None  # Continue playing
            else:
                return "📈 STRUGGLING - Diplomatic efforts failing, war likely"

        return None

    def _determine_final_peace_outcome(self) -> str:
        """Determine final outcome based on metrics."""
        if self.peace_metrics.conflict_prevention >= 70:
            return "🎉 PEACE CHAMPION - Excellent diplomatic leadership prevented major war"
        elif self.peace_metrics.conflict_prevention >= 50:
            return "🤝 DIPLOMATIC SUCCESS - Managed to prevent worst-case scenarios"
        elif self.peace_metrics.conflict_prevention >= 30:
            return "⚖️ MIXED RESULTS - Some success but significant challenges remain"
        else:
            return "❌ DIPLOMATIC FAILURE - Unable to prevent conflict escalation"

    def _display_final_results(self, results: Dict[str, Any]):
        """Display comprehensive final results."""
        print("\n" + "="*80)
        print("🏆 FINAL SIMULATION RESULTS")
        print("="*80)
        
        print(f"🎭 Player Role: {'Emperor Hirohito' if self.player_role == PlayerRole.EMPEROR else 'Chiang Kai-shek'}")
        print(f"🎮 Simulation Mode: {self.simulation_mode.value.upper()}")
        print(f"🔄 Turns Completed: {results['total_turns']}")
        print(f"📊 Final Peace Score: {results['final_peace_score']:.1f}/100")
        print(f"🏅 Outcome: {results['final_peace_outcome']}")
        
        print(f"\n📈 PEACE METRICS BREAKDOWN:")
        print(f"  • Conflict Prevention: {self.peace_metrics.conflict_prevention:.1f}/100")
        print(f"  • Diplomatic Success: {self.peace_metrics.diplomatic_success:.1f}")
        print(f"  • International Reputation: {self.peace_metrics.international_reputation:.1f}/100")
        print(f"  • Internal Stability: {self.peace_metrics.internal_stability:.1f}/100")
        print(f"  • Peace Agreements Reached: {self.peace_metrics.peace_agreements_reached}")
        
        print(f"\n🎯 PLAYER DECISIONS SUMMARY:")
        decision_types = {}
        for decision in self.player_decisions:
            decision_type = decision.get('decision_type') or decision.get('strategy_type', 'Unknown')
            decision_types[decision_type] = decision_types.get(decision_type, 0) + 1
        
        for decision_type, count in decision_types.items():
            print(f"  • {decision_type}: {count} times")
        
        # Performance evaluation
        print(f"\n🎖️ PERFORMANCE EVALUATION:")
        if results['final_peace_score'] >= 80:
            print("  ⭐⭐⭐ OUTSTANDING - Master Diplomat!")
            print("  Your wise leadership prevented a devastating war and saved countless lives.")
        elif results['final_peace_score'] >= 60:
            print("  ⭐⭐ GOOD - Skilled Negotiator")
            print("  You made significant progress toward peace despite difficult circumstances.")
        elif results['final_peace_score'] >= 40:
            print("  ⭐ FAIR - Learning Diplomat")
            print("  You showed some diplomatic skill but there's room for improvement.")
        else:
            print("  💭 CHALLENGING - Consider Different Approaches")
            print("  Diplomacy is difficult! Try different strategies for better outcomes.")
        
        print(f"\n💡 KEY LESSONS LEARNED:")
        self._generate_lessons_learned()
        
        print("="*80)

    def _generate_lessons_learned(self):
        """Generate lessons based on player decisions and outcomes."""
        lessons = []
        
        # Analyze player decision patterns
        diplomatic_decisions = sum(1 for d in self.player_decisions 
                                 if any(keyword in str(d).lower() for keyword in ['diplomatic', 'negotiate', 'peace']))
        military_decisions = sum(1 for d in self.player_decisions 
                               if any(keyword in str(d).lower() for keyword in ['military', 'force', 'attack']))
        
        if diplomatic_decisions > military_decisions * 2:
            lessons.append("✅ Strong commitment to diplomatic solutions")
        elif military_decisions > diplomatic_decisions:
            lessons.append("⚠️ Consider more diplomatic approaches to build trust")
        
        if self.peace_metrics.international_reputation > 70:
            lessons.append("✅ Successfully maintained international credibility")
        elif self.peace_metrics.international_reputation < 40:
            lessons.append("⚠️ International reputation suffered - consider global opinion in decisions")
        
        if self.conflict_intensity == ConflictIntensity.DIPLOMATIC:
            lessons.append("✅ Excellent conflict management - kept tensions at diplomatic level")
        elif self.conflict_intensity in [ConflictIntensity.FULL_WAR, ConflictIntensity.TOTAL_WAR]:
            lessons.append("⚠️ Conflict escalated significantly - early intervention might help")
        
        # Role-specific lessons
        if self.player_role == PlayerRole.EMPEROR:
            lessons.append("👑 Imperial authority is most effective when used for moral leadership")
        else:
            lessons.append("🇨🇳 National unity and international support are key to Chinese success")
        
        for lesson in lessons:
            print(f"  • {lesson}")

    def save_interactive_session(self, results: Dict[str, Any], filename: Optional[str] = None):
        """Save the interactive session results."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"peace_simulation_{self.player_role.value}_{timestamp}.json"
        
        # Add detailed analysis
        detailed_results = {
            **results,
            "detailed_events": [event.__dict__ for event in self.detailed_events],
            "peace_metrics_final": self.peace_metrics.__dict__,
            "negotiation_history": self.negotiation_history,
            "player_decision_analysis": self._analyze_player_decisions()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        self.session_logger.info(f"Interactive session saved to {filename}")
        print(f"💾 Session saved to: {filename}")

    def _analyze_player_decisions(self) -> Dict[str, Any]:
        """Analyze patterns in player decisions."""
        if not self.player_decisions:
            return {"total_decisions": 0}
        
        analysis = {
            "total_decisions": len(self.player_decisions),
            "decision_timeline": [],
            "peace_orientation_score": 0,
            "consistency_score": 0,
            "creativity_score": 0
        }
        
        peace_keywords = ["peace", "diplomatic", "negotiate", "compromise", "mediate", "dialogue"]
        conflict_keywords = ["military", "force", "attack", "ultimatum", "retaliate"]
        
        peace_decisions = 0
        conflict_decisions = 0
        
        for i, decision in enumerate(self.player_decisions):
            decision_text = str(decision).lower()
            
            # Count peace vs conflict orientation
            if any(keyword in decision_text for keyword in peace_keywords):
                peace_decisions += 1
            if any(keyword in decision_text for keyword in conflict_keywords):
                conflict_decisions += 1
            
            # Track timeline
            analysis["decision_timeline"].append({
                "turn": i + 1,
                "decision_type": decision.get("decision_type") or decision.get("strategy_type"),
                "peace_oriented": any(keyword in decision_text for keyword in peace_keywords)
            })
        
        # Calculate scores
        if peace_decisions + conflict_decisions > 0:
            analysis["peace_orientation_score"] = (peace_decisions / (peace_decisions + conflict_decisions)) * 100
        
        # Simple consistency measure
        decision_types = [d.get("decision_type") or d.get("strategy_type") for d in self.player_decisions]
        unique_types = len(set(decision_types))
        analysis["consistency_score"] = max(0, 100 - (unique_types * 10))  # Lower variety = higher consistency
        
        # Creativity based on unique diplomatic approaches
        creative_keywords = ["creative", "innovative", "unique", "unconventional", "novel"]
        creative_decisions = sum(1 for d in self.player_decisions 
                               if any(keyword in str(d).lower() for keyword in creative_keywords))
        analysis["creativity_score"] = min(100, creative_decisions * 25)
        
        return analysis


def main():
    """Main function for interactive peace simulation."""
    parser = argparse.ArgumentParser(description="Interactive Peace Simulation - Prevent the Second Sino-Japanese War")
    
    parser.add_argument("--role", 
                        choices=["emperor", "chiang", "observer"],
                        default="emperor",
                        help="Your role in the simulation")
    
    parser.add_argument("--mode",
                        choices=["default", "dev"],
                        default="default", 
                        help="Simulation mode: default (limited info) or dev (full transparency)")
    
    parser.add_argument("--model",
                        choices=["gpt-4", "claude-2", "groq/llama3-70b-8192", "groq/llama3-8b-8192", "groq/mixtral-8x7b-32768"],
                        default="groq/llama3-70b-8192",
                        help="LLM model for AI agents")
    
    parser.add_argument("--max_turns", type=int, default=15,
                        help="Maximum number of negotiation turns")
    
    parser.add_argument("--save", action="store_true",
                        help="Save simulation results to file")
    
    parser.add_argument("--output", type=str,
                        help="Output filename for results")
    
    args = parser.parse_args()
    
    # Validate environment
    if args.model.startswith("groq/") and not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY environment variable not set")
        return
    elif args.model == "gpt-4" and not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        return
    elif args.model == "claude-2" and not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY environment variable not set")  
        return
    
    try:
        # Initialize simulation
        player_role = PlayerRole(args.role)
        simulation_mode = SimulationMode(args.mode)
        
        simulator = InteractivePeaceSimulator(
            player_role=player_role,
            simulation_mode=simulation_mode,
            llm_model=args.model
        )
        
        # Run interactive simulation
        results = simulator.run_interactive_simulation(max_turns=args.max_turns)
        
        # Save results if requested
        if args.save:
            simulator.save_interactive_session(results, args.output)
        
        print("\n🙏 Thank you for playing the Peace Simulation!")
        print("Remember: In real history, preventing war requires wisdom, courage, and compromise from all sides.")
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye! Thanks for trying to make peace.")
    except Exception as e:
        print(f"❌ Simulation error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()