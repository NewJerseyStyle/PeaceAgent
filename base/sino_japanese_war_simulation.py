#!/usr/bin/env python3
"""
Second Sino-Japanese War Detailed Simulation using CrewAI
========================================================

This module implements a highly granular simulation of the Second Sino-Japanese War (1937-1945)
using CrewAI framework, incorporating detailed factions, social classes, and political dynamics
from both Japanese and Chinese sides.

Based on historical research and social network analysis of 1930s militarism.
"""

import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from crewai import Agent, Task, Crew, Process
from crewai.tools.base_tool import Tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

# Import the base WarAgent class
from waragent_crewai import WarAgentSimulation, CountryProfile, GameState, ActionType


class ConflictIntensity(Enum):
    DIPLOMATIC = "diplomatic"
    SKIRMISH = "skirmish"
    LIMITED_WAR = "limited_war"
    FULL_WAR = "full_war"
    TOTAL_WAR = "total_war"


class ResourceType(Enum):
    MILITARY = "military"
    ECONOMIC = "economic"
    POLITICAL = "political"
    SOCIAL = "social"
    INTERNATIONAL = "international"


@dataclass
class DetailedEvent:
    """Detailed event structure for complex historical simulation."""
    timestamp: str
    event_type: str
    primary_actor: str
    secondary_actors: List[str]
    action_description: str
    resources_involved: Dict[str, int]
    consequences: List[str]
    international_reaction: Dict[str, float]
    conflict_intensity_change: float


class SinoJapaneseWarSimulation(WarAgentSimulation):
    """Enhanced WarAgent simulation specifically for Second Sino-Japanese War."""
    
    def __init__(self, llm_model: str = "groq/llama3-70b-8192"):
        super().__init__(llm_model, "Second Sino-Japanese War")
        self.conflict_intensity = ConflictIntensity.DIPLOMATIC
        self.detailed_events: List[DetailedEvent] = []
        self.resource_flows: Dict[str, Dict[str, int]] = {}
        self.faction_relationships: Dict[str, Dict[str, float]] = {}
        self.international_pressure: Dict[str, float] = {}
        
        # Initialize specialized tracking systems
        self._initialize_resource_tracking()
        self._initialize_faction_dynamics()
        self._setup_specialized_logging()

    def _initialize_resource_tracking(self):
        """Initialize detailed resource tracking for all factions."""
        for country_name in self.game_state.countries.keys():
            self.resource_flows[country_name] = {
                "military_strength": 0,
                "economic_resources": 0,
                "political_influence": 0,
                "popular_support": 0,
                "international_standing": 0
            }

    def _initialize_faction_dynamics(self):
        """Initialize internal faction relationship tracking."""
        # Japanese internal factions
        japanese_factions = {
            "Emperor": {"Tosei_Ha": 0.6, "Kodo_Ha": 0.4, "Imperial_Advisors": 0.9},
            "Tosei_Ha": {"Kwantung_Army": 0.7, "Cabinet_Ministers": 0.8, "Zaibatsu": 0.9},
            "Kodo_Ha": {"Young_Officers": 0.9, "Tosei_Ha": -0.8, "Political_Parties": -0.7},
            "Kwantung_Army": {"China_Expeditionary_Forces": 0.8, "Army_General_Staff": 0.6},
            "Fleet_Faction": {"Naval_Treaty_Faction": -0.6, "Army_Factions": -0.3}
        }
        
        # Chinese internal factions
        chinese_factions = {
            "Chiang_Kai_Shek": {"Chinese_Central_Army": 0.9, "Chinese_Communists": -0.7, "Chinese_Warlords": 0.3},
            "Chinese_Communists": {"Communist_Guerrillas": 0.9, "Chinese_Rural_Peasants": 0.8, "Chiang_Kai_Shek": -0.6},
            "Northern_Warlords": {"Japanese_Collaborators": 0.4, "Chiang_Kai_Shek": -0.3, "Chinese_Communists": -0.5},
            "Central_Warlords": {"Chiang_Kai_Shek": 0.6, "Chinese_Communists": -0.4, "Regional_Autonomy": 0.8},
            "Wang_Jingwei_Left_KMT": {"Japanese_Forces": 0.7, "Chiang_Kai_Shek": -0.9, "Chinese_Communists": -0.8}
        }
        
        self.faction_relationships = {**japanese_factions, **chinese_factions}

    def _setup_specialized_logging(self):
        """Setup specialized logging for detailed war simulation."""
        # Create detailed event logger
        self.event_logger = logging.getLogger("SinoJapaneseWar.Events")
        handler = logging.FileHandler(f"sino_japanese_war_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.event_logger.addHandler(handler)
        self.event_logger.setLevel(logging.INFO)

    def create_specialized_agents(self):
        """Create highly specialized agents for the Second Sino-Japanese War."""
        specialized_agents = {}
        
        # Create Emperor agent with unique dynamics
        emperor_agent = Agent(
            role="Emperor Hirohito - Divine Sovereign",
            goal="Maintain imperial dignity while balancing military factions and international pressure",
            backstory=f"""You are Emperor Hirohito, the divine sovereign of Japan in 1937. The Marco Polo Bridge Incident has occurred, 
            and your military factions are pushing for full-scale war with China. You must:
            
            - Balance the competing Tosei-ha (Control Faction) and remaining Kodo-ha (Imperial Way) influences
            - Maintain your divine status while dealing with practical military and political realities
            - Consider international implications, especially relations with Western powers
            - Manage internal Japanese social pressures from various classes and groups
            
            Your decisions carry ultimate moral authority but must be implemented through your military and civilian advisors.
            You are concerned about the spiritual purity of Japan's mission versus pragmatic expansion needs.
            
            Current situation: The Kwantung Army and China Expeditionary Forces are seeking broader authorization for military action.
            Your Imperial Advisors are counseling caution, while the Tosei-ha is pushing for systematic conquest.""",
            verbose=True,
            allow_delegation=True,
            llm=self.llm,
            tools=[self._create_imperial_decision_tool()]
        )
        specialized_agents["Emperor"] = emperor_agent

        # Create Tosei-ha agent
        tosei_ha_agent = Agent(
            role="Tosei-ha Control Faction - Military Technocrats",
            goal="Establish total war state and systematic conquest of China through bureaucratic efficiency",
            backstory=f"""You are the leadership of the Tosei-ha (Control Faction), the dominant military faction after the 
            February 26 Incident. Led by figures like Tojo Hideki, you represent the modern, technocratic approach to militarism.
            
            Your faction believes in:
            - Systematic total war preparation and industrial mobilization
            - Bureaucratic control over chaotic individual terrorism
            - Gradual expansion of military control over civilian government
            - Efficient resource utilization and strategic planning
            
            You have successfully marginalized the Kodo-ha radicals and now control key positions in the Army General Staff,
            Cabinet, and industrial policy. The Marco Polo Bridge Incident provides the perfect opportunity to implement
            your total war doctrine.
            
            Current priorities:
            - Coordinate with Zaibatsu for industrial mobilization
            - Manage the autonomous actions of Kwantung Army
            - Expand systematic control over Chinese territories
            - Neutralize remaining political opposition through legal means""",
            verbose=True,
            allow_delegation=True,
            llm=self.llm,
            tools=[self._create_military_faction_tool("Tosei_Ha")]
        )
        specialized_agents["Tosei_Ha"] = tosei_ha_agent

        # Create Kwantung Army agent
        kwantung_army_agent = Agent(
            role="Kwantung Army - Autonomous Field Command",
            goal="Expand Japanese control through aggressive field initiatives and fait accompli tactics",
            backstory=f"""You are the Kwantung Army command in Manchuria, the most autonomous and aggressive Japanese military force.
            You have a history of successful gekokujo (insubordination) tactics, having engineered the Mukden Incident in 1931
            and established Manchukuo.
            
            Your operational philosophy:
            - Create facts on the ground that Tokyo must accept
            - Aggressive expansion to secure resources and strategic position
            - Coordinate with but maintain independence from central command
            - Exploit Chinese internal divisions and weakness
            
            The Marco Polo Bridge Incident has started, and you see this as an opportunity to:
            - Expand operations beyond your Manchurian base
            - Coordinate with China Expeditionary Forces for broader campaign
            - Secure additional resource bases in North China
            - Demonstrate continued military effectiveness to justify autonomy
            
            You maintain complex relationships with both Chinese warlords (some collaborative, some hostile) 
            and must balance local initiatives with broader Japanese strategic goals.""",
            verbose=True,
            allow_delegation=True,
            llm=self.llm,
            tools=[self._create_field_army_tool("Kwantung_Army")]
        )
        specialized_agents["Kwantung_Army"] = kwantung_army_agent

        # Create Chiang Kai-shek agent
        chiang_agent = Agent(
            role="Chiang Kai-shek - Nationalist Leader",
            goal="Defend Chinese sovereignty while balancing internal enemies and building modern state",
            backstory=f"""You are Chiang Kai-shek, leader of the Republic of China and the Nationalist (KMT) government.
            The Marco Polo Bridge Incident has forced you into a war you are not fully prepared for, while you still
            face internal Communist and warlord challenges.
            
            Your strategic situation:
            - Your German-trained Central Army is your most reliable force but limited in size
            - You must balance resistance against Japan with suppression of Chinese Communists
            - Regional warlords have varying degrees of loyalty and capability
            - International support is limited but potentially crucial
            - Your capital Nanjing is vulnerable to Japanese attack
            
            Your strategic dilemmas:
            - Should you fully commit to war with Japan or seek negotiated settlement?
            - Can you trust the Communists in a united front against Japan?
            - How much autonomy should you grant to regional commanders?
            - Should you prioritize mobile defense or static defense of key cities?
            
            You have spent years trying to modernize China and centralize authority, but the Japanese invasion
            threatens to undo all progress. Your decisions will determine China's survival as an independent nation.""",
            verbose=True,
            allow_delegation=True,
            llm=self.llm,
            tools=[self._create_nationalist_command_tool()]
        )
        specialized_agents["Chiang_Kai_Shek"] = chiang_agent

        # Create Chinese Communist agent
        communist_agent = Agent(
            role="Chinese Communist Party - Revolutionary Force",
            goal="Build revolutionary base while fighting Japanese invasion and preparing for eventual civil war",
            backstory=f"""You are the Chinese Communist Party leadership, including Mao Zedong and other key figures.
            The Japanese invasion creates both opportunities and challenges for your revolutionary movement.
            
            Your strategic situation:
            - You have been forced into united front with the KMT after Xi'an Incident
            - Your main strength lies in guerrilla warfare and peasant mobilization
            - You control limited but growing base areas in rural regions
            - You have some Soviet support but must be largely self-reliant
            - You see the war as opportunity to build strength while KMT fights Japanese
            
            Your dual objectives:
            - Fight genuine resistance against Japanese invasion to build legitimacy
            - Preserve and expand Communist forces for eventual showdown with KMT
            - Mobilize peasant masses through land reform and social revolution
            - Develop guerrilla warfare capabilities in Japanese rear areas
            
            The United Front is tactical - you know that civil war will resume after Japanese defeat.
            Your challenge is to appear patriotic while building revolutionary capacity.""",
            verbose=True,
            allow_delegation=True,
            llm=self.llm,
            tools=[self._create_communist_organization_tool()]
        )
        specialized_agents["Chinese_Communists"] = communist_agent

        # Create Zaibatsu agent
        zaibatsu_agent = Agent(
            role="Zaibatsu Financial Conglomerates - Industrial Power",
            goal="Maximize profits from war economy while supporting national expansion for resource access",
            backstory=f"""You represent the major Japanese industrial and financial conglomerates - Mitsui, Mitsubishi, 
            Sumitomo, and others. The escalating China conflict presents enormous business opportunities alongside risks.
            
            Your economic position:
            - You control the majority of Japanese heavy industry and finance
            - You have extensive overseas investments and trade networks
            - You need raw materials and markets that expansion can provide
            - You profit enormously from military contracts and war production
            
            Your strategic considerations:
            - Support military expansion that opens new markets and resources
            - Maintain international business relationships where possible
            - Ensure stable labor supply and production capacity
            - Balance cooperation with military demands versus business autonomy
            
            The China war offers opportunities to:
            - Secure raw material sources (iron ore, coal, etc.)
            - Develop new industrial bases in occupied territories
            - Expand market share in Asia-Pacific region
            - Strengthen position versus Western business competitors
            
            However, you also worry about international economic isolation and resource constraints 
            if the war expands beyond manageable limits.""",
            verbose=True,
            allow_delegation=True,
            llm=self.llm,
            tools=[self._create_economic_power_tool()]
        )
        specialized_agents["Zaibatsu"] = zaibatsu_agent

        return specialized_agents

    def _create_imperial_decision_tool(self) -> Tool:
        """Create specialized tool for imperial decision making."""
        def make_imperial_decision(decision_type: str, target_faction: str = "", details: str = "", moral_justification: str = "") -> str:
            """Make imperial decisions with moral authority and factional balance considerations."""
            try:
                decision_data = {
                    "emperor_decision": decision_type,
                    "target_faction": target_faction,
                    "details": details,
                    "moral_justification": moral_justification,
                    "divine_authority_used": True,
                    "turn": self.game_state.current_turn
                }
                
                # Calculate impact on faction relationships
                if target_faction in self.game_state.countries:
                    faction_profile = self.game_state.countries[target_faction]
                    if decision_type == "approve":
                        self._modify_relationship("Emperor", target_faction, 0.2)
                    elif decision_type == "disapprove":
                        self._modify_relationship("Emperor", target_faction, -0.2)
                
                event = DetailedEvent(
                    timestamp=datetime.now().isoformat(),
                    event_type="Imperial Decision",
                    primary_actor="Emperor",
                    secondary_actors=[target_faction] if target_faction else [],
                    action_description=f"{decision_type}: {details}",
                    resources_involved={"divine_authority": 10, "political_influence": 15},
                    consequences=[f"Imperial approval affects {target_faction} legitimacy"],
                    international_reaction={},
                    conflict_intensity_change=0.1 if decision_type == "approve_military_action" else 0.0
                )
                
                self.detailed_events.append(event)
                self.event_logger.info(f"Imperial Decision: {decision_type} - {details}")
                
                return f"Imperial decision executed: {decision_type} regarding {target_faction}. Moral justification: {moral_justification}"
                
            except Exception as e:
                return f"Imperial decision failed: {str(e)}"
        
        return Tool(
            name="imperial_decision_tool",
            description="Make imperial decisions affecting military factions and national policy",
            func=make_imperial_decision
        )

    def _create_military_faction_tool(self, faction_name: str) -> Tool:
        """Create specialized tool for military faction actions."""
        def execute_military_action(action_type: str, target: str = "", resource_commitment: int = 50, strategic_objective: str = "") -> str:
            """Execute military faction actions with resource tracking."""
            try:
                if faction_name not in self.resource_flows:
                    self.resource_flows[faction_name] = {"military_strength": 100, "political_influence": 50}
                
                # Check resource availability
                if self.resource_flows[faction_name]["military_strength"] < resource_commitment:
                    return f"Insufficient military resources for {action_type}. Available: {self.resource_flows[faction_name]['military_strength']}, Required: {resource_commitment}"
                
                # Execute action and update resources
                self.resource_flows[faction_name]["military_strength"] -= resource_commitment
                
                event = DetailedEvent(
                    timestamp=datetime.now().isoformat(),
                    event_type="Military Action",
                    primary_actor=faction_name,
                    secondary_actors=[target] if target else [],
                    action_description=f"{action_type} with {resource_commitment} military units targeting {target}",
                    resources_involved={"military_strength": resource_commitment},
                    consequences=[f"Military pressure on {target}", "Resource depletion"],
                    international_reaction={},
                    conflict_intensity_change=0.2
                )
                
                self.detailed_events.append(event)
                self._update_conflict_intensity(0.2)
                
                return f"{faction_name} successfully executed {action_type} against {target}. Strategic objective: {strategic_objective}"
                
            except Exception as e:
                return f"Military action failed: {str(e)}"
        
        return Tool(
            name=f"{faction_name}_military_tool",
            description=f"Execute military actions for {faction_name} with resource tracking",
            func=execute_military_action
        )

    def _create_field_army_tool(self, army_name: str) -> Tool:
        """Create specialized tool for field army autonomous actions."""
        def execute_field_operation(operation_type: str, target_location: str = "", local_resources: int = 30, gekokujo_level: float = 0.5) -> str:
            """Execute field army operations with gekokujo (insubordination) mechanics."""
            try:
                # Field armies have more autonomy but risk central disapproval
                autonomy_bonus = gekokujo_level * 20
                central_approval_risk = gekokujo_level * -0.3
                
                if army_name not in self.resource_flows:
                    self.resource_flows[army_name] = {"military_strength": 80, "local_support": 60}
                
                self.resource_flows[army_name]["military_strength"] -= local_resources
                
                # Impact relationships with central command based on gekokujo level
                if gekokujo_level > 0.7:
                    self._modify_relationship(army_name, "Army_General_Staff", central_approval_risk)
                    self._modify_relationship(army_name, "Tosei_Ha", central_approval_risk)
                
                event = DetailedEvent(
                    timestamp=datetime.now().isoformat(),
                    event_type="Field Operation",
                    primary_actor=army_name,
                    secondary_actors=[target_location],
                    action_description=f"{operation_type} at {target_location} with gekokujo level {gekokujo_level}",
                    resources_involved={"military_strength": local_resources, "autonomy": int(autonomy_bonus)},
                    consequences=[f"Autonomous action creates facts on ground", "Potential central command tension"],
                    international_reaction={},
                    conflict_intensity_change=gekokujo_level * 0.3
                )
                
                self.detailed_events.append(event)
                self._update_conflict_intensity(gekokujo_level * 0.3)
                
                return f"{army_name} executed {operation_type} at {target_location}. Autonomy bonus: {autonomy_bonus:.1f}, Central risk: {central_approval_risk:.1f}"
                
            except Exception as e:
                return f"Field operation failed: {str(e)}"
        
        return Tool(
            name=f"{army_name}_field_operation_tool",
            description=f"Execute autonomous field operations for {army_name} with gekokujo mechanics",
            func=execute_field_operation
        )

    def _create_nationalist_command_tool(self) -> Tool:
        """Create specialized tool for Chinese Nationalist command decisions."""
        def execute_nationalist_strategy(strategy_type: str, target_region: str = "", resource_allocation: Dict[str, int] = None, unity_priority: float = 0.5) -> str:
            """Execute Nationalist strategic decisions balancing multiple internal and external threats."""
            try:
                if resource_allocation is None:
                    resource_allocation = {"central_army": 40, "regional_forces": 30, "communist_containment": 20}
                
                # Chiang must balance multiple priorities
                total_resources = sum(resource_allocation.values())
                if total_resources > 100:
                    return f"Resource over-allocation detected: {total_resources}%. Please allocate max 100% of available resources."
                
                # Calculate effectiveness based on unity with other Chinese factions
                unity_bonus = unity_priority * 25
                communist_tension = (1 - unity_priority) * 15
                
                event = DetailedEvent(
                    timestamp=datetime.now().isoformat(),
                    event_type="Nationalist Strategy",
                    primary_actor="Chiang_Kai_Shek",
                    secondary_actors=[target_region, "Chinese_Central_Army"],
                    action_description=f"{strategy_type} in {target_region} with unity priority {unity_priority}",
                    resources_involved=resource_allocation,
                    consequences=[f"Unity bonus: {unity_bonus:.1f}%", f"Communist tension: {communist_tension:.1f}%"],
                    international_reaction={},
                    conflict_intensity_change=0.15
                )
                
                self.detailed_events.append(event)
                
                return f"Nationalist strategy {strategy_type} implemented. Unity bonus: {unity_bonus:.1f}%, Resource allocation: {resource_allocation}"
                
            except Exception as e:
                return f"Nationalist strategy failed: {str(e)}"
        
        return Tool(
            name="nationalist_command_tool",
            description="Execute Nationalist strategic decisions with internal faction balance",
            func=execute_nationalist_strategy
        )

    def _create_communist_organization_tool(self) -> Tool:
        """Create specialized tool for Communist organizational and guerrilla activities."""
        def execute_communist_strategy(strategy_type: str, target_area: str = "", mass_mobilization: int = 50, guerrilla_effectiveness: int = 70) -> str:
            """Execute Communist strategy combining mass mobilization and guerrilla warfare."""
            try:
                # Communists excel at mass mobilization and guerrilla warfare
                peasant_support_bonus = mass_mobilization * 0.8
                guerrilla_multiplier = guerrilla_effectiveness / 50.0
                
                # But must balance open resistance with preservation of forces
                exposure_risk = (mass_mobilization + guerrilla_effectiveness) / 200.0 * 30
                
                event = DetailedEvent(
                    timestamp=datetime.now().isoformat(),
                    event_type="Communist Organization",
                    primary_actor="Chinese_Communists",
                    secondary_actors=[target_area, "Communist_Guerrillas", "Chinese_Rural_Peasants"],
                    action_description=f"{strategy_type} in {target_area} with {mass_mobilization}% mobilization",
                    resources_involved={"mass_support": mass_mobilization, "guerrilla_capacity": guerrilla_effectiveness},
                    consequences=[f"Peasant support: +{peasant_support_bonus:.1f}", f"Exposure risk: {exposure_risk:.1f}%"],
                    international_reaction={},
                    conflict_intensity_change=0.1
                )
                
                self.detailed_events.append(event)
                
                # Update Communist strength based on mass mobilization success
                if "Chinese_Communists" in self.resource_flows:
                    self.resource_flows["Chinese_Communists"]["popular_support"] = min(100, 
                        self.resource_flows["Chinese_Communists"].get("popular_support", 50) + peasant_support_bonus / 2)
                
                return f"Communist strategy {strategy_type} executed. Peasant support bonus: {peasant_support_bonus:.1f}, Guerrilla multiplier: {guerrilla_multiplier:.1f}x"
                
            except Exception as e:
                return f"Communist strategy failed: {str(e)}"
        
        return Tool(
            name="communist_organization_tool", 
            description="Execute Communist mass mobilization and guerrilla strategies",
            func=execute_communist_strategy
        )

    def _create_economic_power_tool(self) -> Tool:
        """Create specialized tool for Zaibatsu economic influence."""
        def execute_economic_strategy(strategy_type: str, target_sector: str = "", investment_level: int = 50, international_risk: float = 0.3) -> str:
            """Execute Zaibatsu economic strategies balancing profit and national interest."""
            try:
                # Zaibatsu can provide economic resources but face international risks
                economic_boost = investment_level * 1.2
                international_penalty = international_risk * 20
                
                # Calculate net benefit
                net_benefit = economic_boost - international_penalty
                
                event = DetailedEvent(
                    timestamp=datetime.now().isoformat(),
                    event_type="Economic Strategy",
                    primary_actor="Zaibatsu",
                    secondary_actors=[target_sector, "Japanese_Economy"],
                    action_description=f"{strategy_type} in {target_sector} with {investment_level}% investment",
                    resources_involved={"economic_investment": investment_level, "industrial_capacity": int(economic_boost)},
                    consequences=[f"Economic boost: {economic_boost:.1f}", f"International penalty: -{international_penalty:.1f}"],
                    international_reaction={"Western_Powers": -international_risk, "Asian_Markets": international_risk * 0.5},
                    conflict_intensity_change=0.05
                )
                
                self.detailed_events.append(event)
                
                # Update economic resources
                if "Zaibatsu" in self.resource_flows:
                    self.resource_flows["Zaibatsu"]["economic_resources"] = min(200,
                        self.resource_flows["Zaibatsu"].get("economic_resources", 100) + net_benefit / 2)
                
                return f"Economic strategy {strategy_type} executed. Net economic benefit: {net_benefit:.1f}"
                
            except Exception as e:
                return f"Economic strategy failed: {str(e)}"
        
        return Tool(
            name="zaibatsu_economic_tool",
            description="Execute Zaibatsu economic strategies with international risk considerations",
            func=execute_economic_strategy
        )

    def _modify_relationship(self, actor1: str, actor2: str, change: float):
        """Modify relationship between two actors."""
        if actor1 in self.game_state.international_relations and actor2 in self.game_state.international_relations[actor1]:
            old_value = self.game_state.international_relations[actor1][actor2]
            new_value = max(-1.0, min(1.0, old_value + change))
            self.game_state.international_relations[actor1][actor2] = new_value
            self.game_state.international_relations[actor2][actor1] = new_value
            
            self.event_logger.info(f"Relationship change: {actor1} <-> {actor2}: {old_value:.2f} -> {new_value:.2f}")

    def _update_conflict_intensity(self, change: float):
        """Update overall conflict intensity."""
        intensity_values = list(ConflictIntensity)
        current_index = intensity_values.index(self.conflict_intensity)
        
        # Simple escalation/de-escalation logic
        if change > 0.3 and current_index < len(intensity_values) - 1:
            self.conflict_intensity = intensity_values[current_index + 1]
            self.event_logger.info(f"Conflict intensity escalated to {self.conflict_intensity.value}")
        elif change < -0.3 and current_index > 0:
            self.conflict_intensity = intensity_values[current_index - 1]
            self.event_logger.info(f"Conflict intensity de-escalated to {self.conflict_intensity.value}")

    def create_detailed_turn_tasks(self) -> List[Task]:
        """Create highly detailed tasks for Second Sino-Japanese War simulation."""
        specialized_agents = self.create_specialized_agents()
        tasks = []
        
        # Emperor's strategic decision task
        emperor_task = Task(
            description=f"""As Emperor Hirohito, make crucial decisions about the escalating China conflict.
            
            Current Situation (Turn {self.game_state.current_turn}):
            - Conflict Intensity: {self.conflict_intensity.value}
            - Recent Events: {[event.action_description for event in self.detailed_events[-3:]] if self.detailed_events else 'Initial incident'}
            - Military Pressure: Tosei-ha and Kwantung Army are pushing for expanded operations
            - International Concerns: Western powers are monitoring Japanese actions
            
            Consider the following factors in your decision:
            1. Balance between Tosei-ha efficiency demands and remaining Kodo-ha spiritual concerns
            2. Potential international diplomatic consequences
            3. Resource commitments required for different levels of military action
            4. Impact on imperial dignity and divine authority
            5. Long-term strategic implications for Japanese empire
            
            Make specific decisions about:
            - Level of military authorization to grant
            - Diplomatic approaches to pursue
            - Internal faction management
            - International messaging strategy
            
            Use your imperial decision tool to formalize your choices.""",
            expected_output="Imperial decisions on military authorization, diplomatic strategy, and faction management",
            agent=specialized_agents["Emperor"]
        )
        tasks.append(emperor_task)

        # Tosei-ha strategic planning task
        tosei_ha_task = Task(
            description=f"""As the Tosei-ha Control Faction, develop systematic strategies for the China conflict.
            
            Your current position:
            - You have dominant influence in Army General Staff and Cabinet
            - Zaibatsu cooperation provides industrial capacity
            - Kwantung Army and China Expeditionary Forces are your field instruments
            - You must manage autonomous tendencies of field commands
            
            Develop strategies for:
            1. Systematic conquest and occupation of Chinese territories
            2. Resource mobilization and industrial coordination with Zaibatsu
            3. Management of field army autonomy while maintaining strategic coherence
            4. Political control expansion over remaining civilian institutions
            5. International diplomatic cover for military operations
            
            Your total war doctrine requires careful balance between military effectiveness
            and sustainable resource utilization. Consider both immediate tactical needs
            and long-term strategic objectives.
            
            Use your military faction tool to implement specific operations.""",
            expected_output="Systematic military strategy with resource allocation and field command coordination",
            agent=specialized_agents["Tosei_Ha"]
        )
        tasks.append(tosei_ha_task)

        # Kwantung Army autonomous operations task
        kwantung_task = Task(
            description=f"""As Kwantung Army command, execute field operations using your proven gekokujo tactics.
            
            Your operational situation:
            - You control Manchukuo as your base of operations
            - You have successful history of autonomous action (Mukden Incident)
            - Central command expects results but may not approve methods
            - Local Chinese warlord relationships are complex and shifting
            
            Plan and execute operations considering:
            1. Expansion of control into North China proper
            2. Coordination with China Expeditionary Forces
            3. Exploitation of Chinese warlord divisions
            4. Resource extraction from occupied territories
            5. Balance between autonomy and central coordination
            
            Your gekokujo level determines your independence but affects relationship
            with central command. Higher autonomy may achieve better local results
            but risks central disapproval.
            
            Use your field operation tool to execute specific operations with chosen gekokujo level.""",
            expected_output="Field operations plan with gekokujo tactics and local resource exploitation",
            agent=specialized_agents["Kwantung_Army"]
        )
        tasks.append(kwantung_task)

        # Chiang Kai-shek strategic response task
        chiang_task = Task(
            description=f"""As Chiang Kai-shek, develop Chinese resistance strategy against Japanese invasion.
            
            Your strategic challenges:
            - Japanese forces are technologically superior and better equipped
            - Your Central Army is limited but most reliable
            - Regional warlords have questionable loyalty and varying capabilities
            - Communists are both potential allies and future enemies
            - International support is limited but potentially crucial
            
            Develop strategies addressing:
            1. Military defense priorities - which territories to hold vs. sacrifice
            2. Regional warlord coordination and loyalty management
            3. United Front policy with Communists - cooperation level and limits
            4. International diplomatic efforts for support and mediation
            5. Resource allocation between anti-Japanese and anti-Communist priorities
            
            Your unity priority level affects cooperation with other Chinese factions
            but may compromise your long-term position against Communists.
            Consider both immediate survival and post-war political position.
            
            Use your nationalist command tool to implement strategic decisions.""",
            expected_output="Chinese resistance strategy with faction coordination and resource priorities",
            agent=specialized_agents["Chiang_Kai_Shek"]
        )
        tasks.append(chiang_task)

        # Communist Party dual strategy task
        communist_task = Task(
            description=f"""As Chinese Communist Party leadership, balance resistance against Japan with revolutionary objectives.
            
            Your strategic position:
            - United Front with KMT is tactical necessity but temporary arrangement
            - Your strength lies in mass mobilization and guerrilla warfare
            - Rural base areas provide foundation for expansion
            - Soviet support is limited and unreliable
            - Long-term goal remains Communist revolution
            
            Develop strategies for:
            1. Genuine anti-Japanese resistance to build patriotic legitimacy
            2. Mass mobilization and peasant organization in base areas
            3. Guerrilla warfare development behind Japanese lines
            4. Preservation and expansion of Communist forces for future civil war
            5. Balance between open cooperation with KMT and maintaining independence
            
            Your mass mobilization level affects immediate effectiveness but increases
            exposure to both Japanese and KMT suppression. Guerrilla effectiveness
            multiplies your limited conventional forces.
            
            Use your communist organization tool to implement dual strategy.""",
            expected_output="Communist dual strategy balancing anti-Japanese resistance with revolutionary preparation",
            agent=specialized_agents["Chinese_Communists"]
        )
        tasks.append(communist_task)

        # Zaibatsu economic strategy task
        zaibatsu_task = Task(
            description=f"""As Zaibatsu conglomerates, maximize economic opportunities from the China conflict while managing risks.
            
            Your economic position:
            - You control majority of Japanese industrial and financial capacity
            - War provides enormous opportunities for military contracts and expansion
            - International business relationships may be jeopardized by aggressive expansion
            - Access to Chinese raw materials and markets is strategically valuable
            
            Develop economic strategies for:
            1. Industrial mobilization for war production and military contracts
            2. Resource extraction and market development in occupied Chinese territories
            3. Balance between cooperation with military demands and business autonomy
            4. International relationship management to minimize economic isolation
            5. Investment in war-related industries vs. civilian economic development
            
            Your international risk level affects potential economic penalties from
            Western powers but higher investment may secure better position in
            Japanese-controlled Asian markets.
            
            Use your economic power tool to implement investment and development strategies.""",
            expected_output="Economic mobilization strategy balancing war profits with international business risks",
            agent=specialized_agents["Zaibatsu"]
        )
        tasks.append(zaibatsu_task)

        return tasks

    def run_detailed_simulation(self, max_turns: int = 20) -> Dict[str, Any]:
        """Run detailed Second Sino-Japanese War simulation with specialized mechanics."""
        self.logger.info(f"Starting detailed Second Sino-Japanese War simulation")
        self.logger.info(f"Trigger event: {self.game_state.trigger_event}")
        
        simulation_results = {
            "scenario": "Second Sino-Japanese War - Detailed",
            "trigger_event": self.game_state.trigger_event,
            "initial_conflict_intensity": self.conflict_intensity.value,
            "turns": [],
            "detailed_events": [],
            "resource_tracking": {},
            "faction_dynamics": {},
            "final_state": None
        }
        
        for turn in range(max_turns):
            self.game_state.current_turn = turn + 1
            self.logger.info(f"\n=== TURN {self.game_state.current_turn} ===")
            self.logger.info(f"Current conflict intensity: {self.conflict_intensity.value}")
            
            # Create specialized tasks for this turn
            tasks = self.create_detailed_turn_tasks()
            
            # Create crew with specialized agents
            specialized_agents = self.create_specialized_agents()
            crew = Crew(
                agents=list(specialized_agents.values()),
                tasks=tasks,
                process=Process.sequential,
                verbose=True
            )
            
            # Execute the turn
            try:
                turn_result = crew.kickoff()
                
                # Record detailed turn data
                turn_data = {
                    "turn_number": self.game_state.current_turn,
                    "conflict_intensity": self.conflict_intensity.value,
                    "events": [asdict(event) for event in self.detailed_events[-10:]] if self.detailed_events else [],
                    "resource_flows": self.resource_flows.copy(),
                    "faction_relationships": self._get_current_faction_relationships(),
                    "international_relations": self.game_state.international_relations.copy(),
                    "crew_output": str(turn_result)
                }
                
                simulation_results["turns"].append(turn_data)
                
                # Check for end conditions
                if self._check_decisive_outcome():
                    outcome = self._determine_detailed_outcome()
                    self.logger.info(f"Simulation ended: {outcome}")
                    break
                    
                # Update global stability based on conflict intensity
                if self.conflict_intensity == ConflictIntensity.TOTAL_WAR:
                    self.game_state.global_stability = max(0.1, self.game_state.global_stability - 0.2)
                elif self.conflict_intensity == ConflictIntensity.FULL_WAR:
                    self.game_state.global_stability = max(0.2, self.game_state.global_stability - 0.1)
                
            except Exception as e:
                self.logger.error(f"Turn {self.game_state.current_turn} execution failed: {str(e)}")
                break
        
        # Record final detailed state
        simulation_results["final_state"] = {
            "final_conflict_intensity": self.conflict_intensity.value,
            "final_relationships": self.game_state.international_relations,
            "final_resources": self.resource_flows,
            "total_turns": self.game_state.current_turn,
            "total_events": len(self.detailed_events),
            "outcome": self._determine_detailed_outcome()
        }
        
        simulation_results["detailed_events"] = [asdict(event) for event in self.detailed_events]
        simulation_results["resource_tracking"] = self.resource_flows
        simulation_results["faction_dynamics"] = self.faction_relationships
        
        return simulation_results

    def _get_current_faction_relationships(self) -> Dict[str, Dict[str, float]]:
        """Get current state of all faction relationships."""
        relationships = {}
        for faction1 in self.faction_relationships:
            relationships[faction1] = {}
            for faction2, relationship in self.faction_relationships[faction1].items():
                if faction1 in self.game_state.international_relations and faction2 in self.game_state.international_relations[faction1]:
                    relationships[faction1][faction2] = self.game_state.international_relations[faction1][faction2]
                else:
                    relationships[faction1][faction2] = relationship
        return relationships

    def _check_decisive_outcome(self) -> bool:
        """Check if simulation has reached a decisive outcome."""
        # Check for total Japanese victory
        japanese_strength = sum([
            self.resource_flows.get(faction, {}).get("military_strength", 0)
            for faction in ["Tosei_Ha", "Kwantung_Army", "China_Expeditionary_Forces"]
        ])
        
        chinese_strength = sum([
            self.resource_flows.get(faction, {}).get("military_strength", 0)
            for faction in ["Chiang_Kai_Shek", "Chinese_Communists", "Chinese_Central_Army"]
        ])
        
        # Decisive victory conditions
        if japanese_strength > chinese_strength * 3 and self.conflict_intensity == ConflictIntensity.TOTAL_WAR:
            return True
        elif chinese_strength > japanese_strength * 2 and len(self.detailed_events) > 50:
            return True
        elif self.game_state.global_stability < 0.1:
            return True
        elif self.game_state.current_turn >= 15 and self.conflict_intensity == ConflictIntensity.TOTAL_WAR:
            return True
        
        return False

    def _determine_detailed_outcome(self) -> str:
        """Determine the detailed outcome of the simulation."""
        japanese_control = sum([
            self.resource_flows.get(faction, {}).get("military_strength", 0)
            for faction in ["Tosei_Ha", "Kwantung_Army", "China_Expeditionary_Forces"]
        ])
        
        chinese_resistance = sum([
            self.resource_flows.get(faction, {}).get("military_strength", 0) +
            self.resource_flows.get(faction, {}).get("popular_support", 0)
            for faction in ["Chiang_Kai_Shek", "Chinese_Communists"]
        ])
        
        communist_strength = self.resource_flows.get("Chinese_Communists", {}).get("popular_support", 0)
        
        if japanese_control > chinese_resistance * 2:
            return "Japanese Military Victory - China occupied, puppet governments established"
        elif communist_strength > 70:
            return "Chinese Communist Victory - Popular revolution successful, Japan expelled"
        elif chinese_resistance > japanese_control:
            return "Chinese Nationalist Victory - Japanese expansion halted, territorial integrity preserved"
        elif self.conflict_intensity == ConflictIntensity.TOTAL_WAR:
            return "Total War Stalemate - Prolonged brutal conflict, massive casualties, international intervention likely"
        else:
            return "Limited Conflict - Regional tensions continue, no decisive resolution"

    def export_detailed_analysis(self, results: Dict[str, Any], filename: Optional[str] = None):
        """Export detailed analysis including network dynamics and resource flows."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sino_japanese_war_detailed_{timestamp}.json"
        
        # Add network analysis
        network_analysis = {
            "faction_influence_scores": self._calculate_faction_influence(),
            "relationship_stability": self._analyze_relationship_stability(),
            "resource_flow_patterns": self._analyze_resource_patterns(),
            "conflict_escalation_timeline": self._create_escalation_timeline()
        }
        
        results["network_analysis"] = network_analysis
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Detailed analysis exported to {filename}")

    def _calculate_faction_influence(self) -> Dict[str, float]:
        """Calculate influence scores for all factions."""
        influence_scores = {}
        
        for faction in self.game_state.countries:
            military_score = self.resource_flows.get(faction, {}).get("military_strength", 0) * 0.3
            economic_score = self.resource_flows.get(faction, {}).get("economic_resources", 0) * 0.25
            political_score = self.resource_flows.get(faction, {}).get("political_influence", 0) * 0.25
            popular_score = self.resource_flows.get(faction, {}).get("popular_support", 0) * 0.2
            
            influence_scores[faction] = military_score + economic_score + political_score + popular_score
        
        return influence_scores

    def _analyze_relationship_stability(self) -> Dict[str, float]:
        """Analyze stability of relationships between factions."""
        stability_scores = {}
        
        for faction1 in self.game_state.international_relations:
            stability_sum = 0
            relationship_count = 0
            
            for faction2, relationship in self.game_state.international_relations[faction1].items():
                stability_sum += abs(relationship)  # Higher absolute values indicate more stable (either strongly positive or negative)
                relationship_count += 1
            
            if relationship_count > 0:
                stability_scores[faction1] = stability_sum / relationship_count
            else:
                stability_scores[faction1] = 0.0
        
        return stability_scores

    def _analyze_resource_patterns(self) -> Dict[str, Any]:
        """Analyze resource flow patterns throughout the simulation."""
        patterns = {
            "resource_concentration": {},
            "resource_depletion_rate": {},
            "resource_growth_sectors": []
        }
        
        for faction, resources in self.resource_flows.items():
            total_resources = sum(resources.values())
            patterns["resource_concentration"][faction] = total_resources
            
            # Calculate resource efficiency (resources per military action)
            faction_events = [e for e in self.detailed_events if e.primary_actor == faction]
            if faction_events:
                patterns["resource_depletion_rate"][faction] = total_resources / len(faction_events)
            else:
                patterns["resource_depletion_rate"][faction] = total_resources
        
        return patterns

    def _create_escalation_timeline(self) -> List[Dict[str, Any]]:
        """Create timeline of conflict escalation events."""
        timeline = []
        
        for event in self.detailed_events:
            if event.conflict_intensity_change > 0.1:
                timeline.append({
                    "timestamp": event.timestamp,
                    "event_description": event.action_description,
                    "primary_actor": event.primary_actor,
                    "escalation_level": event.conflict_intensity_change,
                    "international_reaction": event.international_reaction
                })
        
        return sorted(timeline, key=lambda x: x["escalation_level"], reverse=True)


def main():
    """Main function to run detailed Second Sino-Japanese War simulation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Detailed Second Sino-Japanese War simulation using CrewAI")
    parser.add_argument("--model", 
                        choices=["gpt-4", "claude-2", "groq/llama3-70b-8192", "groq/llama3-8b-8192", "groq/mixtral-8x7b-32768"],
                        default="groq/llama3-70b-8192",
                        help="LLM model to use for agents")
    parser.add_argument("--max_turns", type=int, default=20,
                        help="Maximum number of simulation turns")
    parser.add_argument("--trigger", type=str,
                        help="Custom trigger event (overrides default Marco Polo Bridge Incident)")
    parser.add_argument("--output", type=str,
                        help="Output file for detailed results")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--export_analysis", action="store_true",
                        help="Export detailed network analysis")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Set up API keys
    required_env_vars = []
    if args.model == "gpt-4":
        required_env_vars.append("OPENAI_API_KEY")
    elif args.model == "claude-2":
        required_env_vars.append("ANTHROPIC_API_KEY")
    elif args.model.startswith("groq/"):
        required_env_vars.append("GROQ_API_KEY")
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
        return
    
    # Initialize specialized simulation
    simulation = SinoJapaneseWarSimulation(llm_model=args.model)
    
    # Set custom trigger if provided
    if args.trigger:
        simulation.game_state.trigger_event = args.trigger
    
    print(f"\n=== SECOND SINO-JAPANESE WAR DETAILED SIMULATION ===")
    print(f"Model: {args.model}")
    print(f"Trigger Event: {simulation.game_state.trigger_event}")
    print(f"Total Factions: {len(simulation.game_state.countries)}")
    print(f"Max Turns: {args.max_turns}")
    print("=" * 60)
    
    # Run detailed simulation
    try:
        results = simulation.run_detailed_simulation(max_turns=args.max_turns)
        
        # Export results
        if args.export_analysis:
            simulation.export_detailed_analysis(results, args.output)
        else:
            simulation.save_results(results, args.output)
        
        # Print summary
        print(f"\n=== SIMULATION COMPLETE ===")
        print(f"Turns completed: {results['final_state']['total_turns']}")
        print(f"Final conflict intensity: {results['final_state']['final_conflict_intensity']}")
        print(f"Total events: {results['final_state']['total_events']}")
        print(f"Final outcome: {results['final_state']['outcome']}")
        
        # Print top faction influence scores
        if "network_analysis" in results:
            print(f"\nTop Faction Influence Scores:")
            influence_scores = results["network_analysis"]["faction_influence_scores"]
            sorted_factions = sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            for faction, score in sorted_factions:
                print(f"  {faction}: {score:.1f}")
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"Simulation failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
