#!/usr/bin/env python3
"""
WarAgent Implementation using CrewAI Framework
==============================================

This module implements the WarAgent multi-agent simulation system using CrewAI,
allowing for simulation of historical conflicts with country agents, secretary agents,
and international relationship management.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from crewai import Agent, Task, Crew, Process
from crewai.tools.base_tool import Tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq


# Configuration and Data Structures
class ActionType(Enum):
    DIPLOMACY = "diplomacy"
    MILITARY = "military"
    ECONOMIC = "economic"
    INTERNAL = "internal"
    ALLIANCE = "alliance"


@dataclass
class CountryProfile:
    """Country profile containing all relevant information for an agent."""
    name: str
    resources: Dict[str, int]
    military_strength: int
    economic_power: int
    population: int
    alliances: List[str]
    enemies: List[str]
    government_type: str
    leader: str
    historical_context: str
    current_status: str
    domestic_policies: List[str]


@dataclass
class GameState:
    """Current state of the war simulation."""
    current_turn: int
    countries: Dict[str, CountryProfile]
    international_relations: Dict[str, Dict[str, float]]  # country -> country -> relationship score
    recent_events: List[str]
    active_conflicts: List[str]
    global_stability: float
    trigger_event: str


class WarAgentSimulation:
    """Main WarAgent simulation class using CrewAI framework."""
    
    def __init__(self, llm_model: str = "gpt-4", scenario: str = "WWI"):
        self.scenario = scenario
        self.game_state = GameState(
            current_turn=0,
            countries={},
            international_relations={},
            recent_events=[],
            active_conflicts=[],
            global_stability=0.5,
            trigger_event=""
        )
        
        # Initialize LLM
        if llm_model == "gpt-4":
            self.llm = ChatOpenAI(model="gpt-4", temperature=0.7)
        elif llm_model == "claude-2":
            self.llm = ChatAnthropic(model="claude-2", temperature=0.7)
        elif llm_model.startswith("groq/"):
            model_name = llm_model.replace("groq/", "")
            self.llm = ChatGroq(model=model_name, temperature=0.7)
        else:
            raise ValueError(f"Unsupported model: {llm_model}")
        
        self.country_agents: Dict[str, Agent] = {}
        self.secretary_agents: Dict[str, Agent] = {}
        self.board_agent: Optional[Agent] = None
        
        # Initialize scenario data
        self._load_scenario_data()
        self._create_agents()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _load_scenario_data(self):
        """Load historical data for the specified scenario."""
        if self.scenario == "WWI":
            self._load_wwi_data()
        elif self.scenario == "WWII":
            self._load_wwii_data()
        elif self.scenario == "Second Sino-Japanese War":
            self._load_second_sino_japanese_war_data()
        elif self.scenario == "Warring States":
            self._load_warring_states_data()
        else:
            raise ValueError(f"Unsupported scenario: {self.scenario}")

    def _load_wwi_data(self):
        """Load WWI scenario data."""
        countries_data = {
            "Germany": CountryProfile(
                name="Germany",
                resources={"steel": 100, "coal": 120, "manpower": 80},
                military_strength=95,
                economic_power=90,
                population=67000000,
                alliances=["Austria-Hungary"],
                enemies=["France", "Russia", "Britain"],
                government_type="Constitutional Monarchy",
                leader="Kaiser Wilhelm II",
                historical_context="Rising industrial power with growing naval ambitions",
                current_status="Concerned about encirclement by hostile powers",
                domestic_policies=["Naval expansion", "Industrial growth", "Military modernization"]
            ),
            "Britain": CountryProfile(
                name="Britain",
                resources={"steel": 80, "coal": 100, "manpower": 60},
                military_strength=85,
                economic_power=100,
                population=45000000,
                alliances=["France", "Russia"],
                enemies=["Germany"],
                government_type="Constitutional Monarchy",
                leader="King George V",
                historical_context="Global maritime empire with vast colonial resources",
                current_status="Maintaining balance of power in Europe",
                domestic_policies=["Naval supremacy", "Colonial administration", "Free trade"]
            ),
            "France": CountryProfile(
                name="France",
                resources={"steel": 70, "coal": 60, "manpower": 75},
                military_strength=80,
                economic_power=75,
                population=39000000,
                alliances=["Britain", "Russia"],
                enemies=["Germany"],
                government_type="Republic",
                leader="Raymond Poincaré",
                historical_context="Still recovering from Franco-Prussian War defeat",
                current_status="Seeking revenge against Germany and Alsace-Lorraine recovery",
                domestic_policies=["Military buildup", "Colonial expansion", "Alliance strengthening"]
            ),
            "Russia": CountryProfile(
                name="Russia",
                resources={"steel": 60, "coal": 70, "manpower": 120},
                military_strength=75,
                economic_power=60,
                population=175000000,
                alliances=["France", "Britain"],
                enemies=["Germany", "Austria-Hungary"],
                government_type="Absolute Monarchy",
                leader="Tsar Nicholas II",
                historical_context="Vast empire struggling with modernization",
                current_status="Balkan interests conflicting with Austria-Hungary",
                domestic_policies=["Military modernization", "Industrial development", "Pan-Slavism"]
            ),
            "Austria-Hungary": CountryProfile(
                name="Austria-Hungary",
                resources={"steel": 50, "coal": 55, "manpower": 70},
                military_strength=65,
                economic_power=65,
                population=52000000,
                alliances=["Germany"],
                enemies=["Russia", "Serbia"],
                government_type="Dual Monarchy",
                leader="Emperor Franz Joseph I",
                historical_context="Multi-ethnic empire facing internal tensions",
                current_status="Struggling with nationalist movements in the Balkans",
                domestic_policies=["Balkan control", "Internal stability", "German alliance"]
            )
        }
        
        self.game_state.countries = countries_data
        self.game_state.trigger_event = "Assassination of Archduke Franz Ferdinand in Sarajevo"
        
        # Initialize relationship matrix
        for country1 in countries_data:
            self.game_state.international_relations[country1] = {}
            for country2 in countries_data:
                if country1 != country2:
                    if country2 in countries_data[country1].alliances:
                        self.game_state.international_relations[country1][country2] = 0.8
                    elif country2 in countries_data[country1].enemies:
                        self.game_state.international_relations[country1][country2] = -0.6
                    else:
                        self.game_state.international_relations[country1][country2] = 0.1

    def _load_second_sino_japanese_war_data(self):
        """Load Second Sino-Japanese War scenario data with high granularity players."""
        countries_data = {
            # Japanese Imperial Core
            "Emperor": CountryProfile(
                name="Emperor Hirohito",
                resources={"divine_authority": 100, "imperial_treasury": 90, "ceremonial_power": 100},
                military_strength=100,
                economic_power=80,
                population=1,
                alliances=["Imperial_Advisors", "Tosei_Ha"],
                enemies=[],
                government_type="Divine Monarchy",
                leader="Emperor Hirohito",
                historical_context="Divine emperor with constitutional constraints but ultimate moral authority",
                current_status="Balancing military factions while maintaining divine status",
                domestic_policies=["Divine legitimacy", "Military expansion", "Imperial ceremonies"]
            ),
            
            "Imperial_Advisors": CountryProfile(
                name="Imperial Advisors & Naidaijin",
                resources={"political_influence": 85, "court_connections": 95, "policy_guidance": 80},
                military_strength=30,
                economic_power=70,
                population=50,
                alliances=["Emperor", "Tosei_Ha"],
                enemies=["Kodo_Ha_Radicals"],
                government_type="Court Advisory System",
                leader="Prince Saionji Kinmochi",
                historical_context="Traditional court advisors trying to moderate military extremism",
                current_status="Attempting to maintain imperial dignity while managing military pressure",
                domestic_policies=["Court protocol", "Political moderation", "Imperial consultation"]
            ),

            # Japanese Military Factions
            "Tosei_Ha": CountryProfile(
                name="Tosei-ha (Control Faction)",
                resources={"modern_weapons": 95, "bureaucratic_control": 90, "industrial_support": 85},
                military_strength=95,
                economic_power=85,
                population=2000,
                alliances=["Emperor", "Imperial_Advisors", "Army_General_Staff", "Zaibatsu"],
                enemies=["Kodo_Ha", "Chinese_Nationalists"],
                government_type="Military Bureaucracy",
                leader="Tojo Hideki",
                historical_context="Modern, technocratic military faction advocating total war state",
                current_status="Gradually gaining control after 226 Incident, promoting systematic expansion",
                domestic_policies=["Total war preparation", "Industrial mobilization", "Bureaucratic efficiency"]
            ),

            "Kodo_Ha": CountryProfile(
                name="Kōdō-ha (Imperial Way Faction)", 
                resources={"spiritual_fervor": 90, "young_officer_support": 85, "assassination_capability": 70},
                military_strength=70,
                economic_power=40,
                population=1500,
                alliances=["Young_Officers"],
                enemies=["Tosei_Ha", "Political_Parties", "Zaibatsu"],
                government_type="Spiritual Militarism",
                leader="Araki Sadao",
                historical_context="Radical spiritual faction promoting emperor worship and direct action",
                current_status="Weakened after 226 Incident but still influential among young officers",
                domestic_policies=["Showa Restoration", "Anti-capitalism", "Spiritual mobilization"]
            ),

            "Kwantung_Army": CountryProfile(
                name="Kwantung Army",
                resources={"field_autonomy": 95, "manchurian_resources": 80, "combat_experience": 90},
                military_strength=90,
                economic_power=60,
                population=300000,
                alliances=["Army_General_Staff"],
                enemies=["Chinese_Warlords", "Soviet_Union"],
                government_type="Military Field Command",
                leader="Ueda Kenkichi",
                historical_context="Autonomous army in Manchuria, pioneer of aggressive expansion",
                current_status="Controlling Manchukuo and pushing for further expansion into China",
                domestic_policies=["Gekokujo tactics", "Fait accompli strategy", "Resource extraction"]
            ),

            "China_Expeditionary_Forces": CountryProfile(
                name="China Expeditionary Forces",
                resources={"expeditionary_supplies": 80, "regional_control": 70, "occupation_administration": 60},
                military_strength=85,
                economic_power=50,
                population=250000,
                alliances=["Army_General_Staff", "Kwantung_Army"],
                enemies=["Chinese_Nationalists", "Chinese_Communists", "Chinese_Warlords"],
                government_type="Military Occupation Force",
                leader="Matsui Iwane",
                historical_context="Forces deployed for full-scale war against China after Marco Polo Bridge",
                current_status="Engaged in brutal campaign to subjugate Chinese resistance",
                domestic_policies=["Three-all policy", "Puppet government support", "Resource exploitation"]
            ),

            "Korean_Garrison": CountryProfile(
                name="Korean Garrison Army",
                resources={"colonial_control": 85, "korean_resources": 60, "logistics_support": 70},
                military_strength=60,
                economic_power=45,
                population=150000,
                alliances=["Army_General_Staff"],
                enemies=["Korean_Resistance"],
                government_type="Colonial Military Administration",
                leader="Minami Jiro",
                historical_context="Colonial garrison maintaining control over Korea",
                current_status="Suppressing Korean independence movements while supporting China operations",
                domestic_policies=["Cultural assimilation", "Resource extraction", "Resistance suppression"]
            ),

            # Japanese Navy Factions
            "Fleet_Faction": CountryProfile(
                name="Naval Fleet Faction",
                resources={"battleships": 90, "naval_aviation": 70, "naval_budget": 80},
                military_strength=85,
                economic_power=75,
                population=5000,
                alliances=["Naval_General_Staff"],
                enemies=["Naval_Treaty_Faction"],
                government_type="Naval Command",
                leader="Yamamoto Isoroku",
                historical_context="Advocating decisive battle doctrine and large battleship construction",
                current_status="Preparing for potential Pacific War while supporting China operations",
                domestic_policies=["Kantai Kessen doctrine", "Naval aviation development", "Pacific expansion"]
            ),

            "Naval_Treaty_Faction": CountryProfile(
                name="Naval Treaty Faction",
                resources={"diplomatic_connections": 70, "international_legitimacy": 60, "cost_efficiency": 80},
                military_strength=60,
                economic_power=85,
                population=2000,
                alliances=["Political_Parties"],
                enemies=["Fleet_Faction", "Tosei_Ha"],
                government_type="Naval Diplomacy",
                leader="Yonai Mitsumasa",
                historical_context="Naval officers preferring diplomatic solutions and treaty limitations",
                current_status="Increasingly marginalized as war fever grows",
                domestic_policies=["Naval treaties", "International cooperation", "Budget efficiency"]
            ),

            # Japanese Political and Business
            "Imperial_Diet": CountryProfile(
                name="Imperial Diet",
                resources={"legislative_power": 60, "budget_approval": 70, "public_legitimacy": 50},
                military_strength=10,
                economic_power=60,
                population=800,
                alliances=["Political_Parties"],
                enemies=["Tosei_Ha", "Kodo_Ha"],
                government_type="Constitutional Monarchy Parliament",
                leader="Konoe Fumimaro",
                historical_context="Parliament increasingly dominated by military and nationalist parties",
                current_status="Rubber-stamping military policies while losing real power",
                domestic_policies=["National unity", "War support", "Domestic mobilization"]
            ),

            "Cabinet_Ministers": CountryProfile(
                name="Cabinet Ministers",
                resources={"administrative_control": 70, "policy_implementation": 80, "bureaucratic_network": 75},
                military_strength=20,
                economic_power=75,
                population=50,
                alliances=["Imperial_Diet", "Tosei_Ha"],
                enemies=["Kodo_Ha_Radicals"],
                government_type="Executive Cabinet",
                leader="Prince Konoe Fumimaro",
                historical_context="Civilian ministers increasingly subordinated to military demands",
                current_status="Implementing total war mobilization under military pressure",
                domestic_policies=["National mobilization", "Resource allocation", "War administration"]
            ),

            "Zaibatsu": CountryProfile(
                name="Zaibatsu Financial Groups",
                resources={"industrial_capacity": 95, "financial_capital": 90, "international_trade": 80},
                military_strength=30,
                economic_power=95,
                population=10000,
                alliances=["Tosei_Ha", "Cabinet_Ministers"],
                enemies=["Kodo_Ha", "Socialist_Groups"],
                government_type="Corporate Conglomerates",
                leader="Mitsui, Mitsubishi, Sumitomo leaders",
                historical_context="Industrial conglomerates profiting from military expansion",
                current_status="Adapting to war economy while maximizing profits",
                domestic_policies=["Industrial expansion", "Military contracts", "Resource monopolization"]
            ),

            # Chinese Nationalist Government
            "Chiang_Kai_Shek": CountryProfile(
                name="Chiang Kai-shek Central Government",
                resources={"central_authority": 70, "modern_army": 60, "international_support": 50},
                military_strength=70,
                economic_power=45,
                population=400000000,
                alliances=["Chinese_Central_Army", "German_Advisors"],
                enemies=["Japanese_Forces", "Chinese_Communists", "Chinese_Warlords"],
                government_type="Nationalist Authoritarian",
                leader="Chiang Kai-shek",
                historical_context="Nationalist leader trying to unify China while fighting Japanese invasion",
                current_status="Balancing resistance against Japan with suppression of Communists",
                domestic_policies=["Chinese unification", "Military modernization", "Anti-Communist campaigns"]
            ),

            "Chinese_Central_Army": CountryProfile(
                name="Chinese Central Army",
                resources={"german_training": 70, "central_equipment": 60, "loyal_officers": 80},
                military_strength=65,
                economic_power=40,
                population=300000,
                alliances=["Chiang_Kai_Shek", "German_Advisors"],
                enemies=["Japanese_Forces", "Chinese_Warlords"],
                government_type="National Revolutionary Army",
                leader="Chen Cheng",
                historical_context="Chiang's most reliable and modern military forces",
                current_status="Fighting desperate defense against superior Japanese forces",
                domestic_policies=["Modern warfare tactics", "Central command", "Elite training"]
            ),

            "Wang_Jingwei_Left_KMT": CountryProfile(
                name="Wang Jingwei Left KMT",
                resources={"political_legitimacy": 40, "japanese_support": 60, "administrative_experience": 70},
                military_strength=30,
                economic_power=35,
                population=50000,
                alliances=["Japanese_Collaborators"],
                enemies=["Chiang_Kai_Shek", "Chinese_Communists"],
                government_type="Puppet Nationalist Government",
                leader="Wang Jingwei",
                historical_context="Former KMT leftist now collaborating with Japanese",
                current_status="Establishing puppet government with Japanese support",
                domestic_policies=["Sino-Japanese cooperation", "Anti-Communist stance", "Peace negotiations"]
            ),

            # Chinese Regional Warlords  
            "Northern_Warlords": CountryProfile(
                name="Northern Warlords (Zhang Faction)",
                resources={"regional_control": 60, "local_resources": 50, "autonomy": 70},
                military_strength=50,
                economic_power=40,
                population=50000000,
                alliances=["Japanese_Collaborators"],
                enemies=["Chiang_Kai_Shek", "Chinese_Communists"],
                government_type="Regional Military Authority",
                leader="Zhang Xueliang",
                historical_context="Northeastern warlords with complicated relationship to Japanese",
                current_status="Some collaborating, others resisting Japanese occupation",
                domestic_policies=["Regional autonomy", "Survival pragmatism", "Local control"]
            ),

            "Central_Warlords": CountryProfile(
                name="Central Warlords (Feng Faction)",
                resources={"military_modernization": 55, "educational_reform": 60, "regional_development": 50},
                military_strength=55,
                economic_power=45,
                population=30000000,
                alliances=["Chiang_Kai_Shek"],
                enemies=["Japanese_Forces", "Chinese_Communists"],
                government_type="Regional Modernizing Authority",
                leader="Feng Yuxiang",
                historical_context="Christian warlord promoting modernization and education",
                current_status="Nominally supporting Chiang while maintaining regional autonomy",
                domestic_policies=["Regional modernization", "Educational development", "Military reform"]
            ),

            "Southern_Warlords": CountryProfile(
                name="Southern Warlords (Guangxi Clique)",
                resources={"coastal_trade": 65, "naval_connections": 40, "regional_resources": 55},
                military_strength=45,
                economic_power=50,
                population=25000000,
                alliances=["Chiang_Kai_Shek"],
                enemies=["Japanese_Forces"],
                government_type="Regional Maritime Authority", 
                leader="Li Zongren",
                historical_context="Southern warlords controlling coastal trade routes",
                current_status="Supporting national resistance while protecting regional interests",
                domestic_policies=["Coastal defense", "Trade protection", "Regional stability"]
            ),

            # Chinese Communists
            "Chinese_Communists": CountryProfile(
                name="Chinese Communist Party",
                resources={"guerrilla_warfare": 80, "peasant_support": 70, "ideology": 85},
                military_strength=45,
                economic_power=25,
                population=100000,
                alliances=["Soviet_Advisors", "Peasant_Masses"],
                enemies=["Chiang_Kai_Shek", "Japanese_Forces", "Chinese_Warlords"],
                government_type="Communist Revolutionary Party",
                leader="Mao Zedong",
                historical_context="Communist revolutionaries building rural base areas",
                current_status="United front with KMT against Japanese while building strength",
                domestic_policies=["Land reform", "Guerrilla warfare", "Mass mobilization"]
            ),

            "Communist_Guerrillas": CountryProfile(
                name="Communist Guerrilla Forces",
                resources={"local_knowledge": 90, "mobility": 85, "popular_support": 75},
                military_strength=60,
                economic_power=20,
                population=50000,
                alliances=["Chinese_Communists", "Peasant_Masses"],
                enemies=["Japanese_Forces", "Chinese_Warlords"],
                government_type="Revolutionary Armed Forces",
                leader="Zhu De",
                historical_context="Communist military forces specializing in guerrilla tactics",
                current_status="Fighting both Japanese invaders and domestic enemies",
                domestic_policies=["People's war", "Rural mobilization", "Guerrilla tactics"]
            ),

            # Social Groups
            "Japanese_Urban_Upper_Class": CountryProfile(
                name="Japanese Urban Upper Class",
                resources={"wealth": 90, "social_influence": 85, "media_control": 70},
                military_strength=20,
                economic_power=85,
                population=2000000,
                alliances=["Zaibatsu", "Imperial_Diet"],
                enemies=["Socialist_Groups"],
                government_type="Social Elite",
                leader="Various business and social leaders",
                historical_context="Wealthy urban elites supporting imperial expansion",
                current_status="Profiting from war economy while supporting national objectives",
                domestic_policies=["Conservative nationalism", "Economic expansion", "Social hierarchy"]
            ),

            "Japanese_Urban_Middle_Class": CountryProfile(
                name="Japanese Urban Middle Class", 
                resources={"education": 70, "professional_skills": 75, "civic_participation": 60},
                military_strength=30,
                economic_power=60,
                population=8000000,
                alliances=["Imperial_Diet", "Labor_Organizations"],
                enemies=["Extreme_Militarists"],
                government_type="Civil Society",
                leader="Various professional and civic leaders",
                historical_context="Educated middle class torn between liberalism and nationalism",
                current_status="Supporting war effort while maintaining some democratic values",
                domestic_policies=["Moderate nationalism", "Professional advancement", "Civil society"]
            ),

            "Japanese_Urban_Working_Class": CountryProfile(
                name="Japanese Urban Working Class",
                resources={"industrial_labor": 80, "union_organization": 60, "collective_action": 70},
                military_strength=40,
                economic_power=45,
                population=12000000,
                alliances=["Labor_Organizations", "Socialist_Groups"],
                enemies=["Zaibatsu", "Extreme_Militarists"],
                government_type="Labor Movement",
                leader="Various labor leaders",
                historical_context="Industrial workers supporting some social reforms",
                current_status="Mobilized for war production while maintaining labor rights",
                domestic_policies=["Labor rights", "Social welfare", "Industrial mobilization"]
            ),

            "Chinese_Rural_Peasants": CountryProfile(
                name="Chinese Rural Peasants",
                resources={"agricultural_production": 70, "local_knowledge": 85, "population_mass": 90},
                military_strength=35,
                economic_power=30,
                population=350000000,
                alliances=["Chinese_Communists", "Local_Militias"],
                enemies=["Japanese_Occupiers", "Warlord_Exploitation"],
                government_type="Rural Communities",
                leader="Various village leaders and revolutionary committees",
                historical_context="Vast peasant population suffering under war and oppression",
                current_status="Supporting various resistance movements against Japanese occupation",
                domestic_policies=["Land rights", "Self-defense", "Survival strategies"]
            ),

            "Colonial_Subjects": CountryProfile(
                name="Korean and Taiwanese Colonial Subjects",
                resources={"labor_force": 70, "local_knowledge": 80, "resistance_networks": 60},
                military_strength=25,
                economic_power=35,
                population=30000000,
                alliances=["Independence_Movements"],
                enemies=["Japanese_Colonial_Administration"],
                government_type="Colonial Resistance",
                leader="Various independence leaders",
                historical_context="Colonial subjects exploited for Japanese war effort",
                current_status="Forced mobilization while maintaining independence aspirations",
                domestic_policies=["Cultural preservation", "Independence struggle", "Survival resistance"]
            )
        }
        
        self.game_state.countries = countries_data
        self.game_state.trigger_event = "Marco Polo Bridge Incident - July 7, 1937"
        
        # Initialize complex relationship matrix based on historical alliances and conflicts
        self._initialize_sino_japanese_relationships()

    def _initialize_sino_japanese_relationships(self):
        """Initialize the complex relationship matrix for Second Sino-Japanese War."""
        countries = self.game_state.countries
        relations = {}
        
        for country1 in countries:
            relations[country1] = {}
            for country2 in countries:
                if country1 != country2:
                    relations[country1][country2] = self._calculate_initial_relationship(
                        countries[country1], countries[country2]
                    )
        
        self.game_state.international_relations = relations

    def _calculate_initial_relationship(self, country1: CountryProfile, country2: CountryProfile) -> float:
        """Calculate initial relationship between two countries based on alliances and enemies."""
        if country2.name in country1.alliances:
            return 0.8
        elif country2.name in country1.enemies:
            return -0.7
        elif any(enemy in country1.enemies for enemy in country2.alliances):
            return -0.4
        elif any(ally in country1.alliances for ally in country2.alliances):
            return 0.4
        else:
            return 0.0

    def _load_warring_states_data(self):
        """Load Warring States scenario data (placeholder for extensibility)."""
        # This would be implemented for ancient China scenario
        pass

    def _create_agents(self):
        """Create all agents using CrewAI framework."""
        self._create_country_agents()
        self._create_secretary_agents()
        self._create_board_agent()

    def _create_country_agents(self):
        """Create country agents for each participating nation."""
        for country_name, profile in self.game_state.countries.items():
            agent = Agent(
                role=f"{country_name} National Leader",
                goal=f"Protect and advance the interests of {country_name} while responding to the evolving international crisis",
                backstory=f"""You are the leader of {country_name} in the year 1914. 
                Your country has the following characteristics:
                - Government: {profile.government_type}
                - Leader: {profile.leader}
                - Military Strength: {profile.military_strength}/100
                - Economic Power: {profile.economic_power}/100
                - Population: {profile.population:,}
                - Current Alliances: {', '.join(profile.alliances) if profile.alliances else 'None'}
                - Current Enemies: {', '.join(profile.enemies) if profile.enemies else 'None'}
                - Historical Context: {profile.historical_context}
                - Current Status: {profile.current_status}
                - Domestic Policies: {', '.join(profile.domestic_policies)}
                
                You must make decisions that are historically plausible and consistent with your country's 
                interests, capabilities, and constraints. Consider diplomatic, military, economic, and 
                internal actions carefully.""",
                verbose=True,
                allow_delegation=False,
                llm=self.llm,
                tools=[self._create_action_tool(country_name)]
            )
            self.country_agents[country_name] = agent

    def _create_secretary_agents(self):
        """Create secretary agents to verify country agent decisions."""
        for country_name in self.game_state.countries.keys():
            agent = Agent(
                role=f"{country_name} Policy Secretary",
                goal=f"Review and validate the logical consistency and appropriateness of {country_name}'s proposed actions",
                backstory=f"""You are the trusted policy advisor and secretary to the leader of {country_name}. 
                Your role is to:
                1. Review proposed actions for logical consistency
                2. Check if actions align with national interests and capabilities
                3. Identify potential risks or unintended consequences
                4. Suggest modifications if needed
                5. Ensure actions are within the country's resource and military capabilities
                
                You have deep knowledge of international law, diplomacy, military strategy, and your country's 
                specific situation. You are cautious and analytical, helping prevent rash decisions that could 
                harm your nation.""",
                verbose=True,
                allow_delegation=False,
                llm=self.llm,
                tools=[self._create_validation_tool(country_name)]
            )
            self.secretary_agents[country_name] = agent

    def _create_board_agent(self):
        """Create the board agent to manage international relations and global state."""
        self.board_agent = Agent(
            role="International Relations Moderator",
            goal="Manage global diplomatic relations, resolve conflicts, and maintain simulation realism",
            backstory="""You are an omniscient moderator overseeing the international system during this 
            historical crisis. Your responsibilities include:
            1. Processing and implementing actions from all countries
            2. Managing diplomatic relations and alliance changes
            3. Calculating consequences of military and economic actions
            4. Determining global stability and conflict escalation
            5. Ensuring historical plausibility and realism
            6. Managing resource transfers and territorial changes
            
            You have perfect knowledge of all countries' actions and can see the full diplomatic picture. 
            You are neutral but must enforce realistic consequences for all actions.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self._create_board_tools()]
        )

    def _create_action_tool(self, country_name: str) -> Tool:
        """Create action tool for country agents."""
        def execute_action(action_type: str, target_country: str = "", details: str = "") -> str:
            """Execute a country action in the simulation."""
            try:
                action_data = {
                    "country": country_name,
                    "action_type": action_type,
                    "target_country": target_country,
                    "details": details,
                    "turn": self.game_state.current_turn
                }
                
                # Log the action
                self.logger.info(f"{country_name} executes {action_type}: {details}")
                
                # Add to recent events
                event_description = f"{country_name} {action_type}"
                if target_country:
                    event_description += f" towards {target_country}"
                event_description += f": {details}"
                
                self.game_state.recent_events.append(event_description)
                
                return f"Action executed successfully: {event_description}"
                
            except Exception as e:
                return f"Action execution failed: {str(e)}"
        
        return Tool(
            name=f"{country_name}_action_tool",
            description=f"Execute actions for {country_name}. Available action types: diplomacy, military, economic, internal, alliance",
            func=execute_action
        )

    def _create_validation_tool(self, country_name: str) -> Tool:
        """Create validation tool for secretary agents."""
        def validate_action(proposed_action: str, reasoning: str = "") -> str:
            """Validate a proposed action for logical consistency and appropriateness."""
            try:
                country_profile = self.game_state.countries[country_name]
                
                validation_result = {
                    "action": proposed_action,
                    "country": country_name,
                    "reasoning": reasoning,
                    "validation_status": "approved",  # or "rejected" or "modified"
                    "secretary_notes": "",
                    "turn": self.game_state.current_turn
                }
                
                # Basic validation logic would go here
                # For now, we'll approve most reasonable actions
                validation_result["secretary_notes"] = f"Action reviewed and appears consistent with {country_name}'s capabilities and interests."
                
                return json.dumps(validation_result, indent=2)
                
            except Exception as e:
                return f"Validation failed: {str(e)}"
        
        return Tool(
            name=f"{country_name}_validation_tool",
            description=f"Validate proposed actions for {country_name}",
            func=validate_action
        )

    def _create_board_tools(self) -> List[Tool]:
        """Create tools for the board agent."""
        def update_relations(country1: str, country2: str, change: float, reason: str = "") -> str:
            """Update diplomatic relations between two countries."""
            try:
                if country1 in self.game_state.international_relations and country2 in self.game_state.international_relations[country1]:
                    old_value = self.game_state.international_relations[country1][country2]
                    new_value = max(-1.0, min(1.0, old_value + change))
                    self.game_state.international_relations[country1][country2] = new_value
                    
                    # Update reciprocal relationship
                    self.game_state.international_relations[country2][country1] = new_value
                    
                    return f"Updated relations between {country1} and {country2}: {old_value:.2f} -> {new_value:.2f} ({reason})"
                else:
                    return f"Invalid country pair: {country1}, {country2}"
            except Exception as e:
                return f"Relation update failed: {str(e)}"
        
        def calculate_global_stability() -> str:
            """Calculate and update global stability based on current tensions."""
            try:
                total_relations = 0
                count = 0
                
                for country1, relations in self.game_state.international_relations.items():
                    for country2, relation_value in relations.items():
                        total_relations += relation_value
                        count += 1
                
                if count > 0:
                    avg_relations = total_relations / count
                    # Convert to 0-1 scale where 0.5 is neutral
                    self.game_state.global_stability = (avg_relations + 1) / 2
                
                stability_description = "Very Unstable" if self.game_state.global_stability < 0.2 else \
                                      "Unstable" if self.game_state.global_stability < 0.4 else \
                                      "Tense" if self.game_state.global_stability < 0.6 else \
                                      "Stable" if self.game_state.global_stability < 0.8 else "Very Stable"
                
                return f"Global stability: {self.game_state.global_stability:.2f} ({stability_description})"
                
            except Exception as e:
                return f"Stability calculation failed: {str(e)}"
        
        return [
            Tool(
                name="update_relations",
                description="Update diplomatic relations between countries",
                func=update_relations
            ),
            Tool(
                name="calculate_stability",
                description="Calculate global stability based on current international relations",
                func=calculate_global_stability
            )
        ]

    def create_turn_tasks(self) -> List[Task]:
        """Create tasks for the current turn."""
        tasks = []
        
        # Create situation analysis task for each country
        for country_name in self.game_state.countries.keys():
            situation_task = Task(
                description=f"""Analyze the current international situation from {country_name}'s perspective and propose actions.

                Current Turn: {self.game_state.current_turn}
                Trigger Event: {self.game_state.trigger_event}
                Recent Events: {'; '.join(self.game_state.recent_events[-5:]) if self.game_state.recent_events else 'None'}
                Global Stability: {self.game_state.global_stability:.2f}
                Active Conflicts: {'; '.join(self.game_state.active_conflicts) if self.game_state.active_conflicts else 'None'}
                
                Your current relationships with other nations:
                {self._format_relationships(country_name)}
                
                Based on this situation, propose 1-3 specific actions that {country_name} should take this turn. 
                Consider diplomatic, military, economic, or internal policy actions that are appropriate for your 
                country's situation and capabilities.
                
                Format your response as specific, actionable decisions with clear reasoning.""",
                expected_output=f"Detailed analysis and 1-3 specific proposed actions for {country_name}",
                agent=self.country_agents[country_name]
            )
            tasks.append(situation_task)
        
        # Create validation tasks for each country
        for country_name in self.game_state.countries.keys():
            validation_task = Task(
                description=f"""Review the proposed actions for {country_name} and validate their appropriateness.

                Evaluate each proposed action for:
                1. Logical consistency with {country_name}'s capabilities and resources
                2. Alignment with national interests and historical context
                3. Potential risks and unintended consequences
                4. Feasibility given current diplomatic and military situation
                
                Provide detailed feedback and either approve, reject, or suggest modifications for each action.
                If you approve actions, use the validation tool to formally record your approval.
                If you suggest changes, explain the reasoning clearly.""",
                expected_output=f"Validation report for {country_name}'s proposed actions with recommendations",
                agent=self.secretary_agents[country_name]
            )
            tasks.append(validation_task)
        
        # Create board management task
        board_task = Task(
            description=f"""Manage the international situation and process all country actions for turn {self.game_state.current_turn}.

            Process all approved actions from countries and:
            1. Update diplomatic relations based on actions taken
            2. Resolve any conflicts or contradictory actions
            3. Calculate consequences of military and economic actions
            4. Update global stability
            5. Determine if any new conflicts have started
            6. Generate realistic outcomes and events
            
            Ensure all changes are historically plausible and maintain simulation realism.
            Use your tools to update relations and calculate stability as needed.""",
            expected_output="Complete turn resolution with updated international situation",
            agent=self.board_agent
        )
        tasks.append(board_task)
        
        return tasks

    def _format_relationships(self, country_name: str) -> str:
        """Format current relationships for a country."""
        if country_name not in self.game_state.international_relations:
            return "No relationship data available"
        
        relationships = []
        for other_country, relation_value in self.game_state.international_relations[country_name].items():
            relation_desc = "Allied" if relation_value > 0.6 else \
                           "Friendly" if relation_value > 0.2 else \
                           "Neutral" if relation_value > -0.2 else \
                           "Hostile" if relation_value > -0.6 else "At War"
            
            relationships.append(f"- {other_country}: {relation_desc} ({relation_value:.2f})")
        
        return "\n".join(relationships)

    def run_simulation(self, max_turns: int = 10) -> Dict[str, Any]:
        """Run the complete war simulation."""
        self.logger.info(f"Starting WarAgent simulation: {self.scenario}")
        self.logger.info(f"Trigger event: {self.game_state.trigger_event}")
        
        simulation_results = {
            "scenario": self.scenario,
            "trigger_event": self.game_state.trigger_event,
            "turns": [],
            "final_state": None
        }
        
        for turn in range(max_turns):
            self.game_state.current_turn = turn + 1
            self.logger.info(f"\n=== TURN {self.game_state.current_turn} ===")
            
            # Create tasks for this turn
            tasks = self.create_turn_tasks()
            
            # Create crew and execute turn
            crew = Crew(
                agents=list(self.country_agents.values()) + 
                       list(self.secretary_agents.values()) + 
                       [self.board_agent],
                tasks=tasks,
                process=Process.sequential,
                verbose=True
            )
            
            # Execute the turn
            turn_result = crew.kickoff()
            
            # Record turn results
            turn_data = {
                "turn_number": self.game_state.current_turn,
                "events": self.game_state.recent_events.copy(),
                "global_stability": self.game_state.global_stability,
                "active_conflicts": self.game_state.active_conflicts.copy(),
                "relationships": self.game_state.international_relations.copy(),
                "crew_output": str(turn_result)
            }
            simulation_results["turns"].append(turn_data)
            
            # Check for end conditions
            if self.game_state.global_stability < 0.1:
                self.logger.info("Simulation ended due to global instability (war outbreak)")
                break
            elif self.game_state.global_stability > 0.9:
                self.logger.info("Simulation ended due to successful peace maintenance")
                break
            
            # Clear recent events for next turn (keep last 3 for context)
            self.game_state.recent_events = self.game_state.recent_events[-3:]
        
        # Record final state
        simulation_results["final_state"] = {
            "global_stability": self.game_state.global_stability,
            "final_relationships": self.game_state.international_relations,
            "total_turns": self.game_state.current_turn,
            "outcome": self._determine_outcome()
        }
        
        return simulation_results

    def _determine_outcome(self) -> str:
        """Determine the final outcome of the simulation."""
        if self.game_state.global_stability < 0.2:
            return "Major war outbreak - diplomatic failure"
        elif self.game_state.global_stability < 0.4:
            return "Regional conflicts - partial diplomatic failure"
        elif self.game_state.global_stability < 0.6:
            return "Tense peace - fragile diplomatic situation"
        elif self.game_state.global_stability < 0.8:
            return "Stable peace - successful crisis management"
        else:
            return "Strong peace - diplomatic triumph"

    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None):
        """Save simulation results to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"waragent_simulation_{self.scenario}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Simulation results saved to {filename}")


def main():
    """Main function to run WarAgent simulation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="WarAgent simulation using CrewAI")
    parser.add_argument("--model", 
                        choices=["gpt-4", "claude-2", "groq/llama3-70b-8192", "groq/llama3-8b-8192", "groq/mixtral-8x7b-32768"],
                        default="gpt-4",
                        help="LLM model to use for agents")
    parser.add_argument("--scenario", 
                        choices=["WWI", "WWII", "Second Sino-Japanese War", "Warring States"], 
                        default="Second Sino-Japanese War",
                        help="Historical scenario to simulate")
    parser.add_argument("--max_turns", type=int, default=10,
                        help="Maximum number of simulation turns")
    parser.add_argument("--trigger", type=str,
                        help="Custom trigger event (overrides default)")
    parser.add_argument("--output", type=str,
                        help="Output file for results")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize simulation
    simulation = WarAgentSimulation(llm_model=args.model, scenario=args.scenario)
    
    # Set custom trigger if provided
    if args.trigger:
        simulation.game_state.trigger_event = args.trigger
    
    # Run simulation
    results = simulation.run_simulation(max_turns=args.max_turns)
    
    # Save results
    simulation.save_results(results, args.output)
    
    # Print summary
    print(f"\n=== SIMULATION COMPLETE ===")
    print(f"Scenario: {results['scenario']}")
    print(f"Turns completed: {results['final_state']['total_turns']}")
    print(f"Final outcome: {results['final_state']['outcome']}")
    print(f"Global stability: {results['final_state']['global_stability']:.2f}")


if __name__ == "__main__":
    main()
