#!/usr/bin/env python3
"""
Dynamic Peace Simulator - Integration of BDM, Coalition, and CrewAI
====================================================================

This module properly integrates all our systems to create dynamic tension
based on actual faction interactions, not fixed escalation rates.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import random

from multi_agent_coalition_system import MultiAgentCoalitionSystem, Topic, Coalition
from bdm_peace_calculator import PeaceWarCalculator, EMPEROR_ACTIONS, CHIANG_ACTIONS
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq


@dataclass
class SimulationState:
    """Current state of the simulation"""
    year: int
    month: int
    turn: int
    japan_intention: float
    china_readiness: float
    last_player_action: Optional[str] = None
    last_ai_actions: Dict[str, str] = None
    tensions: Dict[str, float] = None  # Track multiple tension sources


class DynamicPeaceSimulator:
    """
    Dynamic simulation using actual agent decisions and game theory
    Instead of fixed rates, tensions rise and fall based on:
    1. Player decisions
    2. AI agent autonomous actions
    3. Coalition dynamics
    4. BDM game theory outcomes
    5. Random events based on historical context
    """

    def __init__(self, player_role: str = "emperor", llm_model: str = "groq/llama3-70b-8192"):
        self.player_role = player_role
        self.coalition_system = MultiAgentCoalitionSystem()
        self.peace_calculator = PeaceWarCalculator()
        self.llm = ChatGroq(model_name=llm_model.split("/")[1] if "/" in llm_model else llm_model)

        # Initialize coalitions
        self._setup_initial_coalitions()

        # Track historical events that can trigger
        self.potential_events = self._initialize_historical_events()

        # Initialize simulation state
        self.state = SimulationState(
            year=1930, month=1, turn=0,
            japan_intention=35, china_readiness=25,
            tensions={
                "military_faction": 35,
                "economic_pressure": 45,
                "nationalist_sentiment": 30,
                "international_pressure": 20
            }
        )

    def _setup_initial_coalitions(self):
        """Setup initial coalition alignments"""
        # 1930 starting coalitions
        self.coalition_system.coalitions["Moderate_Alliance"] = Coalition("Moderate_Alliance", "Emperor")
        self.coalition_system.coalitions["Military_Bloc"] = Coalition("Military_Bloc", "Tosei_Ha")
        self.coalition_system.coalitions["KMT_Government"] = Coalition("KMT_Government", "Chiang_KMT")

    def _initialize_historical_events(self) -> List[Dict]:
        """Historical events that can trigger based on conditions"""
        return [
            {
                "name": "Economic Crisis Deepens",
                "year": 1930,
                "condition": lambda s: s.tensions["economic_pressure"] > 60,
                "effect": {"military_faction": +10, "economic_pressure": +5},
                "message": "Economic crisis pushes military to seek resources abroad"
            },
            {
                "name": "Manchurian Incident Opportunity",
                "year": 1931,
                "condition": lambda s: s.tensions["military_faction"] > 50 and s.month >= 9,
                "effect": {"military_faction": +20, "nationalist_sentiment": +15},
                "message": "Kwantung Army sees opportunity in Manchuria"
            },
            {
                "name": "International Condemnation",
                "year": 1932,
                "condition": lambda s: s.japan_intention > 60,
                "effect": {"international_pressure": +20, "military_faction": -5},
                "message": "League of Nations condemns Japanese actions"
            },
            {
                "name": "Communist-KMT Tension",
                "year": 1930,
                "condition": lambda s: s.china_readiness < 30,
                "effect": {"china_readiness": -10},
                "message": "Internal strife weakens Chinese unity"
            },
            {
                "name": "Peace Opportunity",
                "year": 1930,
                "condition": lambda s: s.tensions["military_faction"] < 30 and s.tensions["economic_pressure"] < 40,
                "effect": {"military_faction": -10, "international_pressure": -10},
                "message": "Diplomatic channels open for negotiation"
            }
        ]

    def simulate_turn(self, player_action: Optional[str] = None) -> Dict:
        """
        Simulate one turn using actual agent decisions and game theory
        """
        self.state.turn += 1
        self.state.month += 1
        if self.state.month > 12:
            self.state.month = 1
            self.state.year += 1

        results = {
            "turn": self.state.turn,
            "date": f"{self.state.year}-{self.state.month:02d}",
            "events": [],
            "ai_decisions": {},
            "tension_changes": {},
            "coalition_changes": []
        }

        # Step 1: Process player action if any
        if player_action:
            self._process_player_action(player_action, results)

        # Step 2: AI Agents make autonomous decisions
        ai_actions = self._get_ai_agent_decisions()
        results["ai_decisions"] = ai_actions

        # Step 3: Update coalition dynamics
        self.coalition_system.simulate_turn(player_actions={
            self.player_role: {"action": player_action} if player_action else {}
        })

        # Step 4: Calculate new positions using BDM model
        japan_pos, japan_details = self.peace_calculator.calculate_country_intention("japan")
        china_pos, china_details = self.peace_calculator.calculate_country_intention("china")

        # Step 5: Apply tension changes based on all actions
        tension_delta = self._calculate_tension_changes(player_action, ai_actions, japan_details, china_details)

        # Step 6: Check for historical events
        triggered_events = self._check_historical_events()
        results["events"].extend(triggered_events)

        # Step 7: Update state with natural ebb and flow
        self._update_state_dynamically(tension_delta, japan_pos, china_pos)

        # Step 8: Check for critical thresholds
        results["japan_intention"] = self.state.japan_intention
        results["china_readiness"] = self.state.china_readiness
        results["tensions"] = self.state.tensions.copy()

        return results

    def _process_player_action(self, action: str, results: Dict):
        """Process player's action through the game systems"""
        if self.player_role == "emperor":
            if action in EMPEROR_ACTIONS:
                effects = EMPEROR_ACTIONS[action]["effects"]
                self.peace_calculator.apply_player_action("japan", action, effects)
                results["events"].append(f"Emperor: {EMPEROR_ACTIONS[action]['description']}")
        else:
            if action in CHIANG_ACTIONS:
                effects = CHIANG_ACTIONS[action]["effects"]
                self.peace_calculator.apply_player_action("china", action, effects)
                results["events"].append(f"Chiang: {CHIANG_ACTIONS[action]['description']}")

    def _get_ai_agent_decisions(self) -> Dict[str, str]:
        """
        Use CrewAI agents to make autonomous faction decisions
        This replaces the fixed escalation rate with actual AI reasoning
        """
        decisions = {}

        # Create agents for key factions
        if self.state.tensions["military_faction"] > 40:
            military_agent = Agent(
                role="Japanese Military Faction",
                goal="Expand Japanese influence while managing risks",
                backstory=f"""You represent the military factions in {self.state.year}.
                Current military tension: {self.state.tensions['military_faction']}%.
                Economic pressure: {self.state.tensions['economic_pressure']}%.
                You must decide whether to push for action or wait.""",
                llm=self.llm,
                verbose=False
            )

            military_task = Task(
                description=f"""Given the current situation in {self.state.year}-{self.state.month:02d}:
                - Japan war intention: {self.state.japan_intention}%
                - Economic pressure: {self.state.tensions['economic_pressure']}%
                - International pressure: {self.state.tensions['international_pressure']}%

                Decide your faction's stance: 'escalate', 'maintain', or 'restrain'
                Consider historical context and your faction's goals.
                Output only one word: escalate, maintain, or restrain.""",
                expected_output="Single word decision",
                agent=military_agent
            )

            crew = Crew(agents=[military_agent], tasks=[military_task], process=Process.sequential)

            try:
                result = crew.kickoff()
                decision = str(result).strip().lower()
                if decision in ['escalate', 'maintain', 'restrain']:
                    decisions['military'] = decision
                else:
                    decisions['military'] = 'maintain'
            except:
                decisions['military'] = 'maintain'

        # Chinese faction decision
        if self.state.china_readiness > 20:
            decisions['chinese_response'] = self._get_chinese_response()

        return decisions

    def _get_chinese_response(self) -> str:
        """Determine Chinese faction response"""
        # Simplified - in full implementation would use CrewAI
        if self.state.japan_intention > 70:
            return 'mobilize'
        elif self.state.japan_intention > 50:
            return 'prepare'
        else:
            return 'focus_internal'

    def _calculate_tension_changes(self, player_action: str, ai_actions: Dict,
                                  japan_details: Dict, china_details: Dict) -> Dict[str, float]:
        """
        Calculate tension changes based on all actions and game theory outcomes
        This creates the dynamic ups and downs, not a fixed rate
        """
        changes = {key: 0.0 for key in self.state.tensions}

        # AI faction decisions affect tensions
        if ai_actions.get('military') == 'escalate':
            changes['military_faction'] += random.uniform(5, 15)
            changes['nationalist_sentiment'] += random.uniform(2, 8)
        elif ai_actions.get('military') == 'restrain':
            changes['military_faction'] -= random.uniform(3, 10)
            changes['international_pressure'] -= random.uniform(1, 5)

        # Economic factors can improve or worsen
        if self.state.year <= 1932:
            # Depression era - generally worsening
            changes['economic_pressure'] += random.uniform(-2, 5)
        else:
            # Recovery possible
            changes['economic_pressure'] += random.uniform(-5, 3)

        # International pressure responds to actions
        if self.state.japan_intention > 60:
            changes['international_pressure'] += random.uniform(2, 8)
        elif self.state.japan_intention < 40:
            changes['international_pressure'] -= random.uniform(1, 5)

        # Random events for realism (peace breakthroughs or crises)
        if random.random() < 0.1:  # 10% chance of significant event
            if random.random() < 0.5:
                # Peace opportunity
                changes['military_faction'] -= random.uniform(5, 15)
                changes['nationalist_sentiment'] -= random.uniform(3, 10)
            else:
                # Crisis
                changes['military_faction'] += random.uniform(5, 15)
                changes['nationalist_sentiment'] += random.uniform(3, 10)

        return changes

    def _check_historical_events(self) -> List[str]:
        """Check if any historical events should trigger"""
        triggered = []

        for event in self.potential_events:
            if event["year"] <= self.state.year:
                if event["condition"](self.state):
                    # Apply event effects
                    for key, change in event["effect"].items():
                        if key in self.state.tensions:
                            self.state.tensions[key] = max(0, min(100,
                                self.state.tensions[key] + change))
                        elif key == "china_readiness":
                            self.state.china_readiness = max(0, min(100,
                                self.state.china_readiness + change))

                    triggered.append(event["message"])

                    # Remove triggered one-time events
                    if random.random() < 0.7:  # Some events can repeat
                        self.potential_events.remove(event)

        return triggered

    def _update_state_dynamically(self, tension_delta: Dict[str, float],
                                 bdm_japan: float, bdm_china: float):
        """
        Update state with natural ebb and flow based on multiple factors
        NOT a fixed escalation rate!
        """
        # Apply tension changes
        for key, change in tension_delta.items():
            self.state.tensions[key] = max(0, min(100,
                self.state.tensions[key] + change))

        # Calculate new war intentions based on multiple factors
        # Weight different tension sources
        japan_intention_factors = {
            'military': self.state.tensions['military_faction'] * 0.4,
            'economic': self.state.tensions['economic_pressure'] * 0.2,
            'nationalist': self.state.tensions['nationalist_sentiment'] * 0.2,
            'bdm_calculation': bdm_japan * 0.2
        }

        # Natural variance (can go up OR down)
        variance = random.uniform(-5, 5)

        # Calculate new intention with possibility of decrease
        new_japan = sum(japan_intention_factors.values()) + variance

        # Smoothing to prevent wild swings but allow change
        self.state.japan_intention = 0.7 * self.state.japan_intention + 0.3 * new_japan
        self.state.japan_intention = max(0, min(100, self.state.japan_intention))

        # China responds to Japan but has own dynamics
        china_factors = {
            'japan_threat': self.state.japan_intention * 0.3,
            'internal_unity': (100 - self.state.tensions.get('internal_strife', 30)) * 0.2,
            'bdm_calculation': bdm_china * 0.3,
            'base_readiness': 20
        }

        new_china = sum(china_factors.values()) + random.uniform(-3, 3)
        self.state.china_readiness = 0.8 * self.state.china_readiness + 0.2 * new_china
        self.state.china_readiness = max(0, min(100, self.state.china_readiness))


def run_example():
    """Example of how the dynamic system works"""
    simulator = DynamicPeaceSimulator(player_role="emperor")

    print("Dynamic Peace Simulation - No Fixed Escalation!")
    print("=" * 60)

    for turn in range(20):
        # Simulate player choosing different actions
        if turn % 3 == 0:
            action = "diplomatic"
        elif turn % 3 == 1:
            action = "restraint"
        else:
            action = "do_nothing"

        result = simulator.simulate_turn(action)

        print(f"\nTurn {result['turn']}: {result['date']}")
        print(f"Japan: {result['japan_intention']:.1f}% | China: {result['china_readiness']:.1f}%")

        if result['ai_decisions']:
            print(f"AI Decisions: {result['ai_decisions']}")

        if result['events']:
            for event in result['events']:
                print(f"  📰 {event}")

        # Show tension dynamics
        print(f"  Tensions: Mil:{simulator.state.tensions['military_faction']:.0f}% "
              f"Eco:{simulator.state.tensions['economic_pressure']:.0f}% "
              f"Nat:{simulator.state.tensions['nationalist_sentiment']:.0f}%")

        # War outbreak check
        if result['japan_intention'] >= 100 or result['china_readiness'] >= 100:
            print(f"\n💥 WAR BREAKS OUT!")
            break

    print("\nNotice how tensions rise and fall naturally based on:")
    print("- AI agent decisions")
    print("- Coalition dynamics")
    print("- Historical events")
    print("- Game theory calculations")
    print("- Random crises and opportunities")
    print("\nNO FIXED ESCALATION RATE!")


if __name__ == "__main__":
    run_example()