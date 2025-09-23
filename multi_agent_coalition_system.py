#!/usr/bin/env python3
"""
Multi-Agent Coalition System for Historical Simulation
=======================================================

Integrates CrewAI agents with BDM coalition dynamics and resource constraints.
Players distribute limited resources across topics using softmax-like constraints.
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

from BDM import BuenoMesquitaModel, Player as BDMPlayer


class Topic(Enum):
    """Key topics/issues that actors must allocate resources to"""
    CIVIL_WAR = "civil_war"           # Internal conflict focus
    ECONOMY = "economy"                # Economic development
    JAPAN_DEFENSE = "japan_defense"    # Anti-Japanese military
    DIPLOMACY = "diplomacy"            # International relations
    SOCIAL_REFORM = "social_reform"    # Domestic reforms
    IDEOLOGY = "ideology"              # Political ideology


@dataclass
class ResourceAllocation:
    """Softmax-constrained resource allocation across topics"""
    raw_preferences: Dict[Topic, float] = field(default_factory=dict)
    temperature: float = 1.0  # Controls how extreme allocations can be

    def get_weights(self) -> Dict[Topic, float]:
        """Apply softmax to ensure weights sum to 1"""
        values = np.array([self.raw_preferences.get(t, 1.0) for t in Topic])
        exp_values = np.exp(values / self.temperature)
        softmax = exp_values / exp_values.sum()
        return {topic: float(weight) for topic, weight in zip(Topic, softmax)}

    def update_preference(self, topic: Topic, delta: float):
        """Update preference for a topic"""
        current = self.raw_preferences.get(topic, 1.0)
        self.raw_preferences[topic] = max(0.1, current + delta)


@dataclass
class Actor:
    """Represents a faction/player in the simulation"""
    name: str
    faction_type: str  # "warlord", "party", "foreign", etc.

    # Resources and capabilities
    military_strength: float = 10.0
    economic_power: float = 10.0
    political_influence: float = 10.0
    international_support: float = 0.0

    # Resource allocation
    resource_allocation: ResourceAllocation = field(default_factory=ResourceAllocation)

    # Positions on topics (0-100 scale)
    base_positions: Dict[Topic, float] = field(default_factory=dict)

    # Coalition membership
    coalition: Optional[str] = None
    independence_threshold: float = 0.2  # Benefit needed to stay independent

    def get_effective_position(self, topic: Topic) -> float:
        """Get position weighted by resource allocation"""
        base = self.base_positions.get(topic, 50)
        weight = self.resource_allocation.get_weights()[topic]
        return base * weight

    def get_total_power(self) -> float:
        """Calculate total power/influence"""
        return (self.military_strength + self.economic_power +
                self.political_influence + self.international_support)

    def to_bdm_player(self, topic: Topic) -> BDMPlayer:
        """Convert to BDM player for specific topic"""
        position = self.get_effective_position(topic)
        weight = self.resource_allocation.get_weights()[topic]

        return BDMPlayer(
            id=self.name,
            position=position,
            salience=weight * 100,  # Convert to 0-100 scale
            clout=self.get_total_power() / 40,  # Normalize
            resolve=50 + weight * 30  # Higher allocation = more resolve
        )


class Coalition:
    """Represents a coalition of actors"""
    def __init__(self, name: str, leader: str):
        self.name = name
        self.leader = leader
        self.members: Set[str] = {leader}
        self.policies: Dict[Topic, float] = {}

    def add_member(self, actor_name: str):
        """Add actor to coalition"""
        self.members.add(actor_name)

    def remove_member(self, actor_name: str):
        """Remove actor from coalition"""
        self.members.discard(actor_name)
        if actor_name == self.leader and self.members:
            self.leader = next(iter(self.members))

    def calculate_coalition_position(self, topic: Topic, actors: Dict[str, Actor]) -> float:
        """Calculate weighted coalition position on topic"""
        total_weight = 0
        weighted_position = 0

        for member_name in self.members:
            if member_name in actors:
                actor = actors[member_name]
                weight = actor.get_total_power() * actor.resource_allocation.get_weights()[topic]
                weighted_position += actor.get_effective_position(topic) * weight
                total_weight += weight

        return weighted_position / total_weight if total_weight > 0 else 50


class MultiAgentCoalitionSystem:
    """Main system managing actors, coalitions, and dynamics"""

    def __init__(self):
        self.actors: Dict[str, Actor] = {}
        self.coalitions: Dict[str, Coalition] = {}
        self.turn = 0
        self.history = []

        # Initialize Chinese factions
        self._initialize_chinese_actors()
        # Initialize Japanese factions
        self._initialize_japanese_actors()

    def _initialize_chinese_actors(self):
        """Initialize Chinese faction actors"""

        # Nationalist Government
        self.actors["Chiang_KMT"] = Actor(
            name="Chiang_KMT",
            faction_type="party",
            military_strength=25,
            economic_power=20,
            political_influence=30,
            international_support=15,
            base_positions={
                Topic.CIVIL_WAR: 80,      # Strongly anti-communist
                Topic.ECONOMY: 60,         # Moderate development
                Topic.JAPAN_DEFENSE: 40,   # Initially defensive
                Topic.DIPLOMACY: 70,       # Pro-Western
                Topic.SOCIAL_REFORM: 40,   # Conservative
                Topic.IDEOLOGY: 30         # Anti-communist
            }
        )

        # Communist Party
        self.actors["CCP"] = Actor(
            name="CCP",
            faction_type="party",
            military_strength=10,
            economic_power=5,
            political_influence=15,
            base_positions={
                Topic.CIVIL_WAR: 60,       # Defensive in civil war
                Topic.ECONOMY: 80,         # Land reform focus
                Topic.JAPAN_DEFENSE: 70,   # Anti-Japanese
                Topic.DIPLOMACY: 40,       # Pro-Soviet
                Topic.SOCIAL_REFORM: 90,   # Revolutionary
                Topic.IDEOLOGY: 95         # Communist
            }
        )

        # Major Warlords
        self.actors["Zhang_Xueliang"] = Actor(
            name="Zhang_Xueliang",
            faction_type="warlord",
            military_strength=15,
            economic_power=12,
            political_influence=10,
            base_positions={
                Topic.CIVIL_WAR: 30,       # Wants peace
                Topic.JAPAN_DEFENSE: 85,   # Very anti-Japanese
                Topic.DIPLOMACY: 60,       # Open to allies
            }
        )

        self.actors["Yan_Xishan"] = Actor(
            name="Yan_Xishan",
            faction_type="warlord",
            military_strength=12,
            economic_power=10,
            political_influence=8,
            base_positions={
                Topic.CIVIL_WAR: 40,
                Topic.ECONOMY: 70,         # Model province
                Topic.JAPAN_DEFENSE: 50,
            }
        )

        self.actors["Guangxi_Clique"] = Actor(
            name="Guangxi_Clique",
            faction_type="warlord",
            military_strength=10,
            economic_power=8,
            political_influence=7,
            base_positions={
                Topic.CIVIL_WAR: 50,
                Topic.JAPAN_DEFENSE: 60,
                Topic.SOCIAL_REFORM: 50
            }
        )

    def _initialize_japanese_actors(self):
        """Initialize Japanese faction actors"""

        self.actors["Tosei_Ha"] = Actor(
            name="Tosei_Ha",
            faction_type="military",
            military_strength=30,
            political_influence=25,
            base_positions={
                Topic.JAPAN_DEFENSE: 90,  # Expansion as defense
                Topic.ECONOMY: 70,
                Topic.IDEOLOGY: 80
            }
        )

        self.actors["Kodo_Ha"] = Actor(
            name="Kodo_Ha",
            faction_type="military",
            military_strength=15,
            political_influence=10,
            base_positions={
                Topic.JAPAN_DEFENSE: 95,
                Topic.IDEOLOGY: 90
            }
        )

    def evaluate_coalition_benefit(self, actor: Actor, coalition: Coalition, topic: Topic) -> float:
        """Evaluate benefit of joining/staying in coalition for a topic"""

        # Convert actors to BDM players
        actors_in_topic = []
        for name, a in self.actors.items():
            if a.resource_allocation.get_weights()[topic] > 0.1:  # Active in topic
                actors_in_topic.append(a.to_bdm_player(topic))

        if not actors_in_topic:
            return 0

        # Run BDM simulation
        model = BuenoMesquitaModel(actors_in_topic, max_iterations=10)
        result = model.run_simulation(verbose=False)

        # Get actor's outcome
        actor_outcome = result['final_positions'].get(actor.name, 50)

        # Calculate coalition outcome
        coalition_position = coalition.calculate_coalition_position(topic, self.actors)

        # Benefit is how close outcome is to actor's preferred position
        benefit = 100 - abs(actor_outcome - actor.base_positions.get(topic, 50))
        coalition_benefit = 100 - abs(coalition_position - actor.base_positions.get(topic, 50))

        return coalition_benefit - benefit

    def update_coalitions(self):
        """Dynamic coalition formation/defection based on BDM calculations"""

        for actor_name, actor in self.actors.items():
            if actor.faction_type == "foreign":
                continue  # Foreign actors don't form coalitions

            best_coalition = None
            best_benefit = actor.independence_threshold

            # Evaluate each coalition
            for coalition_name, coalition in self.coalitions.items():
                total_benefit = 0
                for topic in Topic:
                    weight = actor.resource_allocation.get_weights()[topic]
                    benefit = self.evaluate_coalition_benefit(actor, coalition, topic)
                    total_benefit += benefit * weight

                if total_benefit > best_benefit:
                    best_coalition = coalition_name
                    best_benefit = total_benefit

            # Update coalition membership
            if best_coalition:
                if actor.coalition != best_coalition:
                    # Leave old coalition
                    if actor.coalition and actor.coalition in self.coalitions:
                        self.coalitions[actor.coalition].remove_member(actor_name)

                    # Join new coalition
                    actor.coalition = best_coalition
                    self.coalitions[best_coalition].add_member(actor_name)
            else:
                # Go independent
                if actor.coalition and actor.coalition in self.coalitions:
                    self.coalitions[actor.coalition].remove_member(actor_name)
                actor.coalition = None

    def simulate_turn(self, player_actions: Optional[Dict[str, Dict]] = None):
        """Simulate one turn with optional player input"""

        self.turn += 1

        # Apply player actions if any
        if player_actions:
            for actor_name, actions in player_actions.items():
                if actor_name in self.actors:
                    actor = self.actors[actor_name]

                    # Update resource allocations
                    if "allocations" in actions:
                        for topic, delta in actions["allocations"].items():
                            actor.resource_allocation.update_preference(Topic(topic), delta)

                    # Update positions
                    if "positions" in actions:
                        for topic, position in actions["positions"].items():
                            actor.base_positions[Topic(topic)] = position

        # Update coalitions based on new positions
        self.update_coalitions()

        # Calculate outcomes for each topic
        outcomes = {}
        for topic in Topic:
            # Get all actors involved in this topic
            actors_in_topic = []
            for actor in self.actors.values():
                if actor.resource_allocation.get_weights()[topic] > 0.1:
                    actors_in_topic.append(actor.to_bdm_player(topic))

            if actors_in_topic:
                model = BuenoMesquitaModel(actors_in_topic, max_iterations=10)
                result = model.run_simulation(verbose=False)
                outcomes[topic.value] = {
                    "final_position": result['final_outcome'],
                    "actor_positions": result['final_positions']
                }

        # Record history
        self.history.append({
            "turn": self.turn,
            "coalitions": {
                name: list(coalition.members)
                for name, coalition in self.coalitions.items()
            },
            "outcomes": outcomes
        })

        return outcomes

    def get_status(self) -> Dict:
        """Get current system status"""
        return {
            "turn": self.turn,
            "actors": {
                name: {
                    "power": actor.get_total_power(),
                    "coalition": actor.coalition,
                    "resource_focus": max(
                        actor.resource_allocation.get_weights().items(),
                        key=lambda x: x[1]
                    )[0].value if actor.resource_allocation.get_weights() else None
                }
                for name, actor in self.actors.items()
            },
            "coalitions": {
                name: {
                    "leader": coalition.leader,
                    "members": list(coalition.members),
                    "total_power": sum(
                        self.actors[m].get_total_power()
                        for m in coalition.members
                        if m in self.actors
                    )
                }
                for name, coalition in self.coalitions.items()
            }
        }


# Example usage and testing
if __name__ == "__main__":
    system = MultiAgentCoalitionSystem()

    # Initialize coalitions
    system.coalitions["KMT_Government"] = Coalition("KMT_Government", "Chiang_KMT")
    system.coalitions["United_Front"] = Coalition("United_Front", "CCP")
    system.coalitions["Northern_Alliance"] = Coalition("Northern_Alliance", "Yan_Xishan")

    # Run initial coalition formation
    system.update_coalitions()

    print("Initial Status:")
    print(json.dumps(system.get_status(), indent=2))

    # Simulate player action - Chiang focuses on anti-Japanese defense
    player_actions = {
        "Chiang_KMT": {
            "allocations": {
                "japan_defense": 2.0,
                "civil_war": -1.0
            },
            "positions": {
                "japan_defense": 70  # More anti-Japanese
            }
        }
    }

    outcomes = system.simulate_turn(player_actions)

    print("\nAfter Turn 1:")
    print(json.dumps(system.get_status(), indent=2))
    print("\nOutcomes:")
    print(json.dumps(outcomes, indent=2))