#!/usr/bin/env python3
"""
BDM-based Peace/War Intention Calculator for Peace Simulator
============================================================

Integrates Bueno de Mesquita model to calculate collective peace vs war intentions
for different factions within Japan and China.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
from BDM import BuenoMesquitaModel, Player


@dataclass
class FactionState:
    """Represents a faction's current state in the peace/war calculation"""
    name: str
    peace_position: float  # 0 = extreme peace, 100 = extreme war
    influence: float       # Political influence/clout
    salience: float       # How much they care about this issue
    resolve: float        # How stubborn they are

    def to_bdm_player(self) -> Player:
        """Convert to BDM Player object"""
        return Player(
            id=self.name,
            position=self.peace_position,
            salience=self.salience,
            clout=self.influence,
            resolve=self.resolve
        )


class PeaceWarCalculator:
    """Calculate collective peace/war intention using BDM model"""

    def __init__(self):
        self.japanese_factions = self._initialize_japanese_factions()
        self.chinese_factions = self._initialize_chinese_factions()

    def _initialize_japanese_factions(self) -> Dict[str, FactionState]:
        """Initialize Japanese faction states (1930 starting point)"""
        return {
            "Emperor": FactionState(
                name="Emperor",
                peace_position=20,  # Leans toward peace
                influence=30,       # Symbolic but significant
                salience=70,       # Cares deeply
                resolve=40         # Can be influenced
            ),
            "Tosei_Ha": FactionState(
                name="Tosei-ha (Control Faction)",
                peace_position=75,  # Pro-war but methodical
                influence=25,       # Growing power
                salience=90,       # Very invested
                resolve=80         # Very stubborn
            ),
            "Kodo_Ha": FactionState(
                name="Kodo-ha (Imperial Way)",
                peace_position=85,  # Extremely pro-war
                influence=15,       # Declining but dangerous
                salience=95,       # Fanatically invested
                resolve=90         # Won't compromise
            ),
            "Kwantung_Army": FactionState(
                name="Kwantung Army",
                peace_position=90,  # Aggressive expansionist
                influence=20,       # Autonomous power
                salience=100,      # Completely focused
                resolve=95         # Acts independently
            ),
            "Navy": FactionState(
                name="Imperial Navy",
                peace_position=60,  # Moderate, southern focus
                influence=20,       # Significant voice
                salience=70,       # Interested but flexible
                resolve=60         # Can negotiate
            ),
            "Zaibatsu": FactionState(
                name="Zaibatsu (Corporations)",
                peace_position=45,  # Profit over war
                influence=25,       # Economic power
                salience=80,       # Very interested
                resolve=50         # Pragmatic
            ),
            "Foreign_Ministry": FactionState(
                name="Foreign Ministry",
                peace_position=30,  # Diplomatic solution preferred
                influence=15,       # Limited but present
                salience=85,       # Professional duty
                resolve=45         # Flexible
            ),
            "Diet_Politicians": FactionState(
                name="Diet Politicians",
                peace_position=35,  # Generally peaceful
                influence=10,       # Weakened power
                salience=60,       # Somewhat engaged
                resolve=30         # Easy to pressure
            )
        }

    def _initialize_chinese_factions(self) -> Dict[str, FactionState]:
        """Initialize Chinese faction states (1930 starting point)"""
        return {
            "Chiang_KMT": FactionState(
                name="Chiang/KMT Leadership",
                peace_position=40,  # Wants peace to fight communists
                influence=35,       # Central authority
                salience=85,       # Very concerned
                resolve=65         # Somewhat flexible
            ),
            "Communists": FactionState(
                name="Chinese Communists",
                peace_position=55,  # Mixed - wants KMT weakened
                influence=15,       # Growing influence
                salience=90,       # Highly invested
                resolve=85         # Very determined
            ),
            "Northern_Warlords": FactionState(
                name="Northern Warlords",
                peace_position=35,  # Prefer stability
                influence=20,       # Regional power
                salience=70,       # Concerned
                resolve=50         # Negotiable
            ),
            "Southern_Warlords": FactionState(
                name="Southern Warlords",
                peace_position=30,  # Want peace for autonomy
                influence=15,       # Regional power
                salience=65,       # Moderately concerned
                resolve=55         # Somewhat flexible
            ),
            "Students_Intellectuals": FactionState(
                name="Students/Intellectuals",
                peace_position=70,  # Nationalist, anti-Japanese
                influence=10,       # Moral influence
                salience=95,       # Passionate
                resolve=75         # Committed
            ),
            "Business_Class": FactionState(
                name="Chinese Business Class",
                peace_position=25,  # Want trade not war
                influence=20,       # Economic influence
                salience=75,       # Very concerned
                resolve=40         # Pragmatic
            )
        }

    def apply_player_action(self, country: str, action: str, faction_effects: Dict[str, Dict[str, float]]):
        """
        Apply effects of player action on faction positions

        Args:
            country: "japan" or "china"
            action: The action taken
            faction_effects: Dict of faction_name -> {peace_position: delta, influence: delta, etc}
        """
        factions = self.japanese_factions if country == "japan" else self.chinese_factions

        for faction_name, effects in faction_effects.items():
            if faction_name in factions:
                faction = factions[faction_name]

                # Apply position change
                if "peace_position" in effects:
                    faction.peace_position = max(0, min(100,
                        faction.peace_position + effects["peace_position"]))

                # Apply influence change
                if "influence" in effects:
                    faction.influence = max(1, min(100,
                        faction.influence + effects["influence"]))

                # Apply resolve change
                if "resolve" in effects:
                    faction.resolve = max(0, min(100,
                        faction.resolve + effects["resolve"]))

    def calculate_country_intention(self, country: str) -> Tuple[float, Dict]:
        """
        Calculate overall peace/war intention for a country

        Returns:
            Tuple of (overall_position, faction_details)
            Position: 0 = complete peace, 100 = total war
        """
        factions = self.japanese_factions if country == "japan" else self.chinese_factions

        # Convert to BDM players
        players = [faction.to_bdm_player() for faction in factions.values()]

        # Run BDM model
        model = BuenoMesquitaModel(players, max_iterations=20)
        result = model.run_simulation(verbose=False)

        # Get final consensus position
        overall_position = result['final_outcome']

        # Compile faction details
        faction_details = {}
        for faction_name, faction in factions.items():
            player_id = faction.name
            faction_details[faction_name] = {
                "initial_position": faction.peace_position,
                "final_position": result['final_positions'].get(player_id, faction.peace_position),
                "influence": faction.influence,
                "stance": self._classify_stance(result['final_positions'].get(player_id, faction.peace_position))
            }

        return overall_position, faction_details

    def _classify_stance(self, position: float) -> str:
        """Classify position into stance category"""
        if position < 30:
            return "Strongly Peaceful"
        elif position < 50:
            return "Lean Peaceful"
        elif position < 70:
            return "Lean Warlike"
        elif position < 90:
            return "Strongly Warlike"
        else:
            return "Total War"

    def get_emperor_influence(self) -> float:
        """Get current Emperor influence level"""
        return self.japanese_factions["Emperor"].influence

    def update_emperor_influence(self, delta: float):
        """Update Emperor's influence"""
        emperor = self.japanese_factions["Emperor"]
        emperor.influence = max(5, min(50, emperor.influence + delta))


# Action effect definitions
EMPEROR_ACTIONS = {
    "do_nothing": {
        "description": "保持沉默，不干預政府事務",
        "effects": {
            "Emperor": {"influence": -2},  # Loses influence over time
            "Tosei_Ha": {"peace_position": 3, "influence": 1},
            "Kwantung_Army": {"peace_position": 5, "influence": 1}
        }
    },
    "diplomatic": {
        "description": "發佈和平聲明，呼籲外交解決",
        "effects": {
            "Emperor": {"influence": -3, "peace_position": -5},
            "Foreign_Ministry": {"peace_position": -10, "influence": 3},
            "Tosei_Ha": {"resolve": 5},  # Hardens military resolve
            "Diet_Politicians": {"peace_position": -5}
        }
    },
    "restraint": {
        "description": "私下要求軍部克制",
        "effects": {
            "Emperor": {"influence": -2},
            "Tosei_Ha": {"peace_position": -3},
            "Kwantung_Army": {"peace_position": -2, "resolve": 5}
        }
    },
    "support_military": {
        "description": "支持軍部的'防禦性'行動",
        "effects": {
            "Emperor": {"influence": 2, "peace_position": 10},
            "Tosei_Ha": {"peace_position": 5, "influence": 2},
            "Kwantung_Army": {"peace_position": 8, "resolve": -5}
        }
    },
    "dismiss_minister": {
        "description": "行使權力撤換激進大臣",
        "effects": {
            "Emperor": {"influence": -5},
            "Tosei_Ha": {"peace_position": -5, "influence": -2},
            "Foreign_Ministry": {"influence": 5},
            "Kodo_Ha": {"resolve": 10}  # Angers radicals
        }
    },
    "rally_moderates": {
        "description": "召集溫和派商討和平",
        "effects": {
            "Emperor": {"influence": -1},
            "Zaibatsu": {"peace_position": -8, "influence": 2},
            "Navy": {"peace_position": -5},
            "Foreign_Ministry": {"peace_position": -8, "influence": 2}
        }
    }
}

CHIANG_ACTIONS = {
    "diplomatic": {
        "description": "提議中日直接談判",
        "effects": {
            "Chiang_KMT": {"peace_position": -10},
            "Students_Intellectuals": {"peace_position": 5, "resolve": 5},
            "Business_Class": {"peace_position": -10, "influence": 2}
        }
    },
    "military_preparation": {
        "description": "加強軍事準備，展示決心",
        "effects": {
            "Chiang_KMT": {"peace_position": 15, "influence": 3},
            "Northern_Warlords": {"peace_position": 10},
            "Students_Intellectuals": {"peace_position": -5}
        }
    },
    "unite_with_communists": {
        "description": "與共產黨合作抗日",
        "effects": {
            "Chiang_KMT": {"peace_position": 5},
            "Communists": {"peace_position": 10, "influence": 5},
            "Students_Intellectuals": {"peace_position": -5}
        }
    },
    "international_appeal": {
        "description": "向國際社會求援",
        "effects": {
            "Chiang_KMT": {"influence": 1},
            "Business_Class": {"peace_position": -5},
            "Students_Intellectuals": {"resolve": -5}
        }
    },
    "local_concession": {
        "description": "局部讓步避免全面戰爭",
        "effects": {
            "Chiang_KMT": {"peace_position": -15, "influence": -2},
            "Students_Intellectuals": {"peace_position": 20, "resolve": 10},
            "Northern_Warlords": {"peace_position": -10}
        }
    },
    "mobilize_nationalism": {
        "description": "動員民族主義情緒",
        "effects": {
            "Chiang_KMT": {"peace_position": 20, "influence": 4},
            "Students_Intellectuals": {"peace_position": 25, "influence": 3},
            "Business_Class": {"peace_position": 5}
        }
    }
}


if __name__ == "__main__":
    # Test the calculator
    calculator = PeaceWarCalculator()

    print("Initial State Analysis")
    print("=" * 60)

    # Calculate Japan's initial state
    japan_position, japan_details = calculator.calculate_country_intention("japan")
    print(f"\nJapan Overall Position: {japan_position:.1f}/100 (0=peace, 100=war)")
    print("Japanese Factions:")
    for faction, details in japan_details.items():
        print(f"  {faction}: {details['stance']} (Position: {details['final_position']:.1f})")

    # Calculate China's initial state
    china_position, china_details = calculator.calculate_country_intention("china")
    print(f"\nChina Overall Position: {china_position:.1f}/100")
    print("Chinese Factions:")
    for faction, details in china_details.items():
        print(f"  {faction}: {details['stance']} (Position: {details['final_position']:.1f})")

    # Test Emperor action
    print("\n" + "=" * 60)
    print("Testing Emperor 'do_nothing' action...")
    calculator.apply_player_action("japan", "do_nothing", EMPEROR_ACTIONS["do_nothing"]["effects"])

    japan_position2, japan_details2 = calculator.calculate_country_intention("japan")
    print(f"Japan Position After: {japan_position2:.1f}/100")
    print(f"Change: {japan_position2 - japan_position:+.1f}")
    print(f"Emperor Influence: {calculator.get_emperor_influence():.1f}")