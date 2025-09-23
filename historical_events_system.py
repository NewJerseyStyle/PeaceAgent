#!/usr/bin/env python3
"""
Historical Events System for Peace Simulator
============================================

This module provides a progressive historical events system that can simulate
different trigger points and escalation paths based on player actions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class EventSeverity(Enum):
    MINOR = "minor"          # Local incident
    MODERATE = "moderate"    # Regional tension
    MAJOR = "major"          # National crisis
    CRITICAL = "critical"    # International crisis


@dataclass
class HistoricalEvent:
    """Represents a historical event that can be triggered."""
    name: str
    date: str
    severity: EventSeverity
    description: str
    preconditions: List[str]
    consequences: List[str]
    can_be_prevented: bool
    prevention_difficulty: int  # 1-10 scale
    historical_outcome: str


class HistoricalEventsManager:
    """Manages progressive historical events based on simulation state."""

    def __init__(self, start_year: int = 1930):
        self.start_year = start_year
        self.current_date = f"{start_year}-01-01"
        self.triggered_events: List[str] = []
        self.prevented_events: List[str] = []
        self.events_database = self._initialize_events_database()

    def _initialize_events_database(self) -> Dict[str, HistoricalEvent]:
        """Initialize the database of historical events."""
        return {
            # 1930-1931 Events
            "economic_crisis_1930": HistoricalEvent(
                name="Global Economic Crisis Impact",
                date="1930-01-01",
                severity=EventSeverity.MAJOR,
                description="日本在經濟危機下，日本軍部的主戰派地位上升。農村貧困和城市失業引發社會動盪。",
                preconditions=[],
                consequences=["military_influence_increase", "social_unrest"],
                can_be_prevented=False,
                prevention_difficulty=10,
                historical_outcome="Military factions gained power"
            ),

            "london_naval_treaty_1930": HistoricalEvent(
                name="London Naval Treaty Crisis",
                date="1930-04-22",
                severity=EventSeverity.MODERATE,
                description="倫敦海軍條約引發日本海軍內部分裂，艦隊派與條約派對立加劇。",
                preconditions=["economic_crisis_1930"],
                consequences=["navy_faction_split", "civilian_government_weakened"],
                can_be_prevented=False,
                prevention_difficulty=8,
                historical_outcome="Treaty signed but military opposition grew"
            ),

            "march_incident_1931": HistoricalEvent(
                name="March Incident (三月事件)",
                date="1931-03-20",
                severity=EventSeverity.MODERATE,
                description="軍部激進派計劃政變，試圖建立軍事政府。雖然失敗但顯示軍部野心。",
                preconditions=["military_influence_increase"],
                consequences=["military_radicalization", "government_instability"],
                can_be_prevented=True,
                prevention_difficulty=6,
                historical_outcome="Coup attempt failed but militarists emboldened"
            ),

            "mukden_incident_1931": HistoricalEvent(
                name="Mukden Incident (九一八事變)",
                date="1931-09-18",
                severity=EventSeverity.CRITICAL,
                description="關東軍在瀋陽製造鐵路爆炸事件，藉口入侵滿洲。這是日本軍國主義擴張的關鍵轉折點。",
                preconditions=["military_influence_increase", "march_incident_1931"],
                consequences=["manchuria_occupation", "league_of_nations_involvement", "chinese_resistance"],
                can_be_prevented=True,
                prevention_difficulty=7,
                historical_outcome="Japan occupied Manchuria, created Manchukuo"
            ),

            # 1932-1936 Events
            "january_28_incident_1932": HistoricalEvent(
                name="January 28 Incident (一·二八事變)",
                date="1932-01-28",
                severity=EventSeverity.MAJOR,
                description="日軍在上海製造事端，引發中日在上海的武裝衝突。",
                preconditions=["mukden_incident_1931"],
                consequences=["shanghai_conflict", "international_attention"],
                can_be_prevented=True,
                prevention_difficulty=5,
                historical_outcome="Limited conflict, ceasefire achieved"
            ),

            "may_15_incident_1932": HistoricalEvent(
                name="May 15 Incident (五一五事件)",
                date="1932-05-15",
                severity=EventSeverity.MAJOR,
                description="海軍青年軍官刺殺首相犬養毅，標誌著日本民主政治的終結。",
                preconditions=["military_radicalization"],
                consequences=["civilian_government_collapse", "military_government"],
                can_be_prevented=True,
                prevention_difficulty=6,
                historical_outcome="Prime Minister assassinated, military control increased"
            ),

            "february_26_incident_1936": HistoricalEvent(
                name="February 26 Incident (二二六事件)",
                date="1936-02-26",
                severity=EventSeverity.MAJOR,
                description="皇道派青年軍官發動政變，試圖清君側。雖然失敗但加速軍國主義。",
                preconditions=["military_government", "faction_conflict"],
                consequences=["kodo_ha_purge", "tosei_ha_dominance"],
                can_be_prevented=True,
                prevention_difficulty=5,
                historical_outcome="Coup failed, Control Faction dominated"
            ),

            "xian_incident_1936": HistoricalEvent(
                name="Xi'an Incident (西安事變)",
                date="1936-12-12",
                severity=EventSeverity.MAJOR,
                description="張學良扣留蔣介石，迫使國共合作抗日。改變了中國內戰格局。",
                preconditions=["chinese_civil_war", "japanese_threat"],
                consequences=["united_front_formed", "anti_japanese_coalition"],
                can_be_prevented=True,
                prevention_difficulty=4,
                historical_outcome="KMT-CCP cooperation against Japan"
            ),

            # 1937 Events
            "marco_polo_bridge_1937": HistoricalEvent(
                name="Marco Polo Bridge Incident (盧溝橋事變)",
                date="1937-07-07",
                severity=EventSeverity.CRITICAL,
                description="日軍在盧溝橋製造衝突，成為全面侵華戰爭的導火線。",
                preconditions=["tosei_ha_dominance", "military_expansion_policy"],
                consequences=["full_scale_war", "international_condemnation"],
                can_be_prevented=True,
                prevention_difficulty=8,
                historical_outcome="Full-scale Second Sino-Japanese War began"
            ),

            "shanghai_battle_1937": HistoricalEvent(
                name="Battle of Shanghai (淞滬會戰)",
                date="1937-08-13",
                severity=EventSeverity.CRITICAL,
                description="中日在上海爆發大規模戰役，標誌著戰爭全面升級。",
                preconditions=["marco_polo_bridge_1937"],
                consequences=["massive_casualties", "nanjing_threatened"],
                can_be_prevented=True,
                prevention_difficulty=9,
                historical_outcome="Japanese victory after heavy casualties"
            ),

            "nanjing_massacre_1937": HistoricalEvent(
                name="Nanjing Massacre (南京大屠殺)",
                date="1937-12-13",
                severity=EventSeverity.CRITICAL,
                description="日軍攻陷南京後進行大規模屠殺，震驚世界。",
                preconditions=["shanghai_battle_1937", "nanjing_fall"],
                consequences=["international_outrage", "chinese_determination"],
                can_be_prevented=True,
                prevention_difficulty=9,
                historical_outcome="Mass atrocities committed"
            )
        }

    def check_event_triggers(self, game_state: Dict[str, any], current_turn: int) -> List[HistoricalEvent]:
        """Check which historical events should trigger based on current state."""
        triggered = []

        # Calculate current date based on turn
        days_per_turn = 30  # Assume each turn is roughly a month
        current_date = datetime.strptime(f"{self.start_year}-01-01", "%Y-%m-%d") + timedelta(days=current_turn * days_per_turn)

        for event_id, event in self.events_database.items():
            if event_id in self.triggered_events or event_id in self.prevented_events:
                continue

            event_date = datetime.strptime(event.date, "%Y-%m-%d")

            # Check if we've reached the event date
            if current_date >= event_date:
                # Check preconditions
                preconditions_met = True
                for precondition in event.preconditions:
                    if precondition not in self.triggered_events and precondition not in game_state.get("conditions", []):
                        preconditions_met = False
                        break

                if preconditions_met:
                    # Check if event can be prevented by player actions
                    if event.can_be_prevented and self._check_prevention_conditions(event, game_state):
                        self.prevented_events.append(event_id)
                        triggered.append(self._create_prevention_event(event))
                    else:
                        self.triggered_events.append(event_id)
                        triggered.append(event)

                        # Add consequences to game state
                        if "conditions" not in game_state:
                            game_state["conditions"] = []
                        game_state["conditions"].extend(event.consequences)

        return triggered

    def _check_prevention_conditions(self, event: HistoricalEvent, game_state: Dict[str, any]) -> bool:
        """Check if player actions have prevented an event."""
        peace_score = game_state.get("peace_metrics", {}).get("conflict_prevention", 50)
        diplomatic_success = game_state.get("peace_metrics", {}).get("diplomatic_success", 0)

        # Calculate prevention chance based on peace efforts
        prevention_score = (peace_score / 100 * 5) + (diplomatic_success / 100 * 5)

        # Compare with event difficulty
        return prevention_score >= event.prevention_difficulty

    def _create_prevention_event(self, original_event: HistoricalEvent) -> HistoricalEvent:
        """Create an alternative peaceful event when original is prevented."""
        return HistoricalEvent(
            name=f"{original_event.name} - 避免",
            date=original_event.date,
            severity=EventSeverity.MINOR,
            description=f"透過外交努力和克制，成功避免了{original_event.name}的發生。",
            preconditions=original_event.preconditions,
            consequences=["peace_maintained", "diplomatic_success"],
            can_be_prevented=False,
            prevention_difficulty=0,
            historical_outcome="Crisis averted through diplomacy"
        )

    def get_next_potential_event(self) -> Optional[HistoricalEvent]:
        """Get the next historical event that might occur."""
        current_date = datetime.strptime(self.current_date, "%Y-%m-%d")

        upcoming_events = []
        for event_id, event in self.events_database.items():
            if event_id not in self.triggered_events and event_id not in self.prevented_events:
                event_date = datetime.strptime(event.date, "%Y-%m-%d")
                if event_date > current_date:
                    upcoming_events.append((event_date, event))

        if upcoming_events:
            upcoming_events.sort(key=lambda x: x[0])
            return upcoming_events[0][1]
        return None

    def get_historical_context(self, current_turn: int) -> str:
        """Get historical context description for the current period."""
        year = self.start_year + (current_turn // 12)
        month = (current_turn % 12) + 1

        contexts = {
            1930: "全球經濟大蕭條影響日本，軍國主義思想抬頭。中國內戰持續，國共對立。",
            1931: "日本軍部策劃擴張，關東軍蠢蠢欲動。中國忙於內戰，防務空虛。",
            1932: "滿洲國成立，國際聯盟調查團介入。中國呼籲國際支持。",
            1933: "日本退出國際聯盟，國際孤立加深。希特勒在德國掌權。",
            1934: "日本確立大東亞共榮圈構想。中國進行圍剿紅軍。",
            1935: "華北事變，日本加強對華北控制。中國共產黨長征。",
            1936: "日德防共協定簽署。西安事變促成國共合作。",
            1937: "中日關係極度緊張，戰爭一觸即發。國際局勢動盪不安。"
        }

        return contexts.get(year, f"{year}年{month}月：局勢持續發展中...")