"""
Bueno de Mesquita Policy Prediction Model
A Python implementation of the new forecasting model from "A New Model for Predicting Policy Choices" (2010)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class PlayerType(Enum):
    HAWK = "hawk"
    DOVE = "dove"

class ResponseType(Enum):
    RETALIATORY = "retaliatory"
    PACIFIC = "pacific"

@dataclass
class Player:
    """Represents a player/stakeholder in the policy game"""
    id: str
    position: float  # Current negotiating position (0-100 scale)
    ideal_point: Optional[float] = None  # Ideal position if different from current
    salience: float = 50.0  # Priority/attention to issue (0-100)
    clout: float = 1.0  # Potential influence/persuasion power
    resolve: float = 50.0  # Willingness to compromise vs. stick to position (0-100)

    # Belief probabilities (updated via Bayes' rule)
    hawk_prob: float = 0.5  # Probability of being a hawk
    retaliatory_prob: float = 0.5  # Probability of being retaliatory

    def __post_init__(self):
        if self.ideal_point is None:
            self.ideal_point = self.position

@dataclass
class GameOutcome:
    """Represents the outcome of a dyadic game"""
    player_a: str
    player_b: str
    outcome_type: str
    payoff_a: float
    payoff_b: float
    new_position_a: float
    new_position_b: float
    proposal: Optional[float] = None

class BuenoMesquitaModel:
    """
    Implementation of Bueno de Mesquita's new forecasting model for policy prediction.

    This model simulates iterative negotiations between multiple stakeholders,
    incorporating uncertainty about player types and updating beliefs via Bayes' rule.
    """

    def __init__(self, players: List[Player],
                 theta: float = 0.4, beta: float = 0.4,
                 alpha: float = 0.2, tau: float = 0.15,
                 gamma: float = 0.1, phi: float = 0.05,
                 max_iterations: int = 50):
        """
        Initialize the model with players and cost parameters.

        Args:
            players: List of Player objects
            theta: Weight for position utility in Cobb-Douglas function
            beta: Weight for resolve utility in Cobb-Douglas function
            alpha: Cost of trying to coerce and meeting resistance
            tau: Cost of being coerced and resisting
            gamma: Cost of being coerced and not resisting
            phi: Cost of coercing (failed credible threat)
            max_iterations: Maximum number of game iterations
        """
        self.players = {p.id: p for p in players}
        self.player_ids = list(self.players.keys())
        self.n_players = len(players)

        # Utility function parameters (Cobb-Douglas)
        self.theta = theta
        self.beta = beta

        # Cost parameters
        self.alpha = alpha  # Cost of trying to coerce and meeting resistance
        self.tau = tau      # Cost of being coerced and resisting
        self.gamma = gamma  # Cost of being coerced and not resisting
        self.phi = phi      # Cost of coercing (failed credible threat)

        self.max_iterations = max_iterations
        self.iteration_history = []
        self.outcome_history = []

    def calculate_weighted_median(self) -> float:
        """Calculate the weighted median position of all players"""
        positions = [p.position for p in self.players.values()]
        weights = [p.clout * p.salience for p in self.players.values()]

        # Create weighted data points
        weighted_positions = []
        for pos, weight in zip(positions, weights):
            weighted_positions.extend([pos] * int(weight * 100))

        return np.median(weighted_positions) if weighted_positions else np.mean(positions)

    def calculate_utility(self, player_id: str, outcome_position: float,
                         resolve_outcome: float) -> float:
        """
        Calculate utility using Cobb-Douglas function.

        U = (1 - |position_diff|)^theta * (1 - |resolve_diff|)^beta
        """
        player = self.players[player_id]

        position_diff = abs(player.position - outcome_position) / 100.0
        resolve_diff = abs(player.resolve - resolve_outcome) / 100.0

        position_utility = max(0.001, 1 - position_diff)
        resolve_utility = max(0.001, 1 - resolve_diff)

        return (position_utility ** self.theta) * (resolve_utility ** self.beta)

    def calculate_win_probability(self, player_a_id: str, player_b_id: str) -> float:
        """Calculate probability that player A wins against player B"""
        # Get support from other players
        player_a = self.players[player_a_id]
        player_b = self.players[player_b_id]

        support_a = 0
        support_b = 0

        for pid, player in self.players.items():
            if pid in [player_a_id, player_b_id]:
                continue

            # Players support the side closer to their position
            dist_a = abs(player.position - player_a.position)
            dist_b = abs(player.position - player_b.position)

            if dist_a < dist_b:
                support_a += player.clout * player.salience * (player.resolve / 100.0)
            else:
                support_b += player.clout * player.salience * (player.resolve / 100.0)

        # Add own capabilities
        capability_a = player_a.clout * player_a.salience * (player_a.resolve / 100.0)
        capability_b = player_b.clout * player_b.salience * (player_b.resolve / 100.0)

        total_a = capability_a + support_a
        total_b = capability_b + support_b

        if total_a + total_b == 0:
            return 0.5

        return total_a / (total_a + total_b)

    def generate_proposal(self, proposer_id: str, target_id: str) -> float:
        """Generate an endogenous proposal that maximizes proposer's expected utility"""
        proposer = self.players[proposer_id]
        target = self.players[target_id]

        # Simple heuristic: propose a position that makes target indifferent
        # between accepting and fighting
        compromise_factor = target.resolve / 100.0
        proposal = proposer.position + compromise_factor * (target.position - proposer.position)

        return max(0, min(100, proposal))

    def is_proposal_credible(self, proposal: float, target_id: str) -> bool:
        """Check if a proposal is credible based on target's resolve"""
        target = self.players[target_id]
        position_change = abs(proposal - target.position)
        max_acceptable_change = target.resolve

        return position_change <= max_acceptable_change

    def update_beliefs_bayes(self, observer_id: str, observed_id: str,
                           action: str, outcome: str):
        """Update beliefs about player types using Bayes' rule"""
        observer = self.players[observer_id]

        # Simple belief updating based on observed actions
        if action == "propose" and outcome == "aggressive":
            # More likely to be a hawk
            observer.hawk_prob = min(0.9, observer.hawk_prob * 1.2)
        elif action == "propose" and outcome == "conciliatory":
            # More likely to be a dove
            observer.hawk_prob = max(0.1, observer.hawk_prob * 0.8)
        elif action == "resist" and outcome == "successful":
            # More likely to be retaliatory
            observer.retaliatory_prob = min(0.9, observer.retaliatory_prob * 1.2)
        elif action == "backdown":
            # More likely to be pacific
            observer.retaliatory_prob = max(0.1, observer.retaliatory_prob * 0.8)

    def play_dyadic_game(self, player_a_id: str, player_b_id: str) -> GameOutcome:
        """Play a single dyadic game between two players"""
        player_a = self.players[player_a_id]
        player_b = self.players[player_b_id]

        # Player A decides whether to make a proposal
        proposal = self.generate_proposal(player_a_id, player_b_id)

        if not self.is_proposal_credible(proposal, player_b_id):
            # Proposal not credible, maintain status quo
            return GameOutcome(
                player_a_id, player_b_id, "status_quo",
                self.calculate_utility(player_a_id, player_a.position, player_a.resolve),
                self.calculate_utility(player_b_id, player_b.position, player_b.resolve),
                player_a.position, player_b.position
            )

        # Player B decides response based on expected utilities
        accept_utility = self.calculate_utility(player_b_id, proposal, player_b.resolve)
        resist_utility = self.calculate_utility(player_b_id, player_b.position, player_b.resolve) - self.tau

        if accept_utility > resist_utility:
            # B accepts proposal
            payoff_a = self.calculate_utility(player_a_id, proposal, player_a.resolve)
            payoff_b = accept_utility
            return GameOutcome(
                player_a_id, player_b_id, "accept",
                payoff_a, payoff_b, player_a.position, proposal, proposal
            )
        else:
            # B resists, leading to conflict
            win_prob = self.calculate_win_probability(player_a_id, player_b_id)

            if np.random.random() < win_prob:
                # A wins
                outcome_position = proposal
                payoff_a = self.calculate_utility(player_a_id, outcome_position, player_a.resolve) - self.alpha
                payoff_b = self.calculate_utility(player_b_id, outcome_position, player_b.resolve) - self.tau
            else:
                # B wins
                outcome_position = player_b.position
                payoff_a = self.calculate_utility(player_a_id, outcome_position, player_a.resolve) - self.alpha
                payoff_b = self.calculate_utility(player_b_id, outcome_position, player_b.resolve) - self.tau

            return GameOutcome(
                player_a_id, player_b_id, "conflict",
                payoff_a, payoff_b,
                outcome_position if win_prob > 0.5 else player_a.position,
                outcome_position if win_prob <= 0.5 else player_b.position,
                proposal
            )

    def calculate_round_payoffs(self, outcomes: List[GameOutcome]) -> Dict[str, float]:
        """Calculate total payoffs for each player in a round"""
        payoffs = {pid: 0.0 for pid in self.player_ids}

        for outcome in outcomes:
            payoffs[outcome.player_a] += outcome.payoff_a
            payoffs[outcome.player_b] += outcome.payoff_b

        return payoffs

    def update_positions(self, outcomes: List[GameOutcome]):
        """Update player positions based on credible proposals received"""
        position_changes = {pid: [] for pid in self.player_ids}

        for outcome in outcomes:
            if outcome.proposal is not None and self.is_proposal_credible(outcome.proposal, outcome.player_b):
                position_changes[outcome.player_b].append(outcome.proposal)

        # Update positions as weighted mean of credible proposals
        for pid in self.player_ids:
            if position_changes[pid]:
                # Weight by clout * salience of proposers
                weighted_proposals = []
                weights = []

                for outcome in outcomes:
                    if (outcome.player_b == pid and outcome.proposal is not None and
                        self.is_proposal_credible(outcome.proposal, pid)):
                        proposer = self.players[outcome.player_a]
                        weights.append(proposer.clout * proposer.salience)
                        weighted_proposals.append(outcome.proposal)

                if weighted_proposals:
                    new_position = np.average(weighted_proposals, weights=weights)
                    self.players[pid].position = new_position

    def should_terminate(self, current_payoffs: Dict[str, float],
                        previous_payoffs: Dict[str, float]) -> bool:
        """Check if game should terminate based on payoff improvement"""
        current_sum = sum(current_payoffs.values())
        previous_sum = sum(previous_payoffs.values())

        # Terminate if average welfare is expected to decline
        return current_sum <= previous_sum

    def run_simulation(self, verbose: bool = False) -> Dict:
        """
        Run the complete iterative game simulation.

        Returns:
            Dictionary containing final positions, convergence info, and history
        """
        iteration = 0
        previous_payoffs = {pid: 0.0 for pid in self.player_ids}

        while iteration < self.max_iterations:
            if verbose:
                print(f"\n--- Iteration {iteration + 1} ---")

            # Play all N(N-1) dyadic games
            round_outcomes = []
            for i, player_a_id in enumerate(self.player_ids):
                for j, player_b_id in enumerate(self.player_ids):
                    if i != j:
                        outcome = self.play_dyadic_game(player_a_id, player_b_id)
                        round_outcomes.append(outcome)

            # Calculate payoffs
            current_payoffs = self.calculate_round_payoffs(round_outcomes)

            if verbose:
                print("Current positions:", {pid: f"{self.players[pid].position:.1f}"
                                          for pid in self.player_ids})
                print("Payoffs:", {pid: f"{payoff:.3f}" for pid, payoff in current_payoffs.items()})

            # Store history
            self.iteration_history.append({
                'iteration': iteration,
                'positions': {pid: self.players[pid].position for pid in self.player_ids},
                'payoffs': current_payoffs.copy(),
                'outcomes': round_outcomes.copy()
            })

            # Check termination condition
            if iteration > 0 and self.should_terminate(current_payoffs, previous_payoffs):
                if verbose:
                    print(f"Converged after {iteration + 1} iterations")
                break

            # Update positions based on proposals
            self.update_positions(round_outcomes)

            # Update some model parameters (simple heuristic)
            for pid in self.player_ids:
                player = self.players[pid]
                # Slightly adjust resolve based on success
                if current_payoffs[pid] > previous_payoffs.get(pid, 0):
                    player.resolve = min(100, player.resolve * 1.01)
                else:
                    player.resolve = max(0, player.resolve * 0.99)

            previous_payoffs = current_payoffs
            iteration += 1

        # Calculate final weighted mean outcome
        final_outcome = self.calculate_weighted_median()

        return {
            'final_positions': {pid: self.players[pid].position for pid in self.player_ids},
            'final_outcome': final_outcome,
            'converged': iteration < self.max_iterations,
            'iterations': iteration + 1,
            'final_payoffs': current_payoffs,
            'history': self.iteration_history
        }

    def run_monte_carlo(self, n_simulations: int = 100,
                       uncertainty_range: float = 0.1) -> Dict:
        """
        Run Monte Carlo simulation to generate confidence intervals.

        Args:
            n_simulations: Number of simulations to run
            uncertainty_range: Range of uncertainty in input parameters (as fraction)

        Returns:
            Dictionary with mean predictions and confidence intervals
        """
        results = []
        original_players = {}

        # Store original player states
        for pid, player in self.players.items():
            original_players[pid] = Player(
                id=player.id,
                position=player.position,
                salience=player.salience,
                clout=player.clout,
                resolve=player.resolve
            )

        for sim in range(n_simulations):
            # Reset players to original state
            for pid, orig_player in original_players.items():
                self.players[pid] = Player(
                    id=orig_player.id,
                    position=orig_player.position,
                    salience=orig_player.salience,
                    clout=orig_player.clout,
                    resolve=orig_player.resolve
                )

                # Add uncertainty
                player = self.players[pid]
                player.salience *= np.random.normal(1.0, uncertainty_range)
                player.clout *= np.random.normal(1.0, uncertainty_range)
                player.resolve *= np.random.normal(1.0, uncertainty_range)

                # Ensure bounds
                player.salience = max(0, min(100, player.salience))
                player.clout = max(0.01, player.clout)
                player.resolve = max(0, min(100, player.resolve))

            # Reset history
            self.iteration_history = []

            # Run simulation
            result = self.run_simulation(verbose=False)
            results.append(result['final_outcome'])

        # Calculate statistics
        results = np.array(results)

        return {
            'mean_prediction': np.mean(results),
            'median_prediction': np.median(results),
            'std_prediction': np.std(results),
            'ci_95_lower': np.percentile(results, 2.5),
            'ci_95_upper': np.percentile(results, 97.5),
            'ci_90_lower': np.percentile(results, 5),
            'ci_90_upper': np.percentile(results, 95),
            'all_results': results
        }

    def plot_convergence(self, figsize: Tuple[int, int] = (12, 8)):
        """Plot the convergence of player positions over iterations"""
        if not self.iteration_history:
            print("No simulation history available. Run simulation first.")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # Plot positions over time
        iterations = [h['iteration'] for h in self.iteration_history]
        for pid in self.player_ids:
            positions = [h['positions'][pid] for h in self.iteration_history]
            ax1.plot(iterations, positions, marker='o', label=f'Player {pid}')

        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Position')
        ax1.set_title('Player Position Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot payoffs over time
        for pid in self.player_ids:
            payoffs = [h['payoffs'][pid] for h in self.iteration_history]
            ax2.plot(iterations, payoffs, marker='s', label=f'Player {pid}')

        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Payoff')
        ax2.set_title('Player Payoff Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

# Example usage and testing
def create_example_scenario():
    """Create an example policy scenario for testing"""
    players = [
        Player(id="A", position=20, salience=80, clout=1.5, resolve=70),
        Player(id="B", position=60, salience=90, clout=1.2, resolve=60),
        Player(id="C", position=40, salience=70, clout=1.0, resolve=80),
        Player(id="D", position=80, salience=60, clout=0.8, resolve=50),
    ]
    return players

if __name__ == "__main__":
    # Example usage
    print("Bueno de Mesquita Policy Prediction Model - Example Run")
    print("=" * 60)

    # Create example scenario
    players = create_example_scenario()

    # Initialize model
    model = BuenoMesquitaModel(players)

    print("Initial Positions:")
    for player in players:
        print(f"  Player {player.id}: Position={player.position}, "
              f"Salience={player.salience}, Clout={player.clout}, Resolve={player.resolve}")

    # Run single simulation
    print("\nRunning simulation...")
    result = model.run_simulation(verbose=True)

    print(f"\nFinal Results:")
    print(f"  Predicted Outcome: {result['final_outcome']:.1f}")
    print(f"  Converged: {result['converged']}")
    print(f"  Iterations: {result['iterations']}")

    print(f"\nFinal Positions:")
    for pid, pos in result['final_positions'].items():
        print(f"  Player {pid}: {pos:.1f}")

    # Run Monte Carlo simulation
    print(f"\nRunning Monte Carlo simulation (100 runs)...")
    mc_results = model.run_monte_carlo(n_simulations=100)

    print(f"Monte Carlo Results:")
    print(f"  Mean Prediction: {mc_results['mean_prediction']:.1f}")
    print(f"  95% Confidence Interval: [{mc_results['ci_95_lower']:.1f}, {mc_results['ci_95_upper']:.1f}]")
    print(f"  Standard Deviation: {mc_results['std_prediction']:.1f}")