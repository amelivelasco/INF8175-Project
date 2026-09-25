from math import inf

from seahorse.game.action import Action

from game_state_quoridor import GameStateQuoridor
from player_quoridor import PlayerQuoridor

class MyPlayer(PlayerQuoridor):

    SEARCH_DEPTH = 2
    BEAM_WIDTH = 18
    WIN_SCORE = 10_000

    def __init__(self, piece_type: str, goal_row: int=0, name: str = "bob", *args, **kwargs) -> None:
        """
        Initialize the PlayerQuoridor instance.

        Args:
            piece_type (str): Type of the player's game piece
            goal_row (int): The row the player wants to reach
            name (str, optional): Name of the player (default is "bob")
        """
        super().__init__(piece_type, goal_row, name, *args, **kwargs)

    def compute_action(self, current_state: GameStateQuoridor, remaining_time: float = 15*60, **kwargs) -> Action:
        """
        minimax avec heuristique :
        le chemin le plsu court de l<adversaire moins le notre, on regarde deux tours dans le futur et
        on choisi
        (https://www.youtube.com/watch?v=zp3VMe0Jpf8)
        """
        actions = list(current_state.generate_possible_stateless_actions())
        if not actions:
            raise RuntimeError("No legal action available.")

        candidates = self._ordered_successors(current_state, actions, True)
        best_action = candidates[0][0]
        best_value = -inf

        for action, child in candidates:
            value = self._minimax(child, self.SEARCH_DEPTH - 1, -inf, inf)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def _minimax(self, state: GameStateQuoridor, depth: int,
                 alpha: float, beta: float) -> float:
        if depth == 0 or state.is_done():
            return self._evaluate(state)

        maximizing = state.active_player.id == self.id
        successors = self._ordered_successors(
            state,
            list(state.generate_possible_stateless_actions()),
            maximizing,
        )

        if maximizing:
            value = -inf
            for _, child in successors:
                value = max(value, self._minimax(child, depth - 1, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value

        value = inf
        for _, child in successors:
            value = min(value, self._minimax(child, depth - 1, alpha, beta))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

    def _ordered_successors(self, state: GameStateQuoridor, actions,
                            maximizing: bool):
        successors = [(action, state.apply_action(action)) for action in actions]
        successors.sort(key=lambda item: self._evaluate(item[1]), reverse=maximizing)
        return successors[:self.BEAM_WIDTH]

    def _evaluate(self, state: GameStateQuoridor) -> float:
        own_player = next(player for player in state.players if player.id == self.id)
        opponent = next(player for player in state.players if player.id != self.id)

        if state.scores.get(self.id) == 1.0:
            return self.WIN_SCORE
        if state.scores.get(opponent.id) == 1.0:
            return -self.WIN_SCORE

        own_distance = state._shortest_path(own_player)
        opponent_distance = state._shortest_path(opponent)
        return opponent_distance - own_distance
