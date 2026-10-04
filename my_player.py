from math import inf

from seahorse.game.action import Action

from game_state_quoridor import GameStateQuoridor
from player_quoridor import PlayerQuoridor

class MyPlayer(PlayerQuoridor):

    SEARCH_DEPTH = 3
    BEAM_WIDTH = 24
    WIN_SCORE = 100_000

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
        
        simple_move = self._get_fast_forward_move(current_state)
        if simple_move is not None:
            return simple_move

        actions = list(current_state.generate_possible_stateless_actions())
        if not actions:
            raise RuntimeError("No legal action available.")
        
        candidates = self._ordered_successors(current_state, actions, True)
        best_action = candidates[0][0]
        best_value = -inf
        
        alpha = -inf
        beta = inf

        for action, child in candidates:
            value = self._minimax(child, self.SEARCH_DEPTH - 1, alpha, beta)
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value)

        return best_action

    def _minimax(self, state: GameStateQuoridor, depth: int,
                 alpha: float, beta: float) -> float:
        if depth == 0 or state.is_done():
            return self._evaluate(state)

        actions = list(state.generate_possible_stateless_actions())
        
        if not actions:
            return self._evaluate(state)
        
        maximizing = state.active_player.id == self.id
        
        successors = self._ordered_successors(
            state,
            actions,
            maximizing,
        )

        if maximizing:
            value = -inf
            for _, child in successors:
                value = max(value, self._minimax(child, depth - 1, alpha, beta))
                
                alpha = max(alpha, value) # Propagate the alpha value up the tree
                
                if alpha >= beta:
                    break
            return value

        value = inf
        for _, child in successors:
            value = min(value, self._minimax(child, depth - 1, alpha, beta))
            beta = min(beta, value)
            
            beta = min(beta, value) # Propagate the beta value up the tree
            
            if alpha >= beta:
                break
        return value
    
    def _get_fast_forward_move(self, state):
        own_player = next(
            player for player in state.players
            if player.id == self.id
        )

        opponent = next(
            player for player in state.players
            if player.id != self.id
        )

        my_position = state.rep.pawn_positions[self.id]
        opponent_position = state.rep.pawn_positions[opponent.id]

        my_row, my_col = my_position
        opponent_row, opponent_col = opponent_position

        # If opponent is close to winning, use minimax so we can consider walls.
        opponent_distance = state._shortest_path(opponent)

        if opponent_distance <= 4:
            return None

        # Only generate pawn movement actions.
        # This avoids the expensive wall-generation code.
        move_actions = state._legal_moves()

        if not move_actions:
            return None

        # Direction toward our target row.
        goal_row = own_player.get_goal_row()

        if goal_row < my_row:
            forward_row = my_row - 1
        else:
            forward_row = my_row + 1

        forward_position = (forward_row, my_col)

        for action in move_actions:
            if action.data["destination"] == forward_position:
                return action

        if abs(my_row - opponent_row) + abs(my_col - opponent_col) == 1:
            best_move = None
            best_goal_distance = inf

            for action in move_actions:
                row, col = action.data["destination"]
                goal_distance = abs(row - goal_row)

                if goal_distance < best_goal_distance:
                    best_goal_distance = goal_distance
                    best_move = action

            if best_move is not None:
                return best_move

        return None

    def _ordered_successors(self, state: GameStateQuoridor, actions,
                            maximizing: bool):
        successors = []
        
        for action in actions:
            child = state.apply_action(action)
            score = self._evaluate(child)
            successors.append((action, child, score))       
    
        successors.sort(key=lambda item: item[2], reverse=maximizing)
        return [(action, child) for action, child, _ in successors[:self.BEAM_WIDTH]]

    def _evaluate(self, state: GameStateQuoridor) -> float:
        own_player = next(player for player in state.players if player.id == self.id)
        opponent = next(player for player in state.players if player.id != self.id)

        if state.scores.get(self.id) == 1.0:
            return self.WIN_SCORE
        if state.scores.get(opponent.id) == 1.0:
            return -self.WIN_SCORE

        own_distance = state._shortest_path(own_player)
        opponent_distance = state._shortest_path(opponent)
        
        score = 10 * (opponent_distance - own_distance) # Prefer states where our path is shorter than opponent's
        if own_distance <= 1:
            score += 5000
        elif own_distance == 2:
            score += 800
        elif own_distance == 3:
            score += 300
        elif own_distance == 4:
            score += 100
            
        if opponent_distance <= 1:
            score -= 6000
        elif opponent_distance == 2:
            score -= 1200
        elif opponent_distance == 3:
            score -= 400
        elif opponent_distance == 4:
            score -= 150

        lead = opponent_distance - own_distance
        score += 2 * lead * abs(lead)
        
        if state.active_player.id == self.id:
            score += 2
        else:
            score -= 2

        return score
