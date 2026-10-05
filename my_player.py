from math import inf

from seahorse.game.action import Action

from actions_quoridor import Orientation, Wall
from game_state_quoridor import GameStateQuoridor
from player_quoridor import PlayerQuoridor

class MyPlayer(PlayerQuoridor):

    SEARCH_DEPTH = 4
    BEAM_WIDTH = 16
    TARGETED_WALL_LIMIT = 12
    WIN_SCORE = 100_000
    WALL_MIN_ADVANTAGE = 2

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
        own_player = next(player for player in state.players if player.id == self.id)
        opponent = next(player for player in state.players if player.id != self.id)

        my_row, my_col = state.rep.pawn_positions[self.id]
        opponent_row, opponent_col = state.rep.pawn_positions[opponent.id]

        my_goal = own_player.get_goal_row()

        move_actions = state._legal_moves()

        if not move_actions:
            return None

        forward_action = self._find_forward_action(move_actions, my_row, my_col, my_goal)

        if self._players_are_adjacent(my_row, my_col, opponent_row, opponent_col):
            best_move, _ = self._best_move_by_shortest_path(state, move_actions)
            return best_move

        my_distance = state._shortest_path(own_player)
        opponent_distance = state._shortest_path(opponent)
        crossed_midpoint = self._has_crossed_midpoint(state, own_player)

        if not crossed_midpoint:
            if my_distance > opponent_distance:
                return None

            if forward_action is not None:
                return forward_action

            best_detour, best_detour_distance = self._best_move_by_shortest_path(state, move_actions)

            opponent_distance = state._shortest_path(opponent)

            if (
                best_detour is not None
                and best_detour_distance <= opponent_distance
            ):
                return best_detour

            return None

        if my_distance <= opponent_distance:
            if forward_action is not None:
                return forward_action

            best_detour, best_detour_distance = self._best_move_by_shortest_path(state, move_actions)

            if (
                best_detour is not None
                and best_detour_distance <= opponent_distance
            ):
                return best_detour

            return None

        return None


    def _has_crossed_midpoint(self, state, player):
        row, _ = state.rep.pawn_positions[player.id]
        goal_row = player.get_goal_row()
        midpoint = state.rep.dimension // 2

        if goal_row == 0:
            return row <= midpoint

        return row >= midpoint


    def _find_forward_action(self, move_actions, row, col, goal_row):
        direction = -1 if goal_row < row else 1
        forward_position = (row + direction, col)

        for action in move_actions:
            if action.data["destination"] == forward_position:
                return action

        return None


    def _players_are_adjacent(self, my_row, my_col, opponent_row, opponent_col):
        return (
            abs(my_row - opponent_row)
            + abs(my_col - opponent_col)
            == 1
        )


    def _best_move_by_shortest_path(self, state, move_actions):
        best_move = None
        best_distance = inf

        for action in move_actions:
            child = state.apply_action(action)

            child_player = next(
                player for player in child.players
                if player.id == self.id
            )

            distance = child._shortest_path(child_player)

            if distance < best_distance:
                best_distance = distance
                best_move = action

        return best_move, best_distance
    
    def _straight_path_clear(self, state, player):
        row, col = state.rep.pawn_positions[player.id]
        goal_row = player.get_goal_row()

        direction = -1 if goal_row < row else 1

        while row != goal_row:
            next_position = (row + direction, col)

            if state._blocked_by_wall((row, col), next_position):
                return False

            row += direction

        return True

    def _ordered_successors(self, state: GameStateQuoridor, actions,
                            maximizing: bool):
        successors = []

        actions = self._targeted_actions(state, actions)
        
        for action in actions:
            child = state.apply_action(action)

            if maximizing and self._is_wall(action):
                if not self._is_beneficial_wall(state, child):
                    continue

            score = self._evaluate(child)
            successors.append((action, child, score))       
    
        successors.sort(key=lambda item: item[2], reverse=maximizing)
        return [(action, child) for action, child, _ in successors[:self.BEAM_WIDTH]]

    def _targeted_actions(self, state: GameStateQuoridor, actions):
        active_player = state.active_player
        target_player = next(
            player for player in state.players if player.id != active_player.id
        )
        target_edges = self._shortest_path_edges(state, target_player)
        target_position = state.rep.pawn_positions[target_player.id]

        moves = []
        walls = []
        for action in actions:
            if not self._is_wall(action):
                moves.append(action)
            elif self._wall_blocks_path(action, state, target_edges):
                walls.append(action)

        walls.sort(
            key=lambda action: self._wall_priority(
                action, state, target_edges, target_position
            ),
            reverse=True,
        )
        return moves + walls[:self.TARGETED_WALL_LIMIT]

    def _shortest_path_edges(self, state: GameStateQuoridor, player):
        start = state.rep.pawn_positions[player.id]
        queue = [start]
        previous = {start: None}
        goal = None

        for position in queue:
            if position[0] == player.get_goal_row():
                goal = position
                break
            for neighbour in state._reachable_neighbours(position):
                if neighbour not in previous:
                    previous[neighbour] = position
                    queue.append(neighbour)

        if goal is None:
            return frozenset()

        edges = set()
        position = goal
        while previous[position] is not None:
            parent = previous[position]
            edges.add(frozenset((parent, position)))
            position = parent
        return frozenset(edges)

    def _wall_blocks_path(self, action: Action, state: GameStateQuoridor,
                          path_edges) -> bool:
        destination = action.data["destination"]
        orientation = (
            Orientation.HORIZONTAL
            if action.data["type"] == "horizontal"
            else Orientation.VERTICAL
        )
        wall = Wall(destination[0], destination[1], orientation)
        return any(
            wall in state._candidate_blocking_walls(*tuple(edge))
            for edge in path_edges
        )

    def _wall_priority(self, action: Action, state: GameStateQuoridor,
                       path_edges, target_position) -> tuple[int, int]:
        destination = action.data["destination"]
        orientation = (
            Orientation.HORIZONTAL
            if action.data["type"] == "horizontal"
            else Orientation.VERTICAL
        )
        wall = Wall(destination[0], destination[1], orientation)
        blocked_edges = [
            edge for edge in path_edges
            if wall in state._candidate_blocking_walls(*tuple(edge))
        ]
        nearest_edge = min(
            min(
                abs(square[0] - target_position[0])
                + abs(square[1] - target_position[1])
                for square in edge
            )
            for edge in blocked_edges
        )
        return len(blocked_edges), -nearest_edge

    def _is_wall(self, action: Action) -> bool:
        return action.data["type"] in ("horizontal", "vertical")

    def _is_beneficial_wall(self, state: GameStateQuoridor,
                            child: GameStateQuoridor) -> bool:
        own_player = next(player for player in state.players if player.id == self.id)
        opponent = next(player for player in state.players if player.id != self.id)

        own_cost = (
            child._shortest_path(own_player)
            - state._shortest_path(own_player)
        )
        opponent_delay = (
            child._shortest_path(opponent)
            - state._shortest_path(opponent)
        )

        if opponent_delay <= 0:
            return False
        if own_cost <= 0:
            return True
        return opponent_delay >= own_cost + self.WALL_MIN_ADVANTAGE

    def _evaluate(self, state: GameStateQuoridor) -> float:
        own_player = next(player for player in state.players if player.id == self.id)
        opponent = next(player for player in state.players if player.id != self.id)

        if state.scores.get(self.id) == 1.0:
            return self.WIN_SCORE
        if state.scores.get(opponent.id) == 1.0:
            return -self.WIN_SCORE

        own_distance = state._shortest_path(own_player)
        opponent_distance = state._shortest_path(opponent)
        
        score = 10 * (opponent_distance - own_distance) 
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
