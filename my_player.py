from math import inf

from seahorse.game.action import Action

from actions_quoridor import Orientation, Wall
from game_state_quoridor import GameStateQuoridor
from player_quoridor import PlayerQuoridor
import time

class SearchTimeout(Exception):
    pass

class MyPlayer(PlayerQuoridor):

    SEARCH_DEPTH = 3
    BEAM_WIDTH = 6
    TARGETED_WALL_LIMIT = 5
    WIN_SCORE = 100_000
    WALL_MIN_ADVANTAGE = 2
    SAFE_RACE_LEAD = 2
    WALL_SELF_COST_WEIGHT = 1.5
    WALL_TURN_COST = 0.5
    WALL_RESERVE_EARLY = 3.0
    WALL_RESERVE_LATE = 0.5

    def __init__(self, piece_type: str, goal_row: int=0, name: str = "bob", *args, **kwargs) -> None:
        """
        Initialize the PlayerQuoridor instance.

        Args:
            piece_type (str): Type of the player's game piece
            goal_row (int): The row the player wants to reach
            name (str, optional): Name of the player (default is "bob")
        """
        super().__init__(piece_type, goal_row, name, *args, **kwargs)
        self._distance_cache = {}
    
    def _check_time(self):
        if time.monotonic() >= self._deadline:
            raise SearchTimeout()
        

    def compute_action(self, current_state: GameStateQuoridor, remaining_time: float = 15*60, **kwargs) -> Action:
        """
        minimax avec heuristique :
        le chemin le plsu court de l<adversaire moins le notre, on regarde deux tours dans le futur et
        on choisi
        (https://www.youtube.com/watch?v=zp3VMe0Jpf8)
        """
        self._distance_cache.clear()
        
        start = time.monotonic()
        budget = min(1.0, max(0.05, remaining_time * 0.03))
        self._deadline = start + budget

        actions = list(current_state.generate_possible_stateless_actions())
        if not actions:
            raise RuntimeError("No legal action available.")
        
        best_action = actions[0]

        simple_move = self._get_fast_forward_move(current_state)
        if simple_move is not None:
            return simple_move

        try:
            candidates = self._ordered_successors(current_state, actions, True)
            if not candidates:
                return best_action
            
            best_action = candidates[0][0]
            for depth in range(1, self.SEARCH_DEPTH + 1):
                iteration_best = best_action
                iteration_value = -inf
                alpha, beta = -inf, inf
            

                for action, child in candidates:
                    self._check_time()
                    value = self._minimax(child, depth - 1, alpha, beta)
                    if value > iteration_value:
                        iteration_value = value
                        iteration_best = action
                    alpha = max(alpha, iteration_value)
                    
                best_action = iteration_best
                
        except SearchTimeout:
            pass
        

        return best_action

    def _minimax(self, state: GameStateQuoridor, depth: int,
                 alpha: float, beta: float) -> float:
        
        self._check_time()

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

        my_distance = self._distance(state, own_player)
        opponent_distance = self._distance(state, opponent)
        safe_race = (
            my_distance == 1
            or my_distance + self.SAFE_RACE_LEAD <= opponent_distance
        )

        # mur dwfensif
        if not safe_race:
            return None

        if self._players_are_adjacent(my_row, my_col, opponent_row, opponent_col):
            best_move, _ = self._best_move_by_shortest_path(state, move_actions)
            return best_move

        if forward_action is not None:
            child = state.apply_action(forward_action)
            child_player = next(p for p in child.players if p.id == self.id)

            if self._distance(child, child_player) < my_distance:
                return forward_action

        best_detour, best_detour_distance = self._best_move_by_shortest_path(
            state, move_actions
        )
        if (
            best_detour is not None
            and best_detour_distance + self.SAFE_RACE_LEAD <= opponent_distance
        ):
            return best_detour

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

            distance = self._distance(child, child_player)

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

        for action, child in self._targeted_actions(state, actions):
            self._check_time()
            score = self._evaluate(child)
            successors.append((action, child, score))       
    
        successors.sort(key=lambda item: item[2], reverse=maximizing)
        return [(action, child) for action, child, _ in successors[:self.BEAM_WIDTH]]

    def _targeted_actions(self, state: GameStateQuoridor, actions):
        self._check_time()
        active_player = state.active_player
        target_player = next(
            player for player in state.players if player.id != active_player.id
        )
        target_edges = self._shortest_path_edges(state, target_player)
        own_distance = self._distance(state, active_player)
        opponent_distance = self._distance(state, target_player)

        moves = []
        walls = []
        for action in actions:
            self._check_time()
            if not self._is_wall(action):
                moves.append((action, state.apply_action(action)))
            elif self._wall_blocks_path(action, state, target_edges):
                walls.append(action)

        ranked_walls = []
        for action in walls:
            self._check_time()
            child = state.apply_action(action)
            opponent_delay = self._distance(child, target_player) - opponent_distance
            if opponent_delay <= 0:
                continue

            own_delay = self._distance(child, active_player) - own_distance
            benefit = opponent_delay - self.WALL_SELF_COST_WEIGHT * own_delay - self.WALL_TURN_COST
            if benefit > 0:
                ranked_walls.append((benefit, action, child))
                
        ranked_walls.sort(key=lambda item: item[0], reverse=True)

        return moves +  [(action, child) for _, action, child in ranked_walls[:self.TARGETED_WALL_LIMIT]]

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
        
    def _is_wall(self, action: Action) -> bool:
        return action.data["type"] in ("horizontal", "vertical")
    
    def _distance(self, state: GameStateQuoridor, player) -> int:
        key = (id(state), player.id)
        cached = self._distance_cache.get(key)

        if cached is not None and cached[0] is state:
            return cached[1]

        distance = state._shortest_path(player)
        self._distance_cache[key] = (state, distance)
        return distance


    def _evaluate(self, state: GameStateQuoridor) -> float:
        own_player = next(player for player in state.players if player.id == self.id)
        opponent = next(player for player in state.players if player.id != self.id)

        if state.scores.get(self.id) == 1.0:
            return self.WIN_SCORE
        if state.scores.get(opponent.id) == 1.0:
            return -self.WIN_SCORE

        own_distance = self._distance(state, own_player)
        opponent_distance = self._distance(state, opponent)
        
        own_walls = state.rep.remaining_walls[self.id]
        opponent_walls = state.rep.remaining_walls[opponent.id]

        board_size = state.rep.dimension
        progress = 1 - min(own_distance, opponent_distance) / max(1, board_size - 1)
        progress = max(0.0, min(1.0, progress))

        wall_weight = (
            self.WALL_RESERVE_EARLY * (1 - progress)
            + self.WALL_RESERVE_LATE * progress
        )

        wall_bonus = wall_weight * (own_walls - opponent_walls)
        
        lead = opponent_distance - own_distance
        score = 12 * lead
        
        score += 1.5 * lead * abs(lead)
        
        proximity_bonus = {1: 1200, 2: 350, 3: 150, 4: 60}
        score += proximity_bonus.get(own_distance, 0)
        score -= proximity_bonus.get(opponent_distance, 0)
        
        if state.active_player.id == self.id:
            if own_distance == 1:
                score += 4000
        else:
            if opponent_distance == 1:
                score -= 4500
                
        score += wall_bonus
        return score
