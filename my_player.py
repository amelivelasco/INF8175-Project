from math import inf
from collections import deque

from seahorse.game.action import Action
from seahorse.game.stateless_action import StatelessAction

from actions_quoridor import Orientation, Wall
from game_state_quoridor import GameStateQuoridor
from player_quoridor import PlayerQuoridor
import time

class SearchTimeout(Exception):
    pass

class MyPlayer(PlayerQuoridor):

    SEARCH_DEPTH = 5
    BEAM_WIDTH = 6
    TARGETED_WALL_LIMIT = 5
    WIN_SCORE = 100_000
    WALL_MIN_ADVANTAGE = 2
    SAFE_RACE_LEAD = 2
    WALL_SELF_COST_WEIGHT = 1.5
    WALL_TURN_COST = 0.5
    WALL_RESERVE_EARLY = 3.0
    WALL_RESERVE_LATE = 0.5
    MAX_MOVE_SECONDS = 1.5
    TIME_RESERVE_SECONDS = 0.25
    TIME_FRACTION = 0.025
    MAX_TRANSPOSITION_ENTRIES = 50_000

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
        self._transposition_table = {}
    
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
        self._transposition_table.clear()
        
        start = time.monotonic()
        
        credit = max(0.0, float(remaining_time))
        budget = min(
            self.MAX_MOVE_SECONDS,
            credit * self.TIME_FRACTION,
            max(0.0, credit - self.TIME_RESERVE_SECONDS)
        )
        
        self._deadline = start + budget

        moves = current_state._legal_moves()
        if not moves:
            raise RuntimeError("No legal pawn move available.")

        best_action = moves[0]

        try:
            self._check_time()
            actions = self._candidate_actions(current_state)

            tactical_move = self._tactical_action(current_state, actions)
            if tactical_move is not None:
                return tactical_move

            self._check_time()
            
            simple_move = self._get_fast_forward_move(current_state)
            if simple_move is not None:
                return simple_move
            
            max_depth, self._beam_width = self._search_settings(current_state, credit)
            max_depth = min(max_depth, self.SEARCH_DEPTH)

            candidates = self._ordered_successors(current_state, actions, True)
            if not candidates:
                return best_action

            best_action = candidates[0][0]
            
            for depth in range(1, max_depth + 1):
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
        alpha_start, beta_start = alpha, beta
        key = self._transposition_key(state, depth)
        cached = self._transposition_table.get(key)

        if cached is not None:
            value, bound = cached
            if bound == "exact":
                return value
            if bound == "lower":
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)
            if alpha >= beta:
                return value

        if depth == 0 or state.is_done():
            value = self._evaluate(state)
            self._store_transposition(key, value, alpha_start, beta_start)
            return value

        actions = self._candidate_actions(state)
        
        if not actions:
            value = self._evaluate(state)
            self._store_transposition(key, value, alpha_start, beta_start)
            return value
        
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
            self._store_transposition(key, value, alpha_start, beta_start)
            return value

        value = inf
        for _, child in successors:
            value = min(value, self._minimax(child, depth - 1, alpha, beta))
            beta = min(beta, value)
            if alpha >= beta:
                break
        self._store_transposition(key, value, alpha_start, beta_start)
        return value

    def _transposition_key(self, state: GameStateQuoridor, depth: int):
        rep = state.rep
        return (
            depth,
            state.active_player.id,
            tuple(sorted(rep.pawn_positions.items())),
            rep.walls,
            tuple(sorted(rep.remaining_walls.items())),
            rep.dimension,
        )

    def _store_transposition(self, key, value: float,
                             alpha_start: float, beta_start: float) -> None:
        if len(self._transposition_table) >= self.MAX_TRANSPOSITION_ENTRIES:
            return
        if value <= alpha_start:
            bound = "upper"
        elif value >= beta_start:
            bound = "lower"
        else:
            bound = "exact"
        self._transposition_table[key] = value, bound
    
    def _get_fast_forward_move(self, state):
        own_player = next(player for player in state.players if player.id == self.id)
        opponent = next(player for player in state.players if player.id != self.id)
        
        if state.rep.remaining_walls[opponent.id] != 0:
            return None

        my_row, my_col = state.rep.pawn_positions[self.id]
        opponent_row, opponent_col = state.rep.pawn_positions[opponent.id]

        my_goal = own_player.get_goal_row()

        move_actions = state._legal_moves()

        if not move_actions:
            return None

        forward_action = self._find_forward_action(move_actions, my_row, my_col, my_goal)

        my_distance = self._distance(state, own_player)
        opponent_distance = self._distance(state, opponent)
        
        if my_distance + 3 > opponent_distance:
            return None
        
        if self._players_are_adjacent(my_row, my_col, opponent_row, opponent_col):
            return None
        
        if not self._straight_path_clear(state, own_player):
            return None
        
        if forward_action is None:
            return None

        child = state.apply_action(forward_action)
        child_player = next(p for p in child.players if p.id == self.id)

        if self._distance(child, child_player) != my_distance - 1:
            return None
        
        for reply in child._legal_moves():
            self._check_time()
            reply_state = child.apply_action(reply)

            if reply_state.scores.get(opponent.id) == 1.0:
                return None

            if self._distance(reply_state, opponent) <= my_distance - 1:
                return None
            
        return forward_action


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
        self._check_time()
        active = state.active_player
        opponent = next(p for p in state.players if p.id != active.id)

        # Winning moves always take priority.
        for action in actions:
            if action.data["type"] == "move" and action.data["destination"][0] == active.get_goal_row():
                return [(action, state.apply_action(action))]

        opponent_row = state.rep.pawn_positions[opponent.id][0]
        urgent = abs(opponent_row - opponent.get_goal_row()) <= 2

        if urgent:
            candidates = ((a, state.apply_action(a)) for a in actions)
        else:
            candidates = self._targeted_actions(state, actions)

        safe = []
        unsafe = []

        for action, child in candidates:
            self._check_time()

            if urgent and any(
                move.data["destination"][0] == opponent.get_goal_row()
                for move in child._legal_moves()
            ):
                unsafe.append((action, child))
            else:
                safe.append((action, child))

        selected = safe if safe else unsafe

        ranked = []
        for action, child in selected:
            self._check_time()
            ranked.append((action, child, self._evaluate(child)))

        ranked.sort(key=lambda item: item[2], reverse=maximizing)
        width = getattr(self, "_beam_width", self.BEAM_WIDTH)

        return [(action, child) for action, child, _ in ranked[:width]]


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
        goals = [
            (player.get_goal_row(), col)
            for col in range(state.rep.dimension)
        ]

        def distances_from(sources):
            distances = {source: 0 for source in sources}
            queue = deque(sources)
            while queue:
                self._check_time()
                position = queue.popleft()
                for neighbour in state._reachable_neighbours(position):
                    if neighbour not in distances:
                        distances[neighbour] = distances[position] + 1
                        queue.append(neighbour)
            return distances

        from_start = distances_from([start])
        reachable_goals = [goal for goal in goals if goal in from_start]
        if not reachable_goals:
            return frozenset()
        shortest_distance = min(from_start[goal] for goal in reachable_goals)
        to_goal = distances_from(goals)
        edges = set()

        for position, distance in from_start.items():
            self._check_time()
            for neighbour in state._reachable_neighbours(position):
                if (
                    distance + 1 + to_goal.get(neighbour, inf)
                    == shortest_distance
                ):
                    edges.add(frozenset((position, neighbour)))

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
    
    
    def _tactical_action(self, state: GameStateQuoridor, actions):
        """Take an immediate win or prevent an opponent win on the next turn."""
        own_player = next(p for p in state.players if p.id == self.id)
        opponent = next(p for p in state.players if p.id != self.id)

        # Always take a legal winning pawn move immediately.
        for action in actions:
            if (action.data["type"] == "move"
                    and action.data["destination"][0] == own_player.get_goal_row()):
                return action

        # A pawn more than two rows from the goal cannot finish in one move,
        # even when a jump over the other pawn is possible.
        opponent_row = state.rep.pawn_positions[opponent.id][0]
        if abs(opponent_row - opponent.get_goal_row()) > 2:
            return None

        best_defense = None
        best_score = -inf
        for action in actions:
            # We must leave some time for a valid fallback action if the budget is nearly gone.
            if time.monotonic() >= self._deadline - 0.10:
                break

            child = state.apply_action(action)

            # After our action, the opponent becomes the active player.
            opponent_can_win = any(
                move.data["destination"][0] == opponent.get_goal_row()
                for move in child._legal_moves()
            )
            if opponent_can_win:
                continue

            score = self._evaluate(child)
            if score > best_score:
                best_score = score
                best_defense = action

        return best_defense
    
    def _search_settings(self, state: GameStateQuoridor, remaining_time: float):
        """Choose search depth and beam width based on complexity."""
        walls = sum(state.rep.remaining_walls.values())

        if remaining_time < 10:
            return 2, 4
        if walls >= 12:
            return 3, 5
        if walls >= 5:
            return 4, 6
        return 5, 8
    
    
    def _candidate_actions(self, state: GameStateQuoridor):
        """Generate pawn moves and promising legal wall placements."""
        self._check_time()

        moves = state._legal_moves()
        active = state.active_player
        opponent = next(p for p in state.players if p.id != active.id)

        if state.rep.remaining_walls[active.id] == 0:
            return moves

        opponent_row = state.rep.pawn_positions[opponent.id][0]

        # Search all legal walls when the opponent is close to winning.
        if abs(opponent_row - opponent.get_goal_row()) <= 2:
            return moves + state._legal_walls()

        # Find walls that intersect the opponent's shortest path.
        edges = self._shortest_path_edges(state, opponent)
        wall_set = set()

        for edge in edges:
            self._check_time()
            wall_set.update(state._candidate_blocking_walls(*tuple(edge)))

        walls = []
        for wall in wall_set:
            self._check_time()

            if state._is_wall_legal(wall):
                walls.append(StatelessAction({
                    "type": "horizontal" if wall.orientation == Orientation.HORIZONTAL else "vertical",
                    "destination": (wall.row, wall.col)
                }))

        return moves + walls



