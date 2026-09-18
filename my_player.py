from player_quoridor import PlayerQuoridor
from seahorse.game.action import Action
from game_state_quoridor import GameStateQuoridor
from seahorse.utils.custom_exceptions import MethodNotImplementedError

class MyPlayer(PlayerQuoridor):
    """
    Player class for Quoridor game

    Attributes:
        piece_type (str): piece type of the player
    """

    def __init__(self, piece_type: str, goal_row: int=0, name: str = "bob", *args, **kwargs) -> None:
        """
        Initialize the PlayerQuoridor instance.

        Args:
            piece_type (str): Type of the player's game piece
            goal_row (int): The row the player wants to reach
            name (str, optional): Name of the player (default is "bob")
        """
        super().__init__(piece_type, goal_row, name)

    def compute_action(self, current_state: GameStateQuoridor, remaining_time: float = 15*60, **kwargs) -> Action:
        """
        Use the minimax algorithm to choose the best action based on the heuristic evaluation of game states.

        Args:
            current_state (GameStateQuoridor): The current game state.

        Returns:
            Action: The best action as determined by minimax.
        """

        #TODO
        
        value, action = self.max_value(current_state, float("-inf"), float("inf"), 3)
        
        
        # raise MethodNotImplementedError()
    
        return action  
    
    
    def evaluate(self, state: GameStateQuoridor) -> float:
        if state.is_done():
            if state.scores[self.id] > 0.5:
                return float("inf")
            else:
                return float("-inf")
            
        me = None
        opponent = None
        
        for player in state.players:
            if player.id == self.id:
                me = player
            else:
                opponent = player 
        
        my_distance = state._shortest_path(me)
        opponent_distance = state._shortest_path(opponent)
        return opponent_distance - my_distance
    
    def max_value(self, state: GameStateQuoridor, alpha: float, beta: float, depth):
        if depth == 0 or state.is_done():
            return self.evaluate(state), None
        
        best_value = float("-inf") # v*
        best_action = None # m*
        
        actions = list(state._legal_moves())
        
        for action in actions:
            next_state = state.apply_action(action) # s' = transition(state,action)
            
            value, _ = self.min_value(next_state, alpha, beta, depth - 1)
            
            if best_action == None or value > best_value:
                best_value = value
                best_action = action
                alpha = max(alpha, best_value)
                
            if best_value >= beta:
                return best_value, best_action

        return best_value, best_action
    
    def min_value(self, state: GameStateQuoridor, alpha: float, beta: float, depth):
        if depth == 0 or state.is_done():
            return self.evaluate(state), None
        
        best_value = float("inf")
        best_action = None
        
        actions = list(state._legal_moves())
        
        for action in actions:
            next_state = state.apply_action(action)
            
            value, _ = self.max_value(next_state, alpha, beta, depth - 1)
            
            if best_action == None or value < best_value:
                best_value = value
                best_action = action
                beta = min(beta, best_value)
            if best_value <= alpha: 
                return best_value, best_action
                
        return best_value, best_action