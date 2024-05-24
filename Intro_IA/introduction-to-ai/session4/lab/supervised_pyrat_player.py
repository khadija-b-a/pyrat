#####################################################################################################################################################
######################################################################## INFO #######################################################################
#####################################################################################################################################################

"""
    This program is an improvement of "random_2".
    Here, we add elements that help us explore better the maze.
    More precisely, we keep a list (in a global variable to be updated at each turn) of cells that have already been visited in the game.
    Then, at each turn, we choose in priority a random move among those that lead us to an unvisited cell.
    If no such move exists, we move randomly using the method in "random_2".
"""

#####################################################################################################################################################
###################################################################### IMPORTS ######################################################################
#####################################################################################################################################################

# Import PyRat
from pyrat import *

import torch
import torch.nn as nn
import torch.nn.functional as F


########
# TODO #    Update this constant with the path to the trained model you want to use
########
TRAINED_MODEL_PATH = "introduction-to-ai\session4\lab"
#################################################################################################################
################################################### FUNCTIONS ###################################################
#################################################################################################################


########
# TODO #   Put here the class of your model as you defined it in Lab2b
######## 

    
def build_state ( maze:             Union[numpy.ndarray, Dict[int, Dict[int, int]]],
                  maze_width:       int,
                  maze_height:      int,
                  name:             str,
                  teams:            Dict[str, List[str]],
                  player_locations: Dict[str, int],
                  cheese:           List[int]
                ) ->                torch.tensor:

    """
        This function builds a state tensor to use as an input for the DQN.
        Here we assume a 2-player game.
        In:
            * maze:             Map of the maze, as data type described by PyRat's "maze_representation" option.
            * maze_width:       Width of the maze in number of cells.
            * maze_height:      Height of the maze in number of cells.
            * name:             Name of the player being trained.
            * teams:            Recap of the teams of players.
            * player_locations: Locations for all players in the game.
            * cheese:           List of available pieces of cheese in the maze.
        Out:
            * state: Tensor representing the state of the game.
    """
    
    # Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Function to return an array with the player at the center
    def _center_maze (location):
        channel = torch.zeros(maze_height * 2 - 1, maze_width * 2 - 1, device=device)
        location_row, location_col = location // maze_width, location % maze_width
        for c in cheese:
            c_row, c_col = c // maze_width, c % maze_width
            channel[maze_height - 1 - location_row + c_row, maze_width - 1 - location_col + c_col] = 1
        return channel

    # A channel centered on the player
    player_channel = _center_maze(player_locations[name])
    return player_channel

#####################################################################################################################################################
##################################################### EXECUTED ONCE AT THE BEGINNING OF THE GAME ####################################################
#####################################################################################################################################################

def preprocessing ( maze:             Union[numpy.ndarray, Dict[int, Dict[int, int]]],
                    maze_width:       int,
                    maze_height:      int,
                    name:             str,
                    teams:            Dict[str, List[str]],
                    player_locations: Dict[str, int],
                    cheese:           List[int],
                    possible_actions: List[str],
                    memory:           threading.local
                  ) ->                None:
    

    """
        This function is called once at the beginning of the game.
        It is typically given more time than the turn function, to perform complex computations.
        Store the results of these computations in the provided memory to reuse them later during turns.
        To do so, you can crete entries in the memory dictionary as memory.my_key = my_value.
        In:
            * maze:             Map of the maze, as data type described by PyRat's "maze_representation" option.
            * maze_width:       Width of the maze in number of cells.
            * maze_height:      Height of the maze in number of cells.
            * name:             Name of the player controlled by this function.
            * teams:            Recap of the teams of players.
            * player_locations: Locations for all players in the game.
            * cheese:           List of available pieces of cheese in the maze.
            * possible_actions: List of possible actions.
            * memory:           Local memory to share information between preprocessing, turn and postprocessing.
        Out:
            * None.
    """

    ########
    # TODO #   generate an example of data using the function build_state to get correct input size
    ######## 

    data =build_state(maze,maze_width,maze_height,name, teams,player_locations, cheese)
    

    
    
    # Create an instance of your model with as argument the in_features size   
     
    memory.model = Net(data.shape[0]*data.shape[1])

     ########
    # TODO #  
    ########
    #Load the model's state dictionary from the saved file and put it in eval mode 
    #(the model will be only used for inference, e.g. generate an action given a game configuration)



#####################################################################################################################################################
######################################################### EXECUTED AT EACH TURN OF THE GAME #########################################################
#####################################################################################################################################################

def turn ( maze:             Union[numpy.ndarray, Dict[int, Dict[int, int]]],
           maze_width:       int,
           maze_height:      int,
           name:             str,
           teams:            Dict[str, List[str]],
           player_locations: Dict[str, int],
           player_scores:    Dict[str, float],
           player_muds:      Dict[str, Dict[str, Union[None, int]]],
           cheese:           List[int],
           possible_actions: List[str],
           memory:           threading.local
         ) ->                str:

    """
        This function is called at every turn of the game and should return an action within the set of possible actions.
        You can access the memory you stored during the preprocessing function by doing memory.my_key.
        You can also update the existing memory with new information, or create new entries as memory.my_key = my_value.
        In:
            * maze:             Map of the maze, as data type described by PyRat's "maze_representation" option.
            * maze_width:       Width of the maze in number of cells.
            * maze_height:      Height of the maze in number of cells.
            * name:             Name of the player controlled by this function.
            * teams:            Recap of the teams of players.
            * player_locations: Locations for all players in the game.
            * player_scores:    Scores for all players in the game.
            * player_muds:      Indicates which player is currently crossing mud.
            * cheese:           List of available pieces of cheese in the maze.
            * possible_actions: List of possible actions.
            * memory:           Local memory to share information between preprocessing, turn and postprocessing.
        Out:
            * action: One of the possible actions, as given in possible_actions.
    """
    data = build_state(maze,maze_width,maze_height,name, teams,player_locations, cheese)
    input_flatten = torch.flatten(data)
    output = memory.model(input_flatten)
    action_index = torch.argmax(output)
    action = possible_actions[action_index]
    
    
    return action

#####################################################################################################################################################
######################################################################## GO! ########################################################################
#####################################################################################################################################################

if __name__ == "__main__":

    # Map the function to the character
    players = [{"name": "Supervised Player", "skin": "default", "preprocessing_function": preprocessing, "turn_function": turn}]

    # Customize the game elements
    config = {"maze_width": 5,
              "maze_height": 7,
              "mud_percentage": 0.0,
              "cell_percentage": 100.0,
              "wall_percentage": 0.0,
              "nb_cheese": 4,
              "trace_length": 1000}

    # Start the game
    game = PyRat(players, **config)
    stats = game.start()

    # Show statistics
    print(stats)

#####################################################################################################################################################
#####################################################################################################################################################