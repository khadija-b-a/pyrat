#####################################################################################################################################################
######################################################################## INFO #######################################################################
#####################################################################################################################################################

"""
    This program aims at training the rat to beat a greedy opponent.
    To do so, we use reinforcement learning, and more precisely deep Q learning (DQN).
    If you set the TRAIN_MODEL constant to True, the rat will be training.
    Otherwise it will just try to get the cheese using what it has learned.

    In this lab, we will work on a full maze with no mud.
    The settings have been set to a 10x10 maze with 15 pieces of cheese.
    You can change these settings if you want.

    This file already provides you with a functional RL algorithm.
    However, it is not very efficient, and you will have to improve it.
    Before you do so, you should understand how it works.

    The file makes use of Weights and Biases (WandB) to monitor the training.
    You can disable its use by setting the USE_WANDB constant to False.
    If you want to use it, you will have to create an account on https://wandb.ai/ and get an API key.
    Then, you will have to store that key in a file named "wandb.key" and set the WANDB_KEY_PATH variable accordingly (by defaultit is set to the same directory as this file).

    Here is your mission:
        1 - Have a look at the code and make sure you understand it.
            Also, make sure the code runs for training (TRAIN_MODEL=True) and playing (TRAIN_MODEL=False).
            If not, you may have to install some packages or change some paths.
        2 - Improve the RL algorithm.
            There are multiple ways to do so, and you are free to choose the ones you want.
            Some ideas:
                * Improve the data representation.
                * Change the reward function.
                * Improve the exploration strategy (some keywords you may be looking for are "epsilon greedy" or "softmax policy" for instance).
                * Change the hyperparameters of the algorithm (discount factor, batch size, etc.).
                * Improve the model architecture.
                * Change the optimizer or its parameters.
                * Change the loss function.
                * Etc.
"""

#####################################################################################################################################################
###################################################################### IMPORTS ######################################################################
#####################################################################################################################################################

# Import PyRat
from pyrat import *

# External imports 
import torch
import random
import numpy
import shutil
import tqdm
# import wandb
import sys
import os

# Previously developed functions
import sys
import os
lab_commons_path = os.path.join(os.getcwd(), "..", "..")
if lab_commons_path not in sys.path:
    sys.path.append(lab_commons_path)

import lab_commons.AI.greedy as opponent
from lab_commons.utils import get_opponent_name

#####################################################################################################################################################
############################################################### CONSTANTS & VARIABLES ###############################################################
#####################################################################################################################################################

"""
    Paths where to store the stuff that must be shared from a game to the other.
"""

OUTPUT_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), "RL")
MODEL_FILE_NAME = os.path.join(OUTPUT_DIRECTORY, "RL_model.pt")
OPTIMIZER_FILE_NAME = os.path.join(OUTPUT_DIRECTORY, "RL_optimizer.pt")
EXPERIENCE_FILE_NAME = os.path.join(OUTPUT_DIRECTORY, "RL_experience.pt")

#####################################################################################################################################################

"""
    Indicates if we are in train or test mode.
"""

TRAIN_MODEL = True
RESET_TRAINING = True

#####################################################################################################################################################

"""
    RL algorithm parameters.
"""

MAX_EXPERIENCE_SIZE = 1000
EXPERIENCE_MIN_SIZE_FOR_TRAINING = 1000
NB_BATCHES = 32
BATCH_SIZE = 64
DISCOUNT_FACTOR = 0.9

#####################################################################################################################################################

"""
    Constants for epsilon-greedy strategy.
"""
EPSILON_START = 0.6
EPSILON_END = 0.0
EPSILON_DECAY = 0.9998848773724686
epsilon_at_start = EPSILON_START

#####################################################################################################################################################

"""
    Parameters of the optimizer used to train the model.
"""

LEARNING_RATE = 0.1

#####################################################################################################################################################

"""
    Number of PyRat games from which to learn.
"""

NB_EPISODES = 1000

#####################################################################################################################################################

"""
    Wandb parameters.
"""

USE_WANDB = False # True # It is a good practice to use WandB to monitor the training. You should try to use it at some point.
WANDB_KEY_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "wandb.key")

#####################################################################################################################################################
############################################################### DEEP Q LEARNING MODEL ###############################################################
#####################################################################################################################################################

class DQN (torch.nn.Module):

    #############################################################################################################################################
    #                                                                CONSTRUCTOR                                                                #
    #############################################################################################################################################

    def __init__ ( self:              Self,
                   data_shape:        Tuple[int, ...],
                   actions_dimension: int
                 ) ->                 Self:

        """
            This function is the constructor of the class.
            In:
                * self:              Reference to the current object.
                * data_shape:        Shape of the input data.
                * actions_dimension: Number of possible actions.
            Out:
                * self: Reference to the current object.
        """

        # Inherit from parent class
        super(DQN, self).__init__()

        # Check if a GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define the layers
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(data_shape[0], 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten()
        )
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(64 * data_shape[1] * data_shape[2], 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, actions_dimension)
        )
        
    #############################################################################################################################################
    #                                                               PUBLIC METHODS                                                              #
    #############################################################################################################################################

    def forward ( self: Self,
                  x:    torch.tensor
                ) ->    torch.tensor:

        """
            This function performs a forward pass of the data through the model.
            In:
                * self: Reference to the current object.
                * x:    Input data.
            Out:
                * y: Output data.
        """

        # Forward pass
        x = x.float().to(self.device)
        y = self.conv_layers(x)
        z = self.fc_layers(y)
        return z

#####################################################################################################################################################
##################################################################### FUNCTIONS #####################################################################
#####################################################################################################################################################

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
        return channel.unsqueeze(0)

    # A channel centered on the player
    player_channel = _center_maze(player_locations[name])

    # A channel centered on the opponent
    opponent_name = get_opponent_name(teams, name)
    opponent_channel = _center_maze(player_locations[opponent_name])

    # Concatenate channels
    state = torch.cat([player_channel, opponent_channel], dim=0).double()
    return state

#####################################################################################################################################################

def select_action ( state:              torch.tensor,
                    possible_actions:   List[str],
                    model:              DQN,
                    epsilon:            float,
                  ) ->                  str:

    """
        This function returns one of the possible actions, given the state of the game.
        In:
            * state:              State of the game in which we are now.
            * possible_actions:   Actions that the agent can perform.
            * model:              Deep Q learning model used to predict the next action.
        Out:
            * action: One action chosen from possible_actions.
    """
    if random.random() < epsilon:
        return random.choice(['north', 'east', 'south', 'west'])
    else:
    # We use the model's prediction as a basis for the action
        with torch.no_grad():
            model = model.to(model.device)
            model_outputs = model(state.unsqueeze(0))

        # We get the best action
        action_index = torch.argmax(model_outputs).item()
        action = possible_actions[action_index]
        return action

def decay_epsilon(epsilon):
    global epsilon_at_start
    return max(EPSILON_END, epsilon * EPSILON_DECAY)
####################################################################################################################################################

#####################################################################################################################################################

def compute_reward(
        state: torch.Tensor,
        scores_at_state: Dict[str, float],
        new_state: torch.Tensor,
        scores_at_new_state: Dict[str, float],
        name: str,
        teams: Dict[str, List[str]],
        initial_cheese_count: int,
        player_location: int,
        new_player_location: int,
        cheese_locations: List[int],
        opponent_location: int,
        maze_width: int
    ) -> float:
    """
        This function returns a reward for going from a state to another.
        Here we just return a negative constant, as we want to minimize the number of turns before gathering all cheese.
        In:
            * state:                State of the game before moving.
            * scores_at_state:      Scores of the players before moving.
            * new_state:            State of the game after moving.
            * scores_at_new_state:  Scores of the players after moving.
            * name:                 Name of the player being trained.
            * teams:                Recap of the teams of players.
            * initial_cheese_count: Number of cheese in the maze at the beginning of the game.
        Out:
            * reward: Value of the reward to return.
    """
    def manhattan_distance(a, b):  
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    player_pos = (player_location // maze_width, player_location % maze_width)  
    new_player_pos = (new_player_location // maze_width, new_player_location % maze_width)  
    opponent_pos = (opponent_location // maze_width, opponent_location % maze_width)

    # Initial reward  
    reward = 0.0

    # Determine the phase of the game  
    remaining_cheeses = len(cheese_locations)  
    if remaining_cheeses <= initial_cheese_count * 0.4:  # Final phase (20% of the initial cheese count)  
        endgame_multiplier = 1.5  # Slightly reduced endgame multiplier  
    else:  
        endgame_multiplier = 1.0

    # Reward for collecting cheese  
    if new_player_location in cheese_locations:  
        reward += 10.0 * endgame_multiplier  # Collecting cheese reward remains the same

    # Dense reward for moving closer to the nearest cheese  
    cheese_distances = [manhattan_distance(new_player_pos, (c // maze_width, c % maze_width)) for c in cheese_locations]  
    nearest_cheese_distance = min(cheese_distances) if cheese_distances else 0

    previous_cheese_distances = [manhattan_distance(player_pos, (c // maze_width, c % maze_width)) for c in cheese_locations]  
    previous_nearest_cheese_distance = min(previous_cheese_distances) if previous_cheese_distances else 0

    if nearest_cheese_distance < previous_nearest_cheese_distance:  
        reward += 0.5 * (previous_nearest_cheese_distance - nearest_cheese_distance) * endgame_multiplier  # Reduced reward for moving closer to cheese  
    elif nearest_cheese_distance > previous_nearest_cheese_distance:  
        reward -= 0.5 * (nearest_cheese_distance - previous_nearest_cheese_distance) * endgame_multiplier  # Reduced penalty for moving away from cheese

    # Reward for moving closer to multiple cheeses
    total_cheese_distance = sum(cheese_distances) if cheese_distances else 0
    previous_total_cheese_distance = sum(previous_cheese_distances) if previous_cheese_distances else 0

    if total_cheese_distance < previous_total_cheese_distance:
        reward += 0.3 * (previous_total_cheese_distance - total_cheese_distance) * endgame_multiplier  # Reduced reward for overall closer proximity to multiple cheeses

    # Penalty for letting the opponent get closer to the cheese  
    opponent_cheese_distances = [manhattan_distance(opponent_pos, (c // maze_width, c % maze_width)) for c in cheese_locations]  
    nearest_opponent_cheese_distance = min(opponent_cheese_distances) if opponent_cheese_distances else 0

    if nearest_opponent_cheese_distance < nearest_cheese_distance:  
        reward -= 1.0 * endgame_multiplier  # Penalty if the opponent is closer to a cheese than the player

    # Reward for collecting more cheese than the opponent  
    opponent_name = get_opponent_name(teams, name)  
    if scores_at_new_state[name] > scores_at_new_state[opponent_name]:  
        reward += 5.0 * endgame_multiplier  # Reward for having more cheese than the opponent  
    elif scores_at_new_state[name] < scores_at_new_state[opponent_name]:  
        reward -= 5.0 * endgame_multiplier  # Penalty for having less cheese than the opponent

    # Reward for moving closer to the opponent
    previous_opponent_distance = manhattan_distance(player_pos, opponent_pos)
    new_opponent_distance = manhattan_distance(new_player_pos, opponent_pos)
    if new_opponent_distance < previous_opponent_distance:
        reward += 0.5 * (previous_opponent_distance - new_opponent_distance) * endgame_multiplier  # Reduced reward for moving closer to the opponent

    # Penalty for staying in the same place
    if player_location == new_player_location:
        reward -= 0.5  # Reduced penalty for staying in the same place

    # Reward for reaching strategic positions (for simplicity, let's assume intersections are strategic)
    strategic_positions = [i * maze_width // 2 for i in range(2, maze_width, 2)]  # Example strategic positions
    if new_player_location in strategic_positions:
        reward += 2.0  # Reduced reward for reaching a strategic position

    # Small penalty for each move to encourage efficiency
    reward -= 0.1  # Small penalty per move remains the same

    return reward

def get_opponent_name(teams, name):
    for team, members in teams.items():
        if name in members:
            opponent_team = (set(teams.keys()) - {team}).pop()
            return teams[opponent_team][0]  # Assumes there is only one opponent
    return None


#####################################################################################################################################################

def make_batch ( model:            DQN,
                 experience:       List[Dict[str, Any]],
                 possible_actions: List[str]
               ) ->                Tuple[torch.tensor, torch.tensor]:

    """
        This function builds batches from the memory to train the model on.
        Each batch is a pair (data, target), where each element has batch size as first dimension.
        In:
            * model:            Model to train.
            * experience:       List of experience situations encountered across games.
            * possible_actions: Actions mapped with the output of the model.    
        Out:
            * data:    Batch of data from the memory.
            * targets: Targets associated with the sampled data.
    """
    # Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get indices
    batch_size = min(BATCH_SIZE, len(experience))
    indices = random.sample(range(len(experience)), batch_size)

    # Create the batch
    data = torch.zeros(batch_size, *experience[0]["state"].shape, device=device)
    targets = torch.zeros(batch_size, len(possible_actions), device=device)
    for i in range(batch_size):
        
        # Data is the sampled state
        data[i] = experience[indices[i]]["state"]
        
        # Target is the discounted reward
        with torch.no_grad():
            targets[i] = model(data[i].unsqueeze(0))
            if experience[indices[i]]["over"]:
                targets[i, possible_actions.index(experience[indices[i]]["action"])] = experience[indices[i]]["reward"]
            else:
                model_outputs = model(experience[indices[i]]["new_state"].unsqueeze(0))
                targets[i, possible_actions.index(experience[indices[i]]["action"])] = experience[indices[i]]["reward"] + DISCOUNT_FACTOR * torch.max(model_outputs).item()

    # Done
    return data, targets
    
#####################################################################################################################################################

def train_model ( model:            DQN,
                  optimizer:        torch.optim.Optimizer,
                  experience:       List[Dict[str, Any]],
                  possible_actions: List[str]
                ) ->                None:

    """
        This function trains the model on the experience.
        In:
            * model:            Model to train.
            * optimizer:        Optimizer used to train the model.
            * experience:       List of experience situations encountered across games.
            * possible_actions: Actions mapped with the output of the model.
        Out:
            * None.
    """
    
    # Define the loss function
    loss_function = torch.nn.SmoothL1Loss()

    # Convert it to GPU if possible
    model = model.to(model.device)

    # Ensure model is in train mode
    model.train()
    
    # Train loop
    total_loss = 0
    for b in range(NB_BATCHES):
        
        # Create a random batch
        data, targets = make_batch(model, experience, possible_actions)
        
        # Reset gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        outputs = model(data)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # Accumulate total loss for debug
        total_loss += loss.item()
    
    # Log loss
    if USE_WANDB:
        wandb.log({"total_loss": total_loss})
    
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
    global epsilon_at_start
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

    # Instanciate a DQN model with weights loaded from file (if any)
    state_dimension = build_state(maze, maze_width, maze_height, name, teams, player_locations, cheese).shape
    actions_dimension = len(possible_actions)
    memory.model = DQN(state_dimension, actions_dimension)
    if os.path.exists(MODEL_FILE_NAME):
        memory.model.load_state_dict(torch.load(MODEL_FILE_NAME))
    elif not TRAIN_MODEL:
        print("Doesn't load a state dict")
    
    # Instanciate an optimizer, initialized in its previous state to preserve momentum (if any)
    memory.optimizer = torch.optim.SGD(memory.model.parameters(), lr=LEARNING_RATE)
    if os.path.exists(OPTIMIZER_FILE_NAME):
        memory.optimizer.load_state_dict(torch.load(OPTIMIZER_FILE_NAME))
    elif not TRAIN_MODEL:
        print("Doesn't load a state dict")

    # In train mode, load the experience from previous games (if any)
    memory.experience = []
    memory.epsilon = epsilon_at_start if TRAIN_MODEL else 0
    
    if TRAIN_MODEL:
        if os.path.exists(EXPERIENCE_FILE_NAME):
            memory.experience = torch.load(EXPERIENCE_FILE_NAME)

    # We remember the initial cheese count and previous turn scores
    memory.initial_cheese_count = len(cheese)
    memory.previous_scores = {p: 0 for p in player_locations}

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
    
    # Select an action to perform
    state = build_state(maze, maze_width, maze_height, name, teams, player_locations, cheese)
    action = select_action(state, possible_actions, memory.model, memory.epsilon)
    
    # Stuff to do if training the model
    if TRAIN_MODEL:
        # Remove old experience entries if needed
        if len(memory.experience) >= MAX_EXPERIENCE_SIZE:
            del memory.experience[0]
        # Complement the previous turn and initialize the current one
        if len(memory.experience) > 0 and "over" not in memory.experience[-1]:
            reward = compute_reward(memory.experience[-1]["state"], memory.previous_scores, state, player_scores, name, teams, memory.initial_cheese_count, memory.experience[-1]["player_location"], player_locations[name], cheese, player_locations[get_opponent_name(teams, name)], maze_width)

            memory.experience[-1]["reward"] = reward
            memory.experience[-1]["new_state"] = state
            memory.experience[-1]["over"] = False
        memory.experience.append({"state": state, "action": action, "player_location": player_locations[name]})

        # Save scores for next turn
        memory.previous_scores = player_scores

        # Update epsilon value
        memory.epsilon = decay_epsilon(memory.epsilon)
    # Done
    return action

#####################################################################################################################################################
######################################################## EXECUTED ONCE AT THE END OF THE GAME #######################################################
#####################################################################################################################################################

def postprocessing ( maze:             Union[numpy.ndarray, Dict[int, Dict[int, int]]],
                     maze_width:       int,
                     maze_height:      int,
                     name:             str,
                     teams:            Dict[str, List[str]],
                     player_locations: Dict[str, int],
                     player_scores:    Dict[str, float],
                     player_muds:      Dict[str, Dict[str, Union[None, int]]],
                     cheese:           List[int],
                     possible_actions: List[str],
                     memory:           threading.local,
                     stats:            Dict[str, Any],
                   ) ->                None:

    """
        This function is called once at the end of the game.
        It is not timed, and can be used to make some cleanup, analyses of the completed game, model training, etc.
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
            * None.
    """
    global epsilon_at_start

    # Stuff to do if training the model
    if TRAIN_MODEL:
        
        # Complement the experience and save it
        state = build_state(maze, maze_width, maze_height, name, teams, player_locations, cheese)
        if len(memory.experience) > 0 and "over" not in memory.experience[-1]:
            memory.experience[-1]["reward"] = compute_reward(memory.experience[-1]["state"], memory.previous_scores, state, player_scores, name, teams, memory.initial_cheese_count, memory.experience[-1]["player_location"], player_locations[name], cheese, player_locations[get_opponent_name(teams, name)], maze_width)
            memory.experience[-1]["new_state"] = state
            memory.experience[-1]["over"] = True
        torch.save(memory.experience, EXPERIENCE_FILE_NAME)
        
        # Train the model if we have enough memory
        if len(memory.experience) >= EXPERIENCE_MIN_SIZE_FOR_TRAINING:
            train_model(memory.model, memory.optimizer, memory.experience, possible_actions)
            torch.save(memory.model.state_dict(), MODEL_FILE_NAME)
            torch.save(memory.optimizer.state_dict(), OPTIMIZER_FILE_NAME)
        
        # Epsilon decay
        epsilon_at_start = memory.epsilon
        
        # Log the final scores
        if USE_WANDB:
            opponent_name = get_opponent_name(teams, name)
            wandb.log({"final_score[player]": player_scores[name],
                       "final_score[opponent]": player_scores[opponent_name],
                       "final_score_difference": player_scores[name] - player_scores[opponent_name]})
    
#####################################################################################################################################################
######################################################################## GO! ########################################################################
#####################################################################################################################################################

if __name__ == "__main__":
    # Map the functions to the character
    players = [{"name": "RL",
                    "team": "You",
                    "skin": "rat",
                    "preprocessing_function": preprocessing,
                    "turn_function": turn,
                    "postprocessing_function": postprocessing},
               {"name": "Greedy",
                    "team": "Opponent",
                    "skin": "python",
                    "preprocessing_function": opponent.preprocessing if "preprocessing" in dir(opponent) else None,
                    "turn_function": opponent.turn,
                    "postprocessing_function": opponent.postprocessing if "postprocessing" in dir(opponent) else None}]

    # Customize the game elements
    config = {"maze_width": 7,
              "maze_height": 5,
              "cell_percentage": 100.0,
              "wall_percentage": 0.0,
              "mud_percentage": 0.0,
              "nb_cheese": 10}

    # Train mode
    if TRAIN_MODEL:
        # Initialize epsilon value
        epsilon_at_start = EPSILON_START

        # Remove old files if needed
        if RESET_TRAINING:
            if os.path.exists(OUTPUT_DIRECTORY):
                shutil.rmtree(OUTPUT_DIRECTORY, ignore_errors=True)
            os.mkdir(OUTPUT_DIRECTORY)
        
        # Connect to WandB for monitoring
        if USE_WANDB:
            wandb.login(key=open(WANDB_KEY_PATH).read().strip(), force=True)
            wandb.init(project="PyRat_RL", dir=OUTPUT_DIRECTORY)
        
        # Run multiple games with no GUI
        config["game_mode"] = "sequential"
        config["preprocessing_time"] = 0.0
        config["turn_time"] = 0.0
        config["render_mode"] = "no_rendering"
        for i in tqdm.tqdm(range(NB_EPISODES), desc="Episode", position=0, leave=False):
            print(epsilon_at_start)
            game = PyRat(players, **config)
            stats = game.start()
            if stats == {}:
                break
            
    # Test mode
    else:
        
        # Make a single game with GUI
        game = PyRat(players, **config)
        stats = game.start()
        print(stats)

#####################################################################################################################################################
#####################################################################################################################################################
