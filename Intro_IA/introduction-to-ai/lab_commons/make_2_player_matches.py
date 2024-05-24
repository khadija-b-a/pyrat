#####################################################################################################################################################
######################################################################## INFO #######################################################################
#####################################################################################################################################################

"""
    This script makes multiple games between two programs, and compares the obtained scores.
    It performs two analyses: a quick average analysis and a formal 1 sample T test.
"""

#####################################################################################################################################################
###################################################################### IMPORTS ######################################################################
#####################################################################################################################################################

# Import PyRat
from pyrat import *

# External imports
import matplotlib.pyplot as pyplot
import scipy.stats
import numpy
import types
import tqdm

# Previously developed functions
import lab_commons.AI.greedy as greedy
import lab_commons.AI.random as random

#####################################################################################################################################################
############################################################### VARIABLES & CONSTANTS ###############################################################
#####################################################################################################################################################

"""
    Number of games to make.
"""

NB_GAMES = 100

#####################################################################################################################################################

"""
    Games configuration.
"""

MAZE_WIDTH = 15
MAZE_HEIGHT = 15
MUD_PERCENTAGE = 0.0
CELL_PERCENTAGE = 100.0
WALL_PERCENTAGE = 0.0
NB_CHEESE = 7
TURN_TIME = 0.0
PREPROCESSING_TIME = 0.0
GAME_MODE = "synchronous"

#####################################################################################################################################################
##################################################################### FUNCTIONS #####################################################################
#####################################################################################################################################################

def run_one_game ( program_1:    types.ModuleType,
                   program_2:    types.ModuleType,
                   seed:         int=0,
                   nb_cheese:    int=NB_CHEESE,
                   maze_width:   int=MAZE_WIDTH,
                   maze_height:  int=MAZE_HEIGHT,
                   gui:          bool=False
                 ) ->            Tuple[numpy.ndarray, int]:


    """
        This function runs a PyRat game, with no GUI, for a given seed and program, and returns the obtained stats.
        In:
            * program_1:   First program to use in that game.
            * program_2:   Second program to use in that game.
            * seed:        Random seed used to create the game.
            * nb_cheese:   Number of cheeses to catch.
            * maze_width:  Width of the maze.
            * maze_height: Height of the maze.
            * gui:         Whether to display the game interface (True) or not (False).
        Out:
            * stats:         Statistics output at the end of the game.
            * cheese_matrix: The positions of the cheeses in the maze. (Useful for the Lab 1 and 2.)
            * winner:        Winner of the game (1 for program_1, -1 for program_2, and 0 for a draw) (Useful for the Lab 1 and 2.)
    """
    if gui == True:
        render_mode = "gui"
    else:
        render_mode = "no_rendering"

    # Map the functions to the character
    players = [{"name": program_1.__name__,
                    "team": "1",
                    "location": 0,
                    "preprocessing_function": program_1.preprocessing if "preprocessing" in dir(program_1) else None,
                    "turn_function": program_1.turn,
                    "postprocessing_function": program_1.postprocessing if "postprocessing" in dir(program_1) else None},
               {"name": program_2.__name__,
                    "team": "2",
                    "preprocessing_function": program_2.preprocessing if "preprocessing" in dir(program_2) else None,
                    "turn_function": program_2.turn,
                    "postprocessing_function": program_2.postprocessing if "postprocessing" in dir(program_2) else None,
                    "location": maze_height * maze_width - 1}]

    # Customize the game elements
    config = {"maze_width": maze_width,
              "maze_height": maze_height,
              "mud_percentage": MUD_PERCENTAGE,
              "cell_percentage": CELL_PERCENTAGE,
              "wall_percentage": WALL_PERCENTAGE,
              "nb_cheese": nb_cheese,
              "render_mode": render_mode,
              "preprocessing_time": PREPROCESSING_TIME,
              "turn_time": TURN_TIME,
              "game_mode": GAME_MODE,
              "random_seed": seed}
        
    # Start the game
    game = PyRat(players, **config)

    # Find the cheeses
    cheese = game.cheese
    cheese_matrix = numpy.zeros((maze_height*maze_width)) #numpy.zeros((maze_height, maze_width)) # if you want a 2D matrix
    numpy.put(cheese_matrix, cheese, 1)
    
    # Start the game
    stats = game.start()
    
    # Find the winner
    winner = numpy.sign(stats["players"][program_1.__name__]["score"] - stats["players"][program_2.__name__]["score"])
    
    return stats, cheese_matrix, winner

def run_several_games ( program_1:    types.ModuleType,
                        program_2:    types.ModuleType,
                        seed:         int=0,
                        nb_cheese:    int=NB_CHEESE,
                        maze_width:   int=MAZE_WIDTH,
                        maze_height:  int=MAZE_HEIGHT,
                        nb_games:     int=NB_GAMES
                      ) ->            Tuple[numpy.ndarray, numpy.ndarray]:
                      
    """
        This function runs a specified number of PyRat games, for a given program, with a given number of cheeses and sizes of the maze, and returns the positions of the cheeses and the winner of the game.
        Parameter gui specifies whether to display the game interface (must be false for large simulations).
        The seeds are set in the enumerate function.
        In:
            * program_1:   First program to use in that game.
            * program_2:   Second program to use in that game.
            * seed:        Random seed used to create the game.
            * nb_cheese:   Number of cheeses to catch.
            * maze_width:  Width of the maze.
            * maze_height: Height of the maze.
            * nb_games:    Number of games to run.
        Out:
            * scores:          Score difference at the end of each game.
            * cheese_matrices: The positions of the cheeses in the maze. (Useful for the Lab 1 and 2.)
            * winners:         Winner of the game (1 for program_1, -1 for program_2, and 0 for a draw) (Useful for the Lab 1 and 2.)
    """
    
    # Run multiple games
    scores = []
    cheese_matrices = []
    winners = []

    for seed in tqdm.tqdm(range(nb_games), desc="Game", position=0, leave=False):
        
        # Store score difference as result
        stats, cheese_matrix, winner = run_one_game(program_1, program_2, seed, nb_cheese=nb_cheese, maze_width=maze_width, maze_height=maze_height)
        scores.append(int(stats["players"][program_1.__name__]["score"] - stats["players"][program_2.__name__]["score"]))
        cheese_matrices.append(cheese_matrix)
        winners.append(winner)
    
    return scores, cheese_matrices, winners

    
#####################################################################################################################################################
######################################################################## GO! ########################################################################
#####################################################################################################################################################

if __name__ == "__main__":

    program_1 = random # You should change program_1 to your own AI once you start developing it.
    program_2 = greedy

    results, _, _ = run_several_games(program_1, program_2)
        
    # Show results briefly
    print("#" * 20)
    print("#  Quick analysis  #")
    print("#" * 20)
    rat_victories = [score for score in results if score > 0]
    python_victories = [score for score in results if score < 0]
    nb_draws = NB_GAMES - len(rat_victories) - len(python_victories)
    print(program_1.__name__, "(rat)   <-  ", len(rat_victories), "  -  ", nb_draws, "  -  ", len(python_victories), "  ->  ", program_2.__name__, "(python)")
    print("Average score difference when %s wins:" % program_1.__name__, numpy.mean(rat_victories) if len(rat_victories) > 0 else "n/a")
    print("Average score difference when %s wins:" % program_2.__name__, numpy.mean(numpy.abs(python_victories))if len(python_victories) > 0 else "n/a")

    # More formal statistics to check if the mean of the distribution is significantly different from 0
    print("#" * 21)
    print("#  Formal analysis  #")
    print("#" * 21)
    test_result = scipy.stats.ttest_1samp(results, popmean=0.0)
    print("One sample T-test of the distribution:", test_result)

    # Visualization of histograms of score differences
    bins = range(min(results), max(results) + 2)
    pyplot.figure(figsize=(20, 10))
    pyplot.hist(results, ec="black", bins=bins)
    pyplot.title("Analysis of the game results in terms of victory margin")
    pyplot.xlabel("score(%s) - score(%s)" % (program_1.__name__, program_2.__name__))
    pyplot.xticks([b + 0.5 for b in bins], labels=bins)
    pyplot.xlim(bins[0], bins[-1])
    pyplot.ylabel("Number of games")
    pyplot.show()
    
#####################################################################################################################################################
#####################################################################################################################################################