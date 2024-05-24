#####################################################################################################################################################
######################################################################## INFO #######################################################################
#####################################################################################################################################################

"""
    This file contains useful elements to define a particular player.
    In order to use this player, you need to instanciate it and add it to a game.
    Please refer to example games to see how to do it properly.
"""

#####################################################################################################################################################
###################################################################### IMPORTS ######################################################################
#####################################################################################################################################################

# External imports
from typing import *
from typing_extensions import *
from numbers import *
import random

# PyRat imports
from pyrat import Player, Maze, GameState, Action

#####################################################################################################################################################
###################################################################### CLASSES ######################################################################
#####################################################################################################################################################

class Random3 (Player):

    """
        This player is an improvement of the Random2 player.
        Here, we add elements that help us explore better the maze.
        More precisely, we keep a list (in a global variable to be updated at each turn) of cells that have already been visited in the game.
        Then, at each turn, we choose in priority a random move among those that lead us to an unvisited cell.
        If no such move exists, we move randomly using the method in "random_2".
    """

    #############################################################################################################################################
    #                                                                CONSTRUCTOR                                                                #
    #############################################################################################################################################

    def __init__ ( self:     Self,
                   *args:    Any,
                   **kwargs: Any
                 ) ->        Self:

        """
            This function is the constructor of the class.
            In:
                * self:   Reference to the current object.
                * args:   Arguments to pass to the parent constructor.
                * kwargs: Keyword arguments to pass to the parent constructor.
            Out:
                * A new instance of the class.
        """

        # Inherit from parent class
        super().__init__(*args, **kwargs)

        # We create an attribute to keep track of visited cells
        # By convention, in Python, attributes that are not supposed to be called from outside the class should start with two underscores
        self.__visited_cells = []
       
    #############################################################################################################################################
    #                                                               PUBLIC METHODS                                                              #
    #############################################################################################################################################

    def turn ( self:       Self,
               maze:       Maze,
               game_state: GameState,
             ) ->          Action:

        """
            This method redefines the abstract method of the parent class.
            It is called at each turn of the game.
            It returns an action that explores a random unvisited cell if possible.
            If no such action exists, it returns a random action that does not lead to a wall.
            In:
                * self:       Reference to the current object.
                * maze:       An object representing the maze in which the player plays.
                * game_state: An object representing the state of the game.
            Out:
                * action: One of the possible actions.
        """

        # Mark current cell as visited
        if game_state.player_locations[self.name] not in self.__visited_cells:
            self.__visited_cells.append(game_state.player_locations[self.name])

        # Go to an unvisited neighbor in priority
        neighbors = maze.get_neighbors(game_state.player_locations[self.name])
        unvisited_neighbors = [neighbor for neighbor in neighbors if neighbor not in self.__visited_cells]
        if len(unvisited_neighbors) > 0:
            neighbor = random.choice(unvisited_neighbors)
            
        # If there is no unvisited neighbor, choose one randomly
        else:
            neighbor = random.choice(neighbors)
        
        # Retrieve the corresponding action
        action = maze.locations_to_action(game_state.player_locations[self.name], neighbor)
        return action

#####################################################################################################################################################
#####################################################################################################################################################
