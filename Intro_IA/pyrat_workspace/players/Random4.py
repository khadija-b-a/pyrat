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

class Random4 (Player):

    """
        This player is an improvement of the Random3 player.
        A limitation of Random3 is that when a cell has no unvisited neighbor, then we may move randomly for a long time before reaching an unvisited cell.
        To correct this issue, we keep track of our trajectory.
        This way, if we are stuck, we can backtrack until we reach an already visited cell with an unvisited neighbor.
        This strategy is pretty close to a depth-first seach.
        Note that in this file, we redefine the "preprocessing" function of the parent class, that is executed once at the beginning of the game.
        Here, this function is used to store the initial location of the player.
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

        # We create an attribute to keep track of the trajectory
        # By convention, in Python, attributes that are not supposed to be called from outside the class should start with two underscores
        self.__trajectory = []
       
    #############################################################################################################################################
    #                                                               PUBLIC METHODS                                                              #
    #############################################################################################################################################

    def preprocessing ( self:       Self,
                        maze:       Maze,
                        game_state: GameState,
                      ) ->          None:
        
        """
            This method redefines the method of the parent class.
            It is called once at the beginning of the game.
            Here, we initialize the trajectory with the initial location of the player.
            In:
                * self:       Reference to the current object.
                * maze:       An object representing the maze in which the player plays.
                * game_state: An object representing the state of the game.
            Out:
                * None.
        """
        
        # Store location to initialize trajectory
        self.__trajectory.append(game_state.player_locations[self.name])

    #############################################################################################################################################

    def turn ( self:       Self,
               maze:       Maze,
               game_state: GameState,
             ) ->          Action:

        """
            This method redefines the abstract method of the parent class.
            It is called at each turn of the game.
            It returns an action that explores a random unvisited cell if possible.
            If no such action exists, it returns an action that allows to go back on our trajectory.
            In:
                * self:       Reference to the current object.
                * maze:       An object representing the maze in which the player plays.
                * game_state: An object representing the state of the game.
            Out:
                * action: One of the possible actions.
        """

        # Mark current cell as visited and add it to the trajectory
        if game_state.player_locations[self.name] not in self.__visited_cells:
            self.__visited_cells.append(game_state.player_locations[self.name])
        self.__trajectory.append(game_state.player_locations[self.name])

        # Go to an unvisited neighbor in priority
        neighbors = maze.get_neighbors(game_state.player_locations[self.name])
        unvisited_neighbors = [neighbor for neighbor in neighbors if neighbor not in self.__visited_cells]
        if len(unvisited_neighbors) > 0:
            neighbor = random.choice(unvisited_neighbors)
            
        # If there is no unvisited neighbor, backtrack the trajectory
        else:
            _ = self.__trajectory.pop(-1)
            neighbor = self.__trajectory.pop(-1)
        
        # Retrieve the corresponding action
        action = maze.locations_to_action(game_state.player_locations[self.name], neighbor)
        return action

#####################################################################################################################################################
#####################################################################################################################################################
