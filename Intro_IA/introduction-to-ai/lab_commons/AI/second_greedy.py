#################################################################################################################
###################################################### INFO #####################################################
#################################################################################################################

"""
    This program is an improvement of "greedy_2".
    Here, we define a better score function, which is inversely proportional to the TSP result, starting from the considered vertex, up to a given depth.
    Let's make it play against "greedy_2" to see if it takes the same route.
"""

#################################################################################################################
#################################################### IMPORTS ####################################################
#################################################################################################################

# Import PyRat
from pyrat import *
from lab_commons.utils import locations_to_action, get_neighbors, find_route, dijkstra

# External imports 
import numpy

#################################################################################################################
################################################### FUNCTIONS ###################################################
#################################################################################################################

def graph_to_metagraph ( graph:    Union[numpy.ndarray, Dict[int, Dict[int, int]]],
                         vertices: List[int]
                       ) ->        Tuple[numpy.ndarray, Dict[int, Dict[int, Union[None, int]]]]:
    """
        Function to build a complete graph out of locations of interest in a given graph.
        In:
            * graph:    Graph containing the vertices of interest.
            * vertices: Vertices to use in the complete graph.
        Out:
            * complete_graph: Complete graph of the vertices of interest.
            * routing_tables: Dictionary of routing tables obtained by traversals used to build the complete graph.
    """
    
    # Initialize an adjacency matrix
    complete_graph = numpy.zeros((len(vertices), len(vertices)), dtype=int)
    
    # Perform traversals from each vertex
    routing_tables = {}
    for i in range(len(vertices)):
        distances_to_explored_vertices, routing_table = dijkstra(vertices[i], graph)
        
        # Store weights and routing table
        for j in range(len(vertices)):
            complete_graph[i, j] = distances_to_explored_vertices[vertices[j]]
        routing_tables[vertices[i]] = routing_table
    
    # Return the graph and the routing tables
    return complete_graph, routing_tables

def tsp ( complete_graph: numpy.ndarray,
          source:         int,
          max_depth:      Union[None, int] = None
        ) ->              Tuple[List[int], int]:

    """
        Function to solve the TSP using an exhaustive search up to a given depth.
        In:
            * complete_graph: Complete graph of the vertices of interest.
            * source:         Vertex used to start the search.
            * max_depth:      Maximum number of vertices to search before evaluating the path.
        Out:
            * best_route:  Best route found in the search.
            * best_length: Length of the best route found.
    """
    
    # Check if a max_depth is provided
    if max_depth is None:
        max_depth = complete_graph.shape[0]
    
    # We sort the neighbors in increasing distance
    sorted_neighbors = {}
    for vertex in range(complete_graph.shape[0]):
        neighbors = get_neighbors(vertex, complete_graph)
        neighbors_weights = [complete_graph[vertex, neighbor] for neighbor in neighbors]
        sorted_neighbors[vertex] = [neighbors[i] for i in numpy.argsort(neighbors_weights)]
    
    # Subfunction for recursive calls
    def _tsp (current_vertex, current_visited_vertices, current_route, current_length, current_best_route, current_best_length):
        
        # Backtracking
        if current_length >= current_best_length:
            return current_best_route, current_best_length
        
        # If we have a full path, we evaluate it
        if len(current_visited_vertices) == max_depth:
            if current_length >= current_best_length:
                return current_best_route, current_best_length
            return current_route, current_length
        
        # Otherwise, we explore one more neighbor
        for vertex in sorted_neighbors[current_vertex]:
            if vertex not in current_visited_vertices:
                current_best_route, current_best_length = _tsp(vertex, current_visited_vertices + [vertex], current_route + [vertex], current_length + complete_graph[current_vertex, vertex], current_best_route, current_best_length)
        
        # We propagate the current best
        return current_best_route, current_best_length
    
    # Initialize the search from the source
    best_route, best_length = _tsp(source, [source], [source], 0, None, float("inf"))
    return best_route, best_length

#################################################################################################################

def give_score ( graph:          Union[numpy.ndarray, Dict[int, Dict[int, int]]],
                 current_vertex: int,
                 targets:        List[int],
                 max_depth:      int = 3
               ) ->              Tuple[List[float], Dict[int, Union[None, int]]]:

    """
        Function that associates a score to each target.
        In:
            * graph:          Graph containing the vertices.
            * current_vertex: Current location of the player in the maze.
            * max_depth:      Maximum number of vertices to search before evaluating the path.
            
        Out:
            * scores:        Scores given to the targets.
            * routing_table: Routing table obtained from the current vertex.
    """
    
    # We compute distances from the current vertex to targets
    distances_to_explored_vertices, routing_table = dijkstra(current_vertex, graph)
    scores = [distances_to_explored_vertices[target] for target in targets]

    # For each of these targets that is not already too far, we perform a TSP up to a given depth and accumulate with the distance
    best_score = float("inf")
    for i in range(len(targets)):
        if scores[i] < best_score:
            locations_of_interest = [target for target in targets if target != current_vertex]
            complete_graph, _ = graph_to_metagraph(graph, locations_of_interest)
            _, length = tsp(complete_graph, i, max_depth)
            scores[i] += length
            best_score = min(best_score, scores[i])
    
    # Scores are inversely proportional to that distance
    scores = [1.0 / score for score in scores]
    return scores, routing_table
    
#################################################################################################################
####################################### EXECUTED AT EACH TURN OF THE GAME #######################################
#################################################################################################################

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
    
    # We find the next cheese to get using the greedy method
    scores, routing_table = give_score(maze, player_locations[name], cheese)
    best_target = cheese[numpy.argmax(scores)]
    route = find_route(routing_table, player_locations[name], best_target)

    # We move once in that direction
    action = locations_to_action(route[0], route[1], maze_width)
    return action

#################################################################################################################
###################################################### GO ! #####################################################
#################################################################################################################

if __name__ == "__main__":

    # Map the functions to the characters
    players = [{"name": "Greedy 3", "turn_function": turn, "team": "greedy"},
               {"name": "Opponent", "skin": "rat", "preprocessing_function": opponent.preprocessing if "preprocessing" in dir(opponent) else None, "turn_function": opponent.turn, "team": "other"}]

    # Customize the game elements
    config = {"maze_width": 15,
              "maze_height": 11,
              "mud_percentage": 40.0,
              "nb_cheese": 2,
              "trace_length": 1000}

    # Start the game
    game = PyRat(players, **config)
    stats = game.start()

    # Show statistics
    print(stats)

#################################################################################################################
#################################################################################################################