####################################################################################################################################################
##########################################Stratégie améliorée basique

#Une forte récompense de `10.0` est attribuée pour la collecte de chaque fromage.
#Une récompense graduelle est attribuée pour la réduction de la distance au fromage.
#   * Une pénalité de `1.0` est attribuée si l'adversaire est plus proche d'un fromage que le joueur.
#   * Une récompense de `5.0` est attribuée si le joueur a plus de fromages que l'adversaire, et une pénalité de `5.0` si c'est l'inverse.
#Une petite pénalité de `0.1` pour chaque mouvement encourage l'efficacité des déplacements.
#rendre le jeu plus agressif à la phase final (quand 40% de fromages restants)
#Les récompenses pour la collecte de fromages, la réduction de la distance au fromage, et les interactions avec l'adversaire sont multipliées par `endgame_multiplier` si le jeu est dans sa phase finale.
####################################################################################################################################################



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
    if remaining_cheeses <= initial_cheese_count * 0.2:  # Final phase (20% of the initial cheese count)  
        endgame_multiplier = 2.0  
    else:  
        endgame_multiplier = 1.0

    # Reward for collecting cheese  
    if new_player_location in cheese_locations:  
        reward += 10.0 * endgame_multiplier

    # Dense reward for moving closer to the nearest cheese  
    cheese_distances = [manhattan_distance(new_player_pos, (c // maze_width, c % maze_width)) for c in cheese_locations]  
    nearest_cheese_distance = min(cheese_distances) if cheese_distances else 0

    previous_cheese_distances = [manhattan_distance(player_pos, (c // maze_width, c % maze_width)) for c in cheese_locations]  
    previous_nearest_cheese_distance = min(previous_cheese_distances) if previous_cheese_distances else 0

    if nearest_cheese_distance < previous_nearest_cheese_distance:  
        reward += (previous_nearest_cheese_distance - nearest_cheese_distance) * endgame_multiplier  
    elif nearest_cheese_distance > previous_nearest_cheese_distance:  
        reward -= (nearest_cheese_distance - previous_nearest_cheese_distance) * endgame_multiplier

    # Penalty for letting the opponent get closer to the cheese  
    opponent_cheese_distances = [manhattan_distance(opponent_pos, (c // maze_width, c % maze_width)) for c in cheese_locations]  
    nearest_opponent_cheese_distance = min(opponent_cheese_distances) if opponent_cheese_distances else 0

    if nearest_opponent_cheese_distance < nearest_cheese_distance:  
        reward -= 1.0 * endgame_multiplier  # Penalty if the opponent is closer to a cheese than the player

    # Reward for collecting more cheese than the opponent  
    opponent_name = get_opponent_name(teams, name)  
    if scores_at_new_state[name] > scores_at_new_state[opponent_name]:  
        reward += 5.0 * endgame_multiplier  
    elif scores_at_new_state[name] < scores_at_new_state[opponent_name]:  
        reward -= 5.0 * endgame_multiplier

    # Small penalty for each move to encourage efficiency  
    reward -= 0.1

    return reward

def get_opponent_name(teams, name):
    for team, members in teams.items():
        if name in members:
            opponent_team = (set(teams.keys()) - {team}).pop()
            return teams[opponent_team][0]  # Assumes there is only one opponent
    return None



####################################################################################################################################################
##########################################Stratégie améliorée équilibrée (la meilleure pour l'instant)
#Multiplier de fin de partie réduit : Le multiplicateur de fin de partie est réduit pour diminuer l'agressivité.
#Récompenses et pénalités pour la distance au fromage réduites : Les valeurs pour se rapprocher ou s'éloigner des fromages sont réduites.
#Récompense pour plusieurs fromages réduite : La récompense pour se rapprocher de plusieurs fromages est réduite.
#Récompense pour se rapprocher de l'adversaire réduite : La récompense pour se rapprocher de l'adversaire est réduite.
#Pénalité pour rester immobile réduite : La pénalité pour rester immobile est réduite.
#Récompense pour positions stratégiques réduite : La récompense pour atteindre des positions stratégiques est réduite.
####################################################################################################################################################

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
    if remaining_cheeses <= initial_cheese_count * 0.2:  # Final phase (20% of the initial cheese count)  
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

####################################################################################################################################################
##########################################Stratégie améliorée agressive
#Récompense pour se rapprocher de plusieurs fromages : Si la somme des distances aux fromages est réduite, une récompense est ajoutée.
#Récompense pour atteindre des positions stratégiques : Par exemple, des intersections importantes dans le labyrinthe. Vous pouvez ajuster la logique pour identifier les positions stratégiques spécifiques.
#Pénalité pour rester immobile : Une pénalité est ajoutée si le joueur reste au même endroit.
#Récompense pour se rapprocher de l'adversaire : La récompense pour se rapprocher de l'adversaire est maintenue.
#Penalités et récompenses renforcées : Les valeurs de récompense et de pénalité ont été ajustées pour encourager une stratégie plus agressive.
####################################################################################################################################################
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
    if remaining_cheeses <= initial_cheese_count * 0.2:  # Final phase (20% of the initial cheese count)  
        endgame_multiplier = 2.0  
    else:  
        endgame_multiplier = 1.0

    # Reward for collecting cheese  
    if new_player_location in cheese_locations:  
        reward += 15.0 * endgame_multiplier  # Increased reward for collecting cheese

    # Dense reward for moving closer to the nearest cheese  
    cheese_distances = [manhattan_distance(new_player_pos, (c // maze_width, c % maze_width)) for c in cheese_locations]  
    nearest_cheese_distance = min(cheese_distances) if cheese_distances else 0

    previous_cheese_distances = [manhattan_distance(player_pos, (c // maze_width, c % maze_width)) for c in cheese_locations]  
    previous_nearest_cheese_distance = min(previous_cheese_distances) if previous_cheese_distances else 0

    if nearest_cheese_distance < previous_nearest_cheese_distance:  
        reward += 1.0 * (previous_nearest_cheese_distance - nearest_cheese_distance) * endgame_multiplier  # Increased reward for moving closer to cheese  
    elif nearest_cheese_distance > previous_nearest_cheese_distance:  
        reward -= 1.0 * (nearest_cheese_distance - previous_nearest_cheese_distance) * endgame_multiplier  # Increased penalty for moving away from cheese

    # Reward for moving closer to multiple cheeses
    total_cheese_distance = sum(cheese_distances) if cheese_distances else 0
    previous_total_cheese_distance = sum(previous_cheese_distances) if previous_cheese_distances else 0

    if total_cheese_distance < previous_total_cheese_distance:
        reward += 0.5 * (previous_total_cheese_distance - total_cheese_distance) * endgame_multiplier  # Reward for overall closer proximity to multiple cheeses

    # Penalty for letting the opponent get closer to the cheese  
    opponent_cheese_distances = [manhattan_distance(opponent_pos, (c // maze_width, c % maze_width)) for c in cheese_locations]  
    nearest_opponent_cheese_distance = min(opponent_cheese_distances) if opponent_cheese_distances else 0

    if nearest_opponent_cheese_distance < nearest_cheese_distance:  
        reward -= 2.0 * endgame_multiplier  # Increased penalty if the opponent is closer to a cheese than the player

    # Reward for collecting more cheese than the opponent  
    opponent_name = get_opponent_name(teams, name)  
    if scores_at_new_state[name] > scores_at_new_state[opponent_name]:  
        reward += 10.0 * endgame_multiplier  # Increased reward for having more cheese than the opponent  
    elif scores_at_new_state[name] < scores_at_new_state[opponent_name]:  
        reward -= 10.0 * endgame_multiplier  # Increased penalty for having less cheese than the opponent

    # Reward for moving closer to the opponent
    previous_opponent_distance = manhattan_distance(player_pos, opponent_pos)
    new_opponent_distance = manhattan_distance(new_player_pos, opponent_pos)
    if new_opponent_distance < previous_opponent_distance:
        reward += 1.0 * (previous_opponent_distance - new_opponent_distance) * endgame_multiplier  # Reward for moving closer to the opponent

    # Penalty for staying in the same place
    if player_location == new_player_location:
        reward -= 1.0  # Penalty for staying in the same place

    # Reward for reaching strategic positions (for simplicity, let's assume intersections are strategic)
    strategic_positions = [i * maze_width // 2 for i in range(2, maze_width, 2)]  # Example strategic positions
    if new_player_location in strategic_positions:
        reward += 5.0  # Reward for reaching a strategic position

    # Small penalty for each move to encourage efficiency
    reward -= 0.2  # Slightly increased penalty per move

    return reward

def get_opponent_name(teams, name):
    for team, members in teams.items():
        if name in members:
            opponent_team = (set(teams.keys()) - {team}).pop()
            return teams[opponent_team][0]  # Assumes there is only one opponent
    return None
