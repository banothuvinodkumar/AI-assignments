import heapq
from itertools import combinations

# --- BFS (Breadth-First Search using Priority Queue - Dijkstra's Algorithm) ---
def solve_bridge_problem_bfs(crossing_times, time_limit):
    """
    Solves the bridge crossing problem using a priority queue based BFS (Dijkstra's algorithm)
    to find the path with the minimum total time.
    """
    all_people = frozenset(crossing_times.keys())
    initial_state = (frozenset(all_people), frozenset(), 'start') # (start_side, end_side, umbrella_pos)
    
    pq = [(0, [initial_state])] # (current_time, path_to_state)
    
    # Stores the minimum time to reach a particular state (configuration)
    visited_times = {initial_state: 0} 

    while pq:
        current_time, path = heapq.heappop(pq)
        
        current_config = path[-1]
        start_side, end_side, umbrella_pos = current_config

        # If all people are on the end side, we found a solution.
        # Since it's a priority queue, this is the shortest time found so far.
        if not start_side:
            if current_time <= time_limit:
                return current_time, path, visited_times 
            else:
                # Path found, but exceeds time limit. Continue searching for a better one if available.
                continue 

        # Optimization: If we found a shorter path to this state already, skip.
        if current_time > visited_times.get(current_config, float('inf')):
            continue

        # Determine which side the umbrella is on and from where people will move
        if umbrella_pos == 'start':
            move_from = start_side
            move_to_pos = 'end'
        else:
            move_from = end_side
            move_to_pos = 'start'

        # Explore moving 1 or 2 people
        for num_people in [1, 2]:
            if len(move_from) < num_people:
                continue # Not enough people to form a group

            for group in combinations(move_from, num_people):
                group = frozenset(group)
                trip_time = max(crossing_times[person] for person in group) # Time taken by the slowest person in the group
                new_total_time = current_time + trip_time
                
                # Pruning: If this path already exceeds the time limit, don't explore further
                if new_total_time > time_limit:
                    continue

                # Calculate the new configuration after the move
                if umbrella_pos == 'start':
                    new_start_side = start_side - group
                    new_end_side = end_side | group
                else:
                    new_start_side = start_side | group
                    new_end_side = end_side - group
                
                new_config = (new_start_side, new_end_side, move_to_pos)

                # If this new configuration hasn't been visited or we found a shorter path to it
                if new_config not in visited_times or new_total_time < visited_times[new_config]:
                    visited_times[new_config] = new_total_time
                    new_path = path + [new_config]
                    heapq.heappush(pq, (new_total_time, new_path))
                    
    return None, None, visited_times # Return None if no solution found within the limit

# --- DFS (Depth-First Search) Implementation ---
def solve_bridge_problem_dfs(crossing_times, time_limit):
    """
    Solves the bridge crossing problem using DFS to find the shortest path.
    It explores paths depth-first but keeps track of the minimum time found.
    """
    all_people = frozenset(crossing_times.keys())
    initial_state = (frozenset(all_people), frozenset(), 'start') # (start_side, end_side, umbrella_pos)

    # Stack for DFS: (current_time, current_path, current_config)
    stack = [(0, [initial_state], initial_state)]
    
    # Stores the minimum time to reach a particular state (configuration)
    # Essential for pruning redundant paths in DFS when looking for an optimal solution
    visited_states = {initial_state: 0} 

    min_total_time = float('inf')
    best_path = None
    
    while stack:
        current_time, path, current_config = stack.pop()
        
        start_side, end_side, umbrella_pos = current_config

        # If all people are on the end side, we found a complete path.
        if not start_side:
            if current_time < min_total_time and current_time <= time_limit:
                min_total_time = current_time
                best_path = path
            continue # Continue searching for potentially better (shorter) paths

        # Pruning: If current path time already exceeds the best found so far or the time limit
        if current_time >= min_total_time or current_time > time_limit:
            continue
        
        # Pruning: If we've visited this state with a shorter or equal time, don't re-explore this path
        if current_time > visited_states.get(current_config, float('inf')):
             continue

        # Determine which side the umbrella is on and from where people will move
        if umbrella_pos == 'start':
            move_from = start_side
            move_to_pos = 'end'
        else:
            move_from = end_side
            move_to_pos = 'start'

        possible_moves = []
        for num_people in [1, 2]:
            if len(move_from) < num_people:
                continue
            
            for group in combinations(move_from, num_people):
                group = frozenset(group)
                trip_time = max(crossing_times[person] for person in group)
                new_total_time = current_time + trip_time
                
                # Pruning: Don't explore if already over time limit
                if new_total_time > time_limit: 
                    continue

                # Calculate the new configuration
                if umbrella_pos == 'start':
                    new_start_side = start_side - group
                    new_end_side = end_side | group
                else:
                    new_start_side = start_side | group
                    new_end_side = end_side - group
                
                new_config = (new_start_side, new_end_side, move_to_pos)

                # Only add to stack if it's a new state or a shorter path to an existing state
                if new_config not in visited_states or new_total_time < visited_states[new_config]:
                    visited_states[new_config] = new_total_time
                    possible_moves.append((new_total_time, path + [new_config], new_config))
        
        # Push moves onto the stack. Sorting helps prioritize exploring paths that look promising
        # (i.e., shorter time). Since it's a stack (LIFO), we push the shortest-time move last
        # so it gets processed first.
        for move in sorted(possible_moves, key=lambda x: x[0], reverse=True): 
            stack.append(move)

    return min_total_time if min_total_time != float('inf') else None, best_path, visited_states

# --- Utility Function for Printing Solutions ---
def print_bridge_solution(total_time, path, times, visited_times_map, algorithm_name):
    """
    Prints the steps of the bridge crossing solution.
    """
    print(f"\n--- {algorithm_name} Solution ---")
    if path:
        print(f"🎉 Solution found! Total time: {total_time} minutes.")
        print("-" * 30)
        for i in range(len(path) - 1):
            start_config = path[i]
            end_config = path[i+1]
            
            # The time at each step is the cumulative time to reach 'end_config'
            time_at_step = visited_times_map[end_config] 
            # The time for this specific trip
            trip_time = time_at_step - visited_times_map[start_config]
            
            # Determine which people moved and in which direction
            if start_config[2] == 'start': # Umbrella was on the start side
                moved_people = start_config[0] - end_config[0] # People who left the start side
                direction = "-->" # Moving from start to end
            else: # Umbrella was on the end side
                moved_people = end_config[0] - start_config[0] # People who arrived back at start side
                direction = "<--" # Moving from end to start
            
            print(f"Step {i+1}: {', '.join(sorted(list(moved_people)))} cross {direction} ({trip_time} min)")
            print(f"  > Time elapsed: {time_at_step} min")
            print(f"  > Start side: {sorted(list(end_config[0]))}")
            print(f"  > End side:   {sorted(list(end_config[1]))}")
            print("-" * 30)
    else:
        print("No solution could be found within the time limit.")

# --- Main Execution Block ---
if __name__ == "__main__":
    CROSSING_TIMES = {
        'Amogh': 5,
        'Ameya': 10,
        'Grandmother': 20,
        'Grandfather': 25
    }
    TIME_LIMIT = 60 # minutes

    # Solve using BFS (Dijkstra's)
    total_time_bfs, solution_path_bfs, visited_times_bfs_map = solve_bridge_problem_bfs(CROSSING_TIMES, TIME_LIMIT)
    print_bridge_solution(total_time_bfs, solution_path_bfs, CROSSING_TIMES, visited_times_bfs_map, "BFS (Dijkstra's)")

    # Solve using DFS
    total_time_dfs, solution_path_dfs, visited_times_dfs_map = solve_bridge_problem_dfs(CROSSING_TIMES, TIME_LIMIT)
    # DFS usually finds *a* path first, then optimizes for the best path.
    # The `visited_states_dfs_map` is crucial for the DFS to find the shortest path,
    # as it prunes paths that lead to an already-visited state with a longer time.
    print_bridge_solution(total_time_dfs, solution_path_dfs, CROSSING_TIMES, visited_times_dfs_map, "DFS (with Optimization)")
