import heapq
from itertools import combinations

def solve_bridge_problem(crossing_times, time_limit):
    all_people = frozenset(crossing_times.keys())
    initial_state = (frozenset(all_people), frozenset(), 'start')
    
    pq = [(0, [initial_state])]
    
    visited_times = {initial_state: 0}

    while pq:
        current_time, path = heapq.heappop(pq)
        
        current_config = path[-1]
        start_side, end_side, umbrella_pos = current_config

        if not start_side and current_time <= time_limit:
            return current_time, path

        if umbrella_pos == 'start':
            move_from = start_side
            move_to_pos = 'end'
        else:
            move_from = end_side
            move_to_pos = 'start'

        for num_people in [1, 2]:
            if len(move_from) < num_people:
                continue
            
            for group in combinations(move_from, num_people):
                group = frozenset(group)
                trip_time = max(crossing_times[person] for person in group)
                new_total_time = current_time + trip_time
                
                if umbrella_pos == 'start':
                    new_start_side = start_side - group
                    new_end_side = end_side | group
                else:
                    new_start_side = start_side | group
                    new_end_side = end_side - group
                
                new_config = (new_start_side, new_end_side, move_to_pos)

                if new_config not in visited_times or new_total_time < visited_times[new_config]:
                    visited_times[new_config] = new_total_time
                    new_path = path + [new_config]
                    heapq.heappush(pq, (new_total_time, new_path))
                    
    return None, None

def print_bridge_solution(total_time, path, times):
    if path:
        print(f"🎉 Solution found! Total time: {total_time} minutes.")
        print("-" * 30)
        for i in range(len(path) - 1):
            start_config = path[i]
            end_config = path[i+1]
            
            time_at_step = visited_times[end_config]
            trip_time = time_at_step - visited_times[start_config]
            
            if start_config[2] == 'start':
                moved_people = start_config[0] - end_config[0]
                direction = "-->"
            else:
                moved_people = end_config[0] - start_config[0]
                direction = "<--"
            
            print(f"Step {i+1}: {', '.join(moved_people)} cross {direction} ({trip_time} min)")
            print(f"  > Time elapsed: {time_at_step} min")
            print(f"  > Start side: {sorted(list(end_config[0]))}")
            print(f"  > End side:   {sorted(list(end_config[1]))}")
            print("-" * 30)
    else:
        print("No solution could be found within the time limit.")

if __name__ == "__main__":
    CROSSING_TIMES = {
        'Amogh': 5,
        'Ameya': 10,
        'Grandmother': 20,
        'Grandfather': 25
    }
    TIME_LIMIT = 60

    print("\n--- Solving Bridge Crossing Problem ---")
    
    all_people = frozenset(CROSSING_TIMES.keys())
    initial_config = (all_people, frozenset(), 'start')
    visited_times = {initial_config: 0}

    total_time, solution_path = solve_bridge_problem(CROSSING_TIMES, TIME_LIMIT)
    print_bridge_solution(total_time, solution_path, CROSSING_TIMES)