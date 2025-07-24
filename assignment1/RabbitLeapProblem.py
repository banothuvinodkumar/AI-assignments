from collections import deque

def get_successors(state):
    successors = []
    s = list(state)
    n = len(s)
    
    try:
        empty_index = s.index('_')
    except ValueError:
        return []

    if empty_index > 0 and s[empty_index - 1] == 'E':
        new_s = s[:]
        new_s[empty_index], new_s[empty_index - 1] = new_s[empty_index - 1], new_s[empty_index]
        successors.append(tuple(new_s))

    if empty_index > 1 and s[empty_index - 2] == 'E':
        new_s = s[:]
        new_s[empty_index], new_s[empty_index - 2] = new_s[empty_index - 2], new_s[empty_index]
        successors.append(tuple(new_s))

    if empty_index < n - 1 and s[empty_index + 1] == 'W':
        new_s = s[:]
        new_s[empty_index], new_s[empty_index + 1] = new_s[empty_index + 1], new_s[empty_index]
        successors.append(tuple(new_s))

    if empty_index < n - 2 and s[empty_index + 2] == 'W':
        new_s = s[:]
        new_s[empty_index], new_s[empty_index + 2] = new_s[empty_index + 2], new_s[empty_index]
        successors.append(tuple(new_s))
        
    return successors

def solve_with_bfs(initial_state, goal_state):
    queue = deque([(initial_state, [initial_state])])
    visited = {initial_state}

    while queue:
        current_state, path = queue.popleft()

        if current_state == goal_state:
            return path
        
        for successor in get_successors(current_state):
            if successor not in visited:
                visited.add(successor)
                new_path = path + [successor]
                queue.append((successor, new_path))
    
    return None

def solve_with_dfs(initial_state, goal_state):
    stack = [(initial_state, [initial_state])]
    visited = {initial_state}

    while stack:
        current_state, path = stack.pop()

        if current_state == goal_state:
            return path
        
        for successor in get_successors(current_state):
            if successor not in visited:
                visited.add(successor)
                new_path = path + [successor]
                stack.append((successor, new_path))
                
    return None

def print_path(path):
    if path:
        print(f"Solution found in {len(path) - 1} moves:")
        for i, state in enumerate(path):
            print(f"Step {i}: {' '.join(state)}")
    else:
        print("No solution found.")

if __name__ == "__main__":
    initial = ('E', 'E', 'E', '_', 'W', 'W', 'W')
    goal = ('W', 'W', 'W', '_', 'E', 'E', 'E')

    print("--- Solving Rabbit Leap Problem ---")
    
    print("\n🔎 Searching with Breadth-First Search (BFS)...")
    bfs_path = solve_with_bfs(initial, goal)
    print_path(bfs_path)

    print("\n" + "="*40 + "\n")

    print("🔎 Searching with Depth-First Search (DFS)...")
    dfs_path = solve_with_dfs(initial, goal)
    print_path(dfs_path)