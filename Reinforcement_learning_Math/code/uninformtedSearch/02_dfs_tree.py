

tree = {
    1:[2,9,10],
    2:[3,4],
    3:[],
    4:[5,6,7],
    5:[8],
    6:[],
    7:[],
    8:[],
    9:[],
    10:[]
}


def dfs(tree,start):
    stack = [start]
    visited = []
    while stack:
        # last element: in BFS we do: pop(0)
        node = stack.pop()
        visited.append(node)
        
        # vlaue to correspoing 1 -> [2,9,10]
        for child in reversed(tree[node]):
            if child not in visited and child not in stack:
                stack.append(child)
    return visited

results = dfs(tree,1)
print(results)

