
from collections import defaultdict

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
        
    def addEdge(self,u,v):
        self.graph[u].append(v)
    
    def addNode(self,v,visited):
        visited.add(v)
        print(v, end=" ")
        for neighbour in self.graph[v]:
            if neighbour not in visited:
                self.addNode(neighbour,visited)
    
    def DFS(self,v):
        visited = set()
        self.addNode(v,visited)
        
        
if __name__ == "__main__":
    g = Graph()
    g.addEdge(1,2)
    g.addEdge(1,5)
    g.addEdge(1,3)
    g.addEdge(2,4)
    g.addEdge(2,7)
    g.addEdge(3,5)
    g.addEdge(3,6)
    g.addEdge(4,8)
    g.addEdge(5,7)
    g.addEdge(6,9)
    g.addEdge(7,10)
    g.addEdge(8,10)
    g.addEdge(9,10)
  
    g.DFS(1)
    
