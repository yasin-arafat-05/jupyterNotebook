
from collections import defaultdict


class Graph:
    def __init__(self):
       self.graph = defaultdict(list)
       
    def addEdge(self,u,v):
        self.graph[u].append(v)
        
    def addNode(self,v,visited,depth,depth_limit):
        
        if depth>depth_limit:
            return 
        
        visited.add(v)
        print(v,end=" ")
        
        for neighbour in self.graph[v]:
            if  neighbour not in visited:
                self.addNode(neighbour,visited,depth+1,depth_limit)
    
    def DLS(self,v,depth_limit):
        visited = set()
        self.addNode(v,visited,0,depth_limit)
        
if __name__ == "__main__":
    g = Graph()
    g.addEdge(1,2)
    g.addEdge(1,9)
    g.addEdge(1,10)
    g.addEdge(2,3)
    g.addEdge(2,4)
    g.addEdge(3,5)
    g.addEdge(4,5)
    g.addEdge(4,6)
    g.addEdge(4,7)
    g.addEdge(5,7)
   
    g.DLS(1,2)
    