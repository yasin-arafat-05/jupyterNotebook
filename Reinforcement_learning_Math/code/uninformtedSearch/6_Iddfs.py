
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
            if neighbour not in visited:
                self.addNode(neighbour,visited,depth+1,depth_limit)

        if depth==0:
            print(f"\n finishing visiting: {depth_limit}\n")
            
    def DLS(self,v,target,depth_limit):
        visited = set()
        self.addNode(v,visited,0,depth_limit)
        return target in visited
    
    
    def IDDFS(self,start,target,maxDepth):
        for i in range(maxDepth+1):
            if self.DLS(start,target,i):
                return True 
        return False 
    
    
if __name__ == "__main__":
    g = Graph()
    g.addEdge(1,2)
    g.addEdge(1,9)
    g.addEdge(1,10)
    g.addEdge(2,3)
    g.addEdge(2,4)
    g.addEdge(4,5)
    g.addEdge(4,6)
    g.addEdge(4,7)
    g.addEdge(5,8)
    
    target = 8 
    maxdepth = 4 
    start = 1 
    if g.IDDFS(start,target,maxdepth):
        print("target found")
    else: 
        print("target not found")
        

    

