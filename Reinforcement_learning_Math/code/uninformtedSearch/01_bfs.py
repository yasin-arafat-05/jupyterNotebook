
class Node:
    def __init__(self,val):
        self.val = val 
        self.left = None 
        self.right = None 
    
n = Node(1)
n.left = Node(2)
n.right = Node(3)
n.left.left = Node(4)

#""" 
#************Output Tree****************
#        1 
#       / \
#      /   \ 
#     2     3
#    /
#   / 
#  4
#***************************************
#"""
# BFS
def BFS(root):
    queue = []
    visited = []
    queue.append(root)
    while(len(queue)!=0):
        node = queue.pop(0)
        visited.append(node.val)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)

    return visited

print(BFS(root=n))


