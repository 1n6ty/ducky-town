import json

class Graph:
    
    def __init__(self, path):
        self.path = path
        with open(path, 'r') as f:
            self.file = json.loads(f.read())

    def updateNeighbors(self, id, node):
        for i in range(len(self.file.Nodes)):
            if self.file.Nodes[i].id == id:
                for j in range(len(self.file.Nodes[i].neighbors)):
                    if j == 0: # Rebuild to centers
                        self.file.Nodes[i].neighbors[j] = node
                        with open(self.path, 'w') as f:
                            f.write(json.dumps(self.file, sort_keys = True))
                        return len(self.file.Nodes[i].neighbors) - 1
        return -1

    def addNode(self, newNode):
        if len([i for i in self.file.Nodes if (newNode.center[0] - i.center[0]) ** 2 + (newNode.center[1] - i.center[1]) ** 2 < 2]) == 0:
            newNode.id = len(self.file)
            self.file.Nodes.append(newNode)
            with open(self.path, 'w') as f:
                f.write(json.dumps(self.file, sort_keys = True))
            return newNode.id
        else:
            return -1
    
    def getNode(self, id, pretty = False):
        item = [i for i in self.file.Nodes if i.id == id][0]
        if pretty:
            return str(json.dumps(item, sort_keys = True))
        else:
            return item

    def getGraph(self, pretty = False):
        if pretty:
            return str(json.dumps(self.file, sort_keys = True))
        else:
            return self.file
