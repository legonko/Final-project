import utils


class Graph(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k in self:
            self[k] = list(set(self[k]))
        self.tree = dict()


    def get_edges(self):    
        edges = []
        for n in self:
            for neighbor in self[n]:
                edge = tuple(sorted([n, neighbor[0]]))
                edges.append(edge)
        return list(set(edges))
    

    def remove_edge(self, vert1, vert2, hold=False):
        if vert1 in self:
            for n in self[vert1]:
                if vert2 == n[0]:
                    self[vert1].remove(n)
                    if (not self[vert1]) and (not hold):
                        self.remove_vertex(vert1)
                    break
        
        if vert2 in self:
            for n in self[vert2]:
                if vert1 == n[0]:
                    self[vert2].remove(n)
                    if not (not self[vert2]) and (not hold):
                        self.remove_vertex(vert2)
                    break
            
    
    def add_edge(self, vert1, vert2):
        self[vert1] = list(set((self.get(vert1, list()) + [(vert2, utils.get_dist(vert1, vert2))])))
        # self[vert2] = list(set((self.get(vert2, list()) + [(vert1, utils.get_dist(vert1, vert2))])))


    def add_vert(self, vert):
        self.setdefault(vert, list())
 
    
    def remove_vertex(self, vertex):
        if vertex not in self:
            return
        rm_vertices = []
        for n in self[vertex]:
            rm_vertices.append(n[0])
        for v in rm_vertices:
            self.remove_edge(v, vertex)
        if vertex in self:
            self.pop(vertex)

    def get_weight(self, vert1, vert2):
        for v in self[vert1]:
            if v[0] == vert2:
                return v[1]    