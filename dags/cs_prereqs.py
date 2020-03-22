from graphviz import Digraph

dot = Digraph(comment='CS Course Prerequisites')

adj_list = {
    'Stat 110': [],
    'CS 20': [],
    'CS 50': [],
    'CS 51': ['CS 50'],
    'CS 61': ['CS 50'],
    'CS 121': ['CS 20'],
    'CS 124': ['CS 50', 'CS 51', 'CS 121', 'Stat 110'],
    'CS 182': ['CS 51', 'CS 121']
}

for course in adj_list.keys():
    for prereq in adj_list[course]:
        dot.edge(prereq, course)

dot.render('cs_prereqs.png', view=True)
