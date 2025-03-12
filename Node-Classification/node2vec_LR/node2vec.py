import random
from gensim.models import Word2Vec
def main():
    graph={}

    path=["../cora_test.cites","../cora_train.cites"]
    for i in path:
        with open(i, "r") as file:
            # Read the entire contents of the file
            for line in file:
                line = line.strip()
                line = line.split('\t')
                if line[1] in graph:
                    graph[line[1]].append(line[0])
                else:
                    graph[line[1]]=[line[0]]

    walks=[]

    def generate_random_walks(G,walk_len,iter):
        graph=G
        walk_size=walk_len
        for j in range(iter):
            for i in graph:
                for node in graph[i]:
                    cur_walk=[i,node]
                    src=node
                    while(walk_size-2):
                        if src in graph:
                            random_element = random.choice(graph[src])
                            cur_walk.append(random_element)
                            walk_size=walk_size-1
                            src=random_element
                        else:
                            break
                    walk_size=walk_len
                    walks.append(cur_walk)
                    # if j==0:
                    #     walks.append(cur_walk)
                    # else:
                    #     if len(cur_walk) > 3:
                    #         walks.append(cur_walk)

    generate_random_walks(graph,100,20)
    #print(walks)
    with open("walks.txt",'w') as file:
        for walk in walks:
                file.write(" ".join(map(str, walk)) + "\n")

    #wordtovec
    model = Word2Vec(sentences=walks, vector_size=1024, window=5, min_count=0, workers=4,epochs=100)
    return model