import math
import pickle
# jaccard attempt
def semanticscore(set1: set, set2: set):
    numeratore = len(set1.intersection(set2))
    denominatore = len(set1.union(set2)) # se è apari a 0 significa che nessuno dei due ha nulla
    if denominatore == 0:
        return 0 # arbitrariamente per denotare che entrambi sono privi di qualsiasi info
    else:
        return numeratore/denominatore

# partiamo dalle entità
with open("datasets/DBpedia15k/entity2class_dict.pkl", 'rb') as f:
    entities = pickle.load(f)
scores = {}
for key in entities.keys():
    # print(key)
    # print(entities[key])
    classes1 = entities[key]

    for key1 in entities.keys():
        if key1 != key:
            classes2 = entities[key1]
            #print()
            index = f"{key},{key1}"
            score = semanticscore(classes1, classes2)
            scores[index] = score
            print(f"Simlarità tra {key} e {key1}= {score}")
            #scores[index].add(semanticscore(classes1, classes2))

