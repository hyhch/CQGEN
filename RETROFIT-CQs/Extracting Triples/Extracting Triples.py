import csv
from deeponto.onto import Ontology
from deeponto.onto.projection import OntologyProjector

onto = Ontology('../Data/Ontologies/vicinitycore.owl',ignore_failed_imports=True)  # Load the ontology file
projector = OntologyProjector(bidirectional_taxonomy=False, only_taxonomy=True, include_literals=True)

# Project the ontology into triples
projected_ontology = projector.project(onto)

#--- A function that only returns the concepts of s, p, o
def con(c):
  return c.rsplit('#')[-1]


def save_to_csv(data):
     # Specify the desired file path and name
    file_path ='../Data/ExtractingTriples/saref4env.csv'
    with open(file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter='\t') # the s, p, o saves in the csv file and seperated by space no commas if i want comma delete the delimiter
        writer.writerows(data)


for subj, pred, obj in projected_ontology:
    # Check if there is at least one triple in the Graph
    if (subj, pred, obj) not in projected_ontology:
       raise Exception("It better be!")

#--- Print the number of "triples" in the Graph
print(f"Graph g has {len(projected_ontology)} statements.")
print(projected_ontology)

# extract s, p, o and save it to the list to be saved in csv file
data= []
for s,p,o in projected_ontology:
  sub = con(s)  
  pre = con(p) 
  obj = con(o)  
  data.append([sub, pre, obj])
  save_to_csv(data) 
