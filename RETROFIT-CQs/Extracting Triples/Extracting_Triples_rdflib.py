import csv
from rdflib import Graph
import re


# Load the ontology file using rdflib
ontology_path = '../Data/Ontologies/base_ontology.owl'
g = Graph()
g.parse(ontology_path, format='xml')  # Adjust format if needed (e.g., 'ttl' for Turtle)

# Function to extract the local name of a URI
def con(c):
    return c.split('#')[-1] if '#' in c else c.split('/')[-1]

# Function to save triples to a CSV file
def save_to_csv(data):
    file_path = '../Data/ExtractingTriples/base_ontology.csv'
    with open(file_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter='\t')  # Use tab as delimiter
        writer.writerows(data)
# Function to check if a string is valid (not random or unreadable)
def is_valid(value):
    # Check if the value is alphanumeric, does not contain digits, and is not too short
    return bool(re.match(r'^[a-zA-Z_]+$', value)) and len(value) > 2


# Extract triples and save them to a list
data = []
for subj, pred, obj in g:
    sub = con(str(subj))
    pre = con(str(pred))
    obj = con(str(obj))
    # Skip triples where any part is not recognizable or invalid
    if not sub or not pre or not obj or not (is_valid(sub) and is_valid(pre) and is_valid(obj)):
        continue
    
    data.append([sub, pre, obj])

# Save the extracted triples to a CSV file
save_to_csv(data)

# Print the number of triples in the graph
print(f"Graph g has {len(g)} statements.")
print(data)