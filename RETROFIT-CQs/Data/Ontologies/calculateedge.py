from rdflib import Graph, RDF, OWL

# 加载本体
g = Graph()
g.parse('vicinitycore.owl', format='xml')

# 统计实体（Class）数量，并打印所有Class
classes = set(g.subjects(RDF.type, OWL.Class))
class_count = len(classes)
print(f"实体（Class）数量: {class_count}")
print("所有实体（Class）:")
for c in classes:
    print(c)

# 统计边（ObjectProperty + DatatypeProperty）数量，并打印所有边
object_properties = set(g.subjects(RDF.type, OWL.ObjectProperty))
datatype_properties = set(g.subjects(RDF.type, OWL.DatatypeProperty))
edge_count = len(object_properties) + len(datatype_properties)
print(f"边（ObjectProperty + DatatypeProperty）数量: {edge_count}")
print("所有 ObjectProperty:")
for op in object_properties:
    print(op)
print("所有 DatatypeProperty:")
for dp in datatype_properties:
    print(dp)