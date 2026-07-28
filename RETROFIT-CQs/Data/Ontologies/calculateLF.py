from rdflib import Graph, URIRef, RDF, OWL
import asyncio

class LoadOWL:
    """
    解析 OWL/RDF/TTL 格式的本体文件，提取命名空间和三元组，返回文件路径和解析结果的字典
    """
    name: str = "LoadOWL"
    async def run(self, file_path: str) -> dict:
        g = Graph()
        if file_path.endswith(".owl") or file_path.endswith(".rdf"):
            g.parse(file_path, format='xml')
        elif file_path.endswith(".ttl"):
            g.parse(file_path, format='turtle')
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        namespaces = []
        triples = []
        for prefix, url in g.namespaces():
            namespaces.append({"prefix": prefix, "url": url})
        for s, p, o in g:
            if isinstance(s, URIRef) and isinstance(p, URIRef) and isinstance(o, URIRef):
                triples.append(
                    (
                        str(s.n3(namespace_manager=g.namespace_manager)),
                        str(p.n3(namespace_manager=g.namespace_manager)),
                        str(o.n3(namespace_manager=g.namespace_manager))
                    )
                )
        # 统计Class数量
        class_entities = set(g.subjects(RDF.type, OWL.Class))
        return {"namespaces": namespaces, "triples": triples, "class_entities": class_entities}

# 直接运行入口
if __name__ == "__main__":
    loader = LoadOWL()
    result = asyncio.run(loader.run("base_ontology.owl"))
    print("命名空间：")
    for ns in result["namespaces"]:
        print(ns)
    print(f"三元组数量: {len(result['triples'])}")
    print("前10个三元组示例：")
    for t in result["triples"][:10]:
        print(t)
    print(f"实体（Class）总数: {len(result['class_entities'])}")
    print("所有实体（Class）:")
    for c in result['class_entities']:
        print(c)