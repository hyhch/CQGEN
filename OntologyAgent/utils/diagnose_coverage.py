"""
诊断工具：分析为什么某些LLM的覆盖率低
"""
import json
import sys

def diagnose_coverage_issue(dataset_path, dataset_name, llm_name):
    """
    诊断特定数据集和LLM的覆盖率问题
    """
    results_file = f"{dataset_path}/{dataset_name}/results_{llm_name}.json"
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"文件不存在: {results_file}")
        return
    
    chunks = data.get('subgraph_chunks', [])
    
    print(f"\n{'='*80}")
    print(f"诊断报告: {dataset_name} - {llm_name.upper()}")
    print(f"{'='*80}\n")
    
    # 统计各种问题类型
    total_chunks = len(chunks)
    chunks_with_y_queries = 0  # SPARQL返回Y的子图
    chunks_with_empty_results = 0  # 所有查询都返回空的子图
    chunks_with_no_qualified = 0  # 没有合格问题的子图
    chunks_fully_covered = 0  # 100%覆盖的子图
    
    low_coverage_chunks = []  # 低覆盖率的子图（<50%）
    
    for i, chunk in enumerate(chunks):
        triples = chunk.get('triples', [])
        uncovered = chunk.get('uncovered_entities', [])
        qualified_qs = chunk.get('qualified_questions', [])
        
        # 计算实体数和覆盖率
        entities = set()
        for t in triples:
            entities.add(t[0])
            entities.add(t[2])
        
        coverage = ((len(entities) - len(uncovered)) / len(entities) * 100) if len(entities) > 0 else 0
        
        if coverage == 100:
            chunks_fully_covered += 1
        
        if coverage < 50:
            low_coverage_chunks.append({
                'id': i,
                'coverage': coverage,
                'entities': len(entities),
                'uncovered': len(uncovered),
                'qualified_qs': len(qualified_qs)
            })
        
        if len(qualified_qs) == 0:
            chunks_with_no_qualified += 1
        
        # 检查SPARQL查询结果
        y_count = 0
        empty_result_count = 0
        for q in qualified_qs:
            sparql = q.get('sparql_query', '')
            result = q.get('query_result', '')
            
            if sparql.strip() == 'Y':
                y_count += 1
            elif not result or result == '[]' or result == '{}':
                empty_result_count += 1
        
        if y_count > 0:
            chunks_with_y_queries += 1
        
        if len(qualified_qs) > 0 and empty_result_count == len(qualified_qs):
            chunks_with_empty_results += 1
    
    # 输出统计
    print(f"总子图数: {total_chunks}")
    print(f"100%覆盖的子图数: {chunks_fully_covered} ({chunks_fully_covered/total_chunks*100:.1f}%)")
    print(f"<50%覆盖的子图数: {len(low_coverage_chunks)} ({len(low_coverage_chunks)/total_chunks*100:.1f}%)")
    print()
    
    print(f"问题诊断:")
    print(f"  - 没有合格问题的子图: {chunks_with_no_qualified}")
    print(f"  - 有Y查询的子图: {chunks_with_y_queries}")
    print(f"  - 所有查询都返回空的子图: {chunks_with_empty_results}")
    print()
    
    # 主要问题分析
    print(f"主要问题:")
    if chunks_with_y_queries > 0:
        print(f"  ⚠️ {chunks_with_y_queries} 个子图包含SPARQL查询返回'Y'")
        print(f"     这表示LLM认为查询正确但本体缺少实例数据")
        print(f"     但在覆盖率计算中，这些查询没有覆盖任何实体！")
    
    if chunks_with_empty_results > 0:
        print(f"  ⚠️ {chunks_with_empty_results} 个子图的所有查询都返回空结果")
        print(f"     这些查询可能存在以下问题：")
        print(f"     1. SPARQL查询语法错误")
        print(f"     2. 查询的实体在本体中不存在")
        print(f"     3. 本体确实缺少实例数据")
    
    # 显示低覆盖率子图详情
    if low_coverage_chunks:
        print(f"\n低覆盖率子图详情（前5个）:")
        print(f"{'子图ID':<8} {'覆盖率':<10} {'实体数':<10} {'未覆盖':<10} {'合格问题':<10}")
        print(f"{'-'*60}")
        for chunk_info in sorted(low_coverage_chunks, key=lambda x: x['coverage'])[:5]:
            print(f"{chunk_info['id']+1:<8} {chunk_info['coverage']:<10.2f}% "
                  f"{chunk_info['entities']:<10} {chunk_info['uncovered']:<10} "
                  f"{chunk_info['qualified_qs']:<10}")
    
    print(f"\n{'='*80}\n")
    
    # 返回诊断结果
    return {
        'chunks_with_y_queries': chunks_with_y_queries,
        'chunks_with_empty_results': chunks_with_empty_results,
        'chunks_with_no_qualified': chunks_with_no_qualified,
        'low_coverage_chunks': len(low_coverage_chunks)
    }

def compare_llm_issues(dataset_path, dataset_name):
    """
    比较不同LLM的问题
    """
    llm_names = ['gpt4o', 'llama33', 'qwen']
    
    print(f"\n{'='*80}")
    print(f"LLM问题对比: {dataset_name}")
    print(f"{'='*80}\n")
    
    results = {}
    for llm in llm_names:
        results[llm] = diagnose_coverage_issue(dataset_path, dataset_name, llm)
    
    # 对比总结
    print(f"\n{'='*80}")
    print(f"问题对比总结:")
    print(f"{'='*80}")
    print(f"{'LLM':<12} {'Y查询子图':<15} {'空结果子图':<15} {'低覆盖子图':<15}")
    print(f"{'-'*80}")
    
    for llm in llm_names:
        if results[llm]:
            r = results[llm]
            print(f"{llm.upper():<12} {r['chunks_with_y_queries']:<15} "
                  f"{r['chunks_with_empty_results']:<15} {r['low_coverage_chunks']:<15}")
    
    print(f"\n结论:")
    print(f"  LLAMA33表现最好的可能原因：")
    print(f"    1. 生成的SPARQL查询质量更高")
    print(f"    2. 更少的'Y'查询（空结果被误判为合理）")
    print(f"    3. 查询结果更准确地覆盖了本体中的实体")

if __name__ == "__main__":
    dataset_path = "./dataset"
    
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
        llm_name = sys.argv[2] if len(sys.argv) > 2 else None
        
        if llm_name:
            diagnose_coverage_issue(dataset_path, dataset_name, llm_name)
        else:
            compare_llm_issues(dataset_path, dataset_name)
    else:
        print("用法:")
        print("  python diagnose_coverage.py <dataset_name> [llm_name]")
        print("\n示例:")
        print("  python diagnose_coverage.py vicinitycore")
        print("  python diagnose_coverage.py vicinitycore gpt4o")
