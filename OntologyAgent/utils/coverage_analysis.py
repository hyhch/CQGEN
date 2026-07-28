"""
覆盖率分析工具
用于分析能力问题对本体子图实体的覆盖率
"""
import json
import os
import sys

def extract_tbox_classes(triples):
    """
    从三元组中提取TBox中的类（owl:Class）
    """
    classes = set()
    for triple in triples:
        # 查找 rdf:type owl:Class 的主语
        if triple[1] == 'rdf:type' and triple[2] == 'owl:Class':
            classes.add(triple[0])
    return classes

def extract_all_entities(triples):
    """
    提取所有实体（主语和宾语）
    """
    entities = set()
    for triple in triples:
        entities.add(triple[0])
        entities.add(triple[2])
    return entities

def calculate_coverage(chunk_info, mode='classes'):
    """
    计算单个子图的实体覆盖率
    
    mode:
        'classes' - 仅统计owl:Class（TBox类）
        'all' - 统计所有实体（原逻辑）
    """
    triples = chunk_info.get("triples", [])
    
    if mode == 'classes':
        # 仅统计类
        all_entities = extract_tbox_classes(triples)
    else:
        # 统计所有实体
        all_entities = extract_all_entities(triples)
    
    total_entities = len(all_entities)
    uncovered_entities = set(chunk_info.get("uncovered_entities", []))
    
    # 只保留属于我们统计范围的未覆盖实体
    uncovered_entities = uncovered_entities & all_entities
    
    covered_entities = total_entities - len(uncovered_entities)
    coverage_rate = (covered_entities / total_entities * 100) if total_entities > 0 else 0
    
    return {
        "total_entities": total_entities,
        "covered_entities": covered_entities,
        "uncovered_entities_count": len(uncovered_entities),
        "coverage_rate": coverage_rate,
        "uncovered_entities_list": list(uncovered_entities)
    }

def analyze_results_file(file_path, mode='classes'):
    """
    分析单个results.json文件的覆盖率
    
    mode:
        'classes' - 仅统计owl:Class（TBox类）【推荐】
        'all' - 统计所有实体
    """
    mode_desc = "TBox类(owl:Class)" if mode == 'classes' else "所有实体"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    subgraph_chunks = data.get("subgraph_chunks", [])
    
    print(f"\n{'='*80}")
    print(f"分析文件: {file_path} [{mode_desc}]")
    print(f"{'='*80}")
    
    # 计算全局唯一实体数
    all_triples = []
    for chunk in subgraph_chunks:
        all_triples.extend(chunk.get('triples', []))
    
    if mode == 'classes':
        all_entities = extract_tbox_classes(all_triples)
    else:
        all_entities = extract_all_entities(all_triples)
    
    # 收集未覆盖实体
    all_uncovered = set()
    for chunk in subgraph_chunks:
        for entity in chunk.get('uncovered_entities', []):
            if entity in all_entities:
                all_uncovered.add(entity)
    
    total_entities = len(all_entities)
    uncovered_count = len(all_uncovered)
    covered_count = total_entities - uncovered_count
    overall_coverage_rate = (covered_count / total_entities * 100) if total_entities > 0 else 0
    
    # 显示每个子图的统计
    for idx, chunk in enumerate(subgraph_chunks):
        coverage_stats = calculate_coverage(chunk, mode=mode)
        
        print(f"\n子图 {idx + 1}:")
        print(f"  - 三元组数: {len(chunk.get('triples', []))}")
        print(f"  - 总{mode_desc}数: {coverage_stats['total_entities']}")
        print(f"  - 已覆盖数: {coverage_stats['covered_entities']}")
        print(f"  - 未覆盖数: {coverage_stats['uncovered_entities_count']}")
        print(f"  - 覆盖率: {coverage_stats['coverage_rate']:.2f}%")
        print(f"  - 能力问题数: {len(chunk.get('competency_questions', []))}")
        print(f"  - 合格问题数: {len(chunk.get('qualified_questions', []))}")
    
    print(f"\n{'='*80}")
    print(f"总体统计 [{mode_desc}]:")
    print(f"  - 子图数量: {len(subgraph_chunks)}")
    print(f"  - 总数: {total_entities}")
    print(f"  - 已覆盖数: {covered_count}")
    print(f"  - 未覆盖数: {uncovered_count}")
    print(f"  - 总体覆盖率: {overall_coverage_rate:.2f}%")
    print(f"  - 总能力问题数: {len(data.get('retrofitted_competency_questions', []))}")
    print(f"{'='*80}\n")
    
    return {
        "total_subgraphs": len(subgraph_chunks),
        "total_entities": total_entities,
        "total_covered_entities": covered_count,
        "overall_coverage_rate": overall_coverage_rate
    }

def analyze_dataset(dataset_path, dataset_name, mode='classes'):
    """
    分析数据集中的所有results文件
    
    mode:
        'classes' - 仅统计owl:Class（TBox类）【推荐】
        'all' - 统计所有实体
    """
    results_file = os.path.join(dataset_path, dataset_name, "results.json")
    
    if not os.path.exists(results_file):
        print(f"文件不存在: {results_file}")
        return None
    
    return analyze_results_file(results_file, mode=mode)

def compare_multiple_results(dataset_path, dataset_names):
    """
    比较多个数据集的覆盖率
    """
    print(f"\n{'='*80}")
    print(f"比较多个数据集的覆盖率")
    print(f"{'='*80}\n")
    
    results_summary = []
    
    for dataset_name in dataset_names:
        results_file = os.path.join(dataset_path, dataset_name, "results.json")
        if os.path.exists(results_file):
            stats = analyze_results_file(results_file)
            results_summary.append({
                "dataset": dataset_name,
                "stats": stats
            })
    
    # 输出比较表格
    print(f"\n{'='*80}")
    print(f"数据集覆盖率对比:")
    print(f"{'='*80}")
    print(f"{'数据集':<20} {'子图数':<10} {'总实体':<10} {'覆盖实体':<10} {'覆盖率':<10}")
    print(f"{'-'*80}")
    
    for item in results_summary:
        stats = item["stats"]
        print(f"{item['dataset']:<20} {stats['total_subgraphs']:<10} {stats['total_entities']:<10} "
              f"{stats['total_covered_entities']:<10} {stats['overall_coverage_rate']:<10.2f}%")
    
    print(f"{'='*80}\n")

def compare_llm_results(dataset_path, dataset_name, llm_names=None, mode='classes'):
    """
    比较同一数据集在不同LLM下的覆盖率表现
    
    mode:
        'classes' - 仅统计owl:Class（TBox类）【推荐】
        'all' - 统计所有实体
    """
    if llm_names is None:
        llm_names = ["gpt4o", "llama33", "qwen"]
    
    mode_desc = "TBox类(owl:Class)" if mode == 'classes' else "所有实体"
    
    print(f"\n{'='*80}")
    print(f"数据集 '{dataset_name}' 在不同LLM下的覆盖率对比 [{mode_desc}]")
    print(f"{'='*80}\n")
    
    results_summary = []
    
    for llm_name in llm_names:
        results_file = os.path.join(dataset_path, dataset_name, f"results_{llm_name}.json")
        if os.path.exists(results_file):
            print(f"\n--- LLM: {llm_name.upper()} ---")
            
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            subgraph_chunks = data.get("subgraph_chunks", [])
            
            # 计算全局唯一实体数（避免重复计数）
            all_triples = []
            for chunk in subgraph_chunks:
                all_triples.extend(chunk.get('triples', []))
            
            if mode == 'classes':
                all_entities = extract_tbox_classes(all_triples)
            else:
                all_entities = extract_all_entities(all_triples)
            
            # 收集未覆盖实体（只统计属于我们范围的）
            all_uncovered = set()
            for chunk in subgraph_chunks:
                for entity in chunk.get('uncovered_entities', []):
                    if entity in all_entities:
                        all_uncovered.add(entity)
            
            total_entities = len(all_entities)
            uncovered_count = len(all_uncovered)
            covered_count = total_entities - uncovered_count
            overall_coverage_rate = (covered_count / total_entities * 100) if total_entities > 0 else 0
            
            stats = {
                "llm": llm_name.upper(),
                "total_subgraphs": len(subgraph_chunks),
                "total_entities": total_entities,
                "total_covered_entities": covered_count,
                "overall_coverage_rate": overall_coverage_rate,
                "total_questions": len(data.get('retrofitted_competency_questions', []))
            }
            
            results_summary.append(stats)
            
            print(f"  子图数量: {stats['total_subgraphs']}")
            print(f"  总{mode_desc}数: {stats['total_entities']}")
            print(f"  已覆盖数: {stats['total_covered_entities']}")
            print(f"  总体覆盖率: {stats['overall_coverage_rate']:.2f}%")
            print(f"  生成的能力问题数: {stats['total_questions']}")
        else:
            print(f"\n文件不存在: {results_file}")
    
    # 输出对比表格
    if results_summary:
        print(f"\n{'='*80}")
        print(f"LLM覆盖率对比表 [{mode_desc}]:")
        print(f"{'='*80}")
        print(f"{'LLM':<12} {'子图数':<8} {'总数':<10} {'覆盖数':<10} {'覆盖率':<12} {'问题数':<10}")
        print(f"{'-'*80}")
        
        for stats in results_summary:
            print(f"{stats['llm']:<12} {stats['total_subgraphs']:<8} {stats['total_entities']:<10} "
                  f"{stats['total_covered_entities']:<10} {stats['overall_coverage_rate']:<12.2f}% {stats['total_questions']:<10}")
        
        # 找出最佳LLM
        best_llm = max(results_summary, key=lambda x: x['overall_coverage_rate'])
        print(f"\n🏆 最佳覆盖率: {best_llm['llm']} ({best_llm['overall_coverage_rate']:.2f}%)")
        print(f"{'='*80}\n")
    
    return results_summary

def compare_all_datasets_all_llms(dataset_path, dataset_names=None, llm_names=None, mode='classes'):
    """
    比较所有数据集在所有LLM下的覆盖率表现（综合对比）
    
    mode:
        'classes' - 仅统计owl:Class（TBox类）【推荐】
        'all' - 统计所有实体
    """
    if dataset_names is None:
        dataset_names = ["demcare", "saref4env", "vicinitycore", "videogameontology", "onem2m"]
    if llm_names is None:
        llm_names = ["gpt4o", "llama33", "qwen"]
    
    mode_desc = "TBox类(owl:Class)" if mode == 'classes' else "所有实体"
    
    print(f"\n{'='*80}")
    print(f"所有数据集 × 所有LLM 覆盖率综合对比 [{mode_desc}]")
    print(f"{'='*80}\n")
    
    # 收集所有数据
    all_results = {}
    
    for dataset_name in dataset_names:
        all_results[dataset_name] = {}
        for llm_name in llm_names:
            results_file = os.path.join(dataset_path, dataset_name, f"results_{llm_name}.json")
            if os.path.exists(results_file):
                with open(results_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                subgraph_chunks = data.get("subgraph_chunks", [])
                
                # 计算全局唯一实体数
                all_triples = []
                for chunk in subgraph_chunks:
                    all_triples.extend(chunk.get('triples', []))
                
                if mode == 'classes':
                    all_entities = extract_tbox_classes(all_triples)
                else:
                    all_entities = extract_all_entities(all_triples)
                
                # 收集未覆盖实体
                all_uncovered = set()
                for chunk in subgraph_chunks:
                    for entity in chunk.get('uncovered_entities', []):
                        if entity in all_entities:
                            all_uncovered.add(entity)
                
                total_entities = len(all_entities)
                uncovered_count = len(all_uncovered)
                covered_count = total_entities - uncovered_count
                overall_coverage_rate = (covered_count / total_entities * 100) if total_entities > 0 else 0
                
                all_results[dataset_name][llm_name] = {
                    "coverage_rate": overall_coverage_rate,
                    "covered_entities": covered_count,
                    "total_entities": total_entities,
                    "questions": len(data.get('retrofitted_competency_questions', []))
                }
    
    # 输出综合对比表格
    print(f"{'数据集':<20} ", end="")
    for llm_name in llm_names:
        print(f"{llm_name.upper():<15}", end="")
    print()
    print(f"{'-'*80}")
    
    for dataset_name in dataset_names:
        if dataset_name in all_results and all_results[dataset_name]:
            print(f"{dataset_name:<20} ", end="")
            for llm_name in llm_names:
                if llm_name in all_results[dataset_name]:
                    rate = all_results[dataset_name][llm_name]["coverage_rate"]
                    print(f"{rate:>6.2f}%        ", end="")
                else:
                    print(f"{'N/A':<15}", end="")
            print()
    
    # 计算每个LLM的平均覆盖率
    print(f"\n{'='*80}")
    print(f"各LLM平均覆盖率 [{mode_desc}]:")
    print(f"{'='*80}")
    
    for llm_name in llm_names:
        rates = []
        for dataset_name in dataset_names:
            if dataset_name in all_results and llm_name in all_results[dataset_name]:
                rates.append(all_results[dataset_name][llm_name]["coverage_rate"])
        
        if rates:
            avg_rate = sum(rates) / len(rates)
            print(f"  {llm_name.upper():<12}: {avg_rate:.2f}% (基于 {len(rates)} 个数据集)")
    
    print(f"{'='*80}\n")
    
    return all_results

if __name__ == "__main__":
    # 示例用法
    dataset_path = "./dataset"
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        # 分析单个数据集
        if command not in ["llm", "all"]:
            dataset_name = command
            mode = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] in ['classes', 'all'] else 'classes'
            analyze_dataset(dataset_path, dataset_name, mode=mode)
        # 比较单个数据集在不同LLM下的表现
        elif command == "llm" and len(sys.argv) > 2:
            dataset_name = sys.argv[2]
            mode = 'classes'
            llm_names = []
            for arg in sys.argv[3:]:
                if arg in ['classes', 'all']:
                    mode = arg
                else:
                    llm_names.append(arg)
            compare_llm_results(dataset_path, dataset_name, llm_names if llm_names else None, mode=mode)
        # 综合对比所有数据集和所有LLM
        elif command == "all":
            mode = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] in ['classes', 'all'] else 'classes'
            compare_all_datasets_all_llms(dataset_path, mode=mode)
        else:
            print("用法:")
            print("  python coverage_analysis.py <dataset_name> [classes|all]           # 分析单个数据集")
            print("  python coverage_analysis.py llm <dataset_name> [llm1 llm2...] [classes|all]  # 比较LLM")
            print("  python coverage_analysis.py all [classes|all]                      # 综合对比")
            print("\n  mode参数:")
            print("    classes (默认) - 仅统计TBox类(owl:Class)覆盖率")
            print("    all           - 统计所有实体覆盖率")
    else:
        # 默认：综合对比所有数据集和所有LLM，使用classes模式
        compare_all_datasets_all_llms(dataset_path, mode='classes')
