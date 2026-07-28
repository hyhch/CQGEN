#!/usr/bin/env python3
"""LLM4KE: LLM-based Competency Question Generation & Evaluation for Ontologies.

Usage:
  # Generate CQs
  python llm4ke.py generate Odeuropa --task all_classes --llm qwen

  # Evaluate generated CQs
  python llm4ke.py evaluate Odeuropa --threshold 0.6

  # Generate + evaluate in one step
  python llm4ke.py run Odeuropa --task all_classes --llm qwen

  # Evaluate with simple/complex classification stats
  python llm4ke.py evaluate Odeuropa --labels src/CQs_labeled.xlsx

  # Evaluate all ontologies
  python llm4ke.py evaluate all
"""

import argparse
import json
import logging
import os

from src.ontology import (
    load_ontology, extract_classes, extract_properties, extract_schema,
    simplify, select_in_batches, load_ground_truth, load_description,
    props_for_classes, schema_for_classes,
)
from src.LocalTemplate import LocalTemplate
from src.llm_backend import create_llm, generate_cqs, parse_cqs
from src.evaluator import evaluate_ontology, evaluate_all

log = logging.getLogger(__name__)

# Task name → output subdirectory name (matching existing data_out structure)
TASK_TO_MODE = {
    'all_classes': 'all_classes',
    'all_classes+properties': 'classes_and_properties',
    'logic': 'logic',
}


def cmd_generate(args):
    """Generate competency questions for an ontology."""
    input_path = os.path.join(args.data_dir, args.ontology)

    # Load ontology
    graph = load_ontology(input_path)
    classes = extract_classes(graph)
    properties = extract_properties(graph)
    schema = extract_schema(graph, properties)
    description = load_description(input_path) if args.include_description else ''

    log.info("Ontology %s: %d classes, %d properties", args.ontology, len(classes), len(properties))

    # Load prompt template
    template_path = os.path.join('src', 'prompt_templates', f'{args.task}.yml')
    prompt_template = LocalTemplate.load(template_path)

    # Build batches
    classes_batches = [[simplify(c) for c in batch] for batch in select_in_batches(classes)]
    property_batches = [props_for_classes(schema, batch) for batch in classes_batches]
    schema_batches = [schema_for_classes(schema, batch) for batch in classes_batches]

    # Load examples if requested
    examples = ''
    if args.n_examples > 0:
        gt_cqs = load_ground_truth(input_path)
        examples = 'For example:\n -' + '\n -'.join(gt_cqs[:args.n_examples])

    # Build input batches for the prompt
    input_batches = []
    for c_batch, p_batch, s_batch in zip(classes_batches, property_batches, schema_batches):
        ont_input = {
            'name': args.ontology,
            'description': description,
            'n': args.n_cqs,
            'classes': '\n- '.join(c_batch),
            'properties': '\n- '.join(p_batch),
            'schema': '\n- '.join(f'({s}, {p}, {o})' for s, p, o in s_batch),
            'examples': examples,
        }
        input_batches.append({k: ont_input[k] for k in prompt_template.input})

    # Call LLM
    llm = create_llm(args.llm, args.config)
    log.info("Calling LLM: %s", args.llm)
    raw_responses = generate_cqs(llm, prompt_template, input_batches)
    cqs = parse_cqs(raw_responses)
    log.info("Generated %d CQs", len(cqs))

    # Save output — auto-aligned to evaluation directory structure
    mode_dir = TASK_TO_MODE.get(args.task, args.task)
    output_dir = os.path.join(args.output_dir, args.ontology, mode_dir)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f'{args.ontology}_{args.llm}_{args.n_examples}.txt')
    with open(output_file, 'w') as f:
        f.write('\n'.join(cqs))
    log.info("Saved CQs to %s", output_file)

    # Save raw LLM responses for traceability
    raw_file = os.path.join(output_dir, f'{args.ontology}_{args.llm}_{args.n_examples}_raw.json')
    with open(raw_file, 'w') as f:
        json.dump(raw_responses, f, indent=2, ensure_ascii=False)
    log.info("Raw responses saved to %s", raw_file)

    print(f"Generated {len(cqs)} CQs -> {output_file}")
    return cqs


def cmd_evaluate(args):
    """Evaluate generated CQs against ground truth."""
    labels_path = getattr(args, 'labels', None)

    if args.ontology == 'all':
        results = evaluate_all(args.threshold, args.output_dir, args.data_dir, labels_path)
    else:
        results = evaluate_ontology(args.ontology, args.threshold, args.output_dir, args.data_dir, labels_path)

    # Save results
    result_file = f'results_{args.ontology}.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log.info("Results saved to %s", result_file)

    # Print summary
    print(f"\nEvaluation results ({len(results)} files):")
    for r in results:
        line = (f"  {r['onto']}/{r['mode']}/{r['llm']}({r['examples']}ex): "
                f"P={r['precision']:.2%} R={r['recall']:.2%} F1={r['f1_score']:.2%}")
        if 'simple_hit_rate' in r:
            line += f"  simple={r['simple_hit_rate']:.2%} complex={r['complex_hit_rate']:.2%}"
        print(line)

    return results


def cmd_run(args):
    """Generate and evaluate in one step."""
    cmd_generate(args)
    cmd_evaluate(args)


def main():
    parser = argparse.ArgumentParser(
        prog='llm4ke',
        description='LLM-based Competency Question Generation & Evaluation for Ontologies',
    )
    parser.add_argument('--config', default='config.yml', help='LLM config file')
    parser.add_argument('--data-dir', default='data', help='Root data directory')
    parser.add_argument('--output-dir', default='data_out', help='Root output directory')
    parser.add_argument('--log', type=int, default=20, choices=[10, 20, 30, 40, 50],
                        help='Log level: DEBUG=10 INFO=20 WARNING=30 ERROR=40 CRITICAL=50')

    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- generate ---
    gen = subparsers.add_parser('generate', help='Generate CQs from an ontology')
    gen.add_argument('ontology', help='Ontology name (folder name under data/)')
    gen.add_argument('--task', required=True,
                     choices=['all_classes', 'all_classes+properties', 'logic'])
    gen.add_argument('--llm', required=True, help='LLM backend name (defined in config.yml)')
    gen.add_argument('--n-cqs', type=int, default=10, help='Number of CQs to request per batch')
    gen.add_argument('--n-examples', type=int, default=0, help='Number of example CQs in prompt')
    gen.add_argument('--include-description', action='store_true')
    gen.set_defaults(func=cmd_generate)

    # --- evaluate ---
    ev = subparsers.add_parser('evaluate', help='Evaluate generated CQs')
    ev.add_argument('ontology', help='Ontology name or "all"')
    ev.add_argument('--threshold', type=float, default=0.6, help='Similarity threshold')
    ev.add_argument('--labels', default=None, help='Path to CQs_labeled.xlsx for classification stats')
    ev.set_defaults(func=cmd_evaluate)

    # --- run (generate + evaluate) ---
    run = subparsers.add_parser('run', help='Generate + evaluate in one step')
    run.add_argument('ontology', help='Ontology name')
    run.add_argument('--task', required=True,
                     choices=['all_classes', 'all_classes+properties', 'logic'])
    run.add_argument('--llm', required=True, help='LLM backend name')
    run.add_argument('--n-cqs', type=int, default=10)
    run.add_argument('--n-examples', type=int, default=0)
    run.add_argument('--include-description', action='store_true')
    run.add_argument('--threshold', type=float, default=0.6)
    run.add_argument('--labels', default=None, help='Path to CQs_labeled.xlsx')
    run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s:%(name)s:%(funcName)s:%(message)s',
        level=args.log,
    )
    args.func(args)


if __name__ == '__main__':
    main()
