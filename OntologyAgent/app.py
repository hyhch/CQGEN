import json
import asyncio
import os
import threading
import queue
import gradio as gr

from config.Config import (
    DATA_DIR_PATH, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_API_TYPE,
    SEGMENTATION_METHOD, RANDOM_SEED, MAX_NUM_SUB_TRIPLES,
)

# --- Discover available datasets ---
def get_available_datasets():
    datasets = []
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    if not os.path.isdir(dataset_dir):
        return datasets
    for name in sorted(os.listdir(dataset_dir)):
        json_path = os.path.join(dataset_dir, name, f"{name}.json")
        if os.path.isfile(json_path):
            datasets.append(name)
    return datasets

AVAILABLE_DATASETS = get_available_datasets()

# --- Provider presets ---
PROVIDER_PRESETS = {
    "DashScope (qwen-max)": {
        "api_type": "openai",
        "model": "qwen-max",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": LLM_API_KEY,
    },
    "Azure (gpt-4o)": {
        "api_type": "azure",
        "model": "gpt-4o",
        "base_url": "https://your-azure-endpoint.openai.azure.com/",
        "api_key": "",
    },
    "Ollama (local)": {
        "api_type": "ollama",
        "model": "llama3.3:latest",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    },
}

# --- Pipeline runner ---
def run_pipeline(provider, model_name, api_key, base_url, dataset_name,
                 cq_examples_num, max_iterations, segmentation_method):
    """Generator that yields (progress_log, cqs_text, coverage_df, details_json) tuples."""
    log_queue = queue.Queue()
    result_holder = {"result": None, "error": None}

    def progress_callback(msg: str):
        log_queue.put(msg)

    def _run():
        try:
            from src.CQRetrofit import main as cq_main

            # Build llm_config from the selected provider preset
            preset = PROVIDER_PRESETS.get(provider, {})
            llm_config = {
                "api_type": preset.get("api_type", LLM_API_TYPE),
                "model": model_name,
                "api_key": api_key,
                "base_url": base_url,
            }

            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                cq_main(
                    data_dir=data_dir,
                    ontology_name=dataset_name,
                    cq_examples_num=int(cq_examples_num),
                    max_loop_count=int(max_iterations),
                    max_sub_triples=MAX_NUM_SUB_TRIPLES,
                    segmentation_method=segmentation_method,
                    random_seed=RANDOM_SEED,
                    progress_callback=progress_callback,
                    llm_config=llm_config,
                )
            )
            loop.close()
            result_holder["result"] = result
        except Exception as e:
            import traceback
            result_holder["error"] = traceback.format_exc()
            log_queue.put(f"\n[ERROR] {e}")
        finally:
            log_queue.put(None)  # sentinel

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    progress_log = ""
    while True:
        try:
            msg = log_queue.get(timeout=0.5)
        except queue.Empty:
            # Yield current state while waiting
            yield progress_log, "", [], "{}"
            continue

        if msg is None:
            break
        progress_log += msg + "\n"
        yield progress_log, "", [], "{}"

    # Build final outputs
    result = result_holder["result"]
    error = result_holder["error"]

    if error:
        progress_log += f"\n\nPipeline failed.\n{error}"
        yield progress_log, "", [], "{}"
        return

    if result is None:
        progress_log += "\n\nNo results returned."
        yield progress_log, "", [], "{}"
        return

    # 1) CQs text
    cqs = result.get("retrofitted_competency_questions", [])
    cqs_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(cqs))

    # 2) Coverage table
    stats = result.get("overall_coverage_stats", {})
    coverage_rows = []
    for detail in stats.get("subgraph_coverage_details", []):
        coverage_rows.append([
            f"Subgraph {detail['subgraph_id'] + 1}",
            detail["total_entities"],
            detail["covered_entities"],
            f"{detail['coverage_rate']:.1f}%",
        ])
    coverage_rows.append([
        "Overall",
        stats.get("total_entities", 0),
        stats.get("total_covered_entities", 0),
        f"{stats.get('overall_coverage_rate', 0):.1f}%",
    ])

    # 3) JSON details (strip heavy fields for display)
    details = {
        "overall_coverage_stats": stats,
        "total_generated_cqs": len(cqs),
        "retrofitted_competency_questions": cqs,
    }
    details_json = json.dumps(details, ensure_ascii=False, indent=2)

    progress_log += "\n\nDone!"
    yield progress_log, cqs_text, coverage_rows, details_json


# --- UI ---
def on_provider_change(provider):
    """Auto-fill model/key/url when provider dropdown changes."""
    preset = PROVIDER_PRESETS.get(provider, {})
    return (
        preset.get("model", ""),
        preset.get("api_key", ""),
        preset.get("base_url", ""),
    )


with gr.Blocks(title="OntologyAgent - CQ Retrofitting") as demo:
    gr.Markdown("# OntologyAgent - Competency Question Retrofitting")
    gr.Markdown("Generate and validate competency questions for OWL/RDF ontologies using multi-agent LLM pipeline.")

    with gr.Row():
        # ---- Left panel: controls ----
        with gr.Column(scale=1):
            gr.Markdown("### Configuration")
            provider_dropdown = gr.Dropdown(
                choices=list(PROVIDER_PRESETS.keys()),
                value="DashScope (qwen-max)",
                label="Model Provider",
            )
            model_textbox = gr.Textbox(
                value=LLM_MODEL,
                label="Model Name",
            )
            api_key_textbox = gr.Textbox(
                value=LLM_API_KEY,
                label="API Key",
                type="password",
            )
            base_url_textbox = gr.Textbox(
                value=LLM_BASE_URL,
                label="Base URL",
            )

            gr.Markdown("### Dataset")
            dataset_dropdown = gr.Dropdown(
                choices=AVAILABLE_DATASETS,
                value=AVAILABLE_DATASETS[0] if AVAILABLE_DATASETS else None,
                label="Ontology Dataset",
            )

            gr.Markdown("### Parameters")
            cq_examples_slider = gr.Slider(
                minimum=1, maximum=30, value=5, step=1,
                label="CQ Examples Count",
            )
            max_iter_slider = gr.Slider(
                minimum=1, maximum=10, value=3, step=1,
                label="Max Iterations",
            )
            segmentation_dropdown = gr.Dropdown(
                choices=["auto", "metis", "louvain", "leiden", "spectral", "random"],
                value=SEGMENTATION_METHOD,
                label="Segmentation Method",
            )

            run_btn = gr.Button("Run Pipeline", variant="primary", size="lg")

        # ---- Right panel: output ----
        with gr.Column(scale=2):
            progress_log = gr.Textbox(
                label="Progress Log",
                lines=20,
                max_lines=40,
                interactive=False,
            )
            with gr.Tabs():
                with gr.Tab("Generated CQs"):
                    cqs_output = gr.Textbox(
                        label="Competency Questions",
                        lines=15,
                        max_lines=30,
                        interactive=False,
                    )
                with gr.Tab("Coverage Stats"):
                    coverage_table = gr.Dataframe(
                        headers=["Subgraph", "Total Entities", "Covered Entities", "Coverage Rate"],
                        label="Coverage Statistics",
                        interactive=False,
                    )
                with gr.Tab("Details (JSON)"):
                    details_json = gr.Textbox(
                        label="Full Results JSON",
                        lines=20,
                        max_lines=40,
                        interactive=False,
                    )

    # --- Events ---
    provider_dropdown.change(
        fn=on_provider_change,
        inputs=[provider_dropdown],
        outputs=[model_textbox, api_key_textbox, base_url_textbox],
    )

    run_btn.click(
        fn=run_pipeline,
        inputs=[
            provider_dropdown, model_textbox, api_key_textbox, base_url_textbox,
            dataset_dropdown, cq_examples_slider, max_iter_slider,
            segmentation_dropdown,
        ],
        outputs=[progress_log, cqs_output, coverage_table, details_json],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
