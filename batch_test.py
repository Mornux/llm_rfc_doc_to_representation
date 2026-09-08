from extraction_runner import METHODS, run_extraction
from ollama_client import ensure_ollama_server, load_models
from result_exporter import export_extraction_results
from rfc_loader import discover_rfc_numbers


def run_batch_test():
    ensure_ollama_server()

    models = load_models()
    rfc_numbers = discover_rfc_numbers(5)

    for rfc_number in rfc_numbers:
        for method in METHODS:
            print("RFC", rfc_number, "method", method)

            results = run_extraction(rfc_number, method, models)
            exported_results = export_extraction_results(results, rfc_number, method)

            for model_name, answer, result_path, error in exported_results:
                if error or result_path is None:
                    print(model_name, "failed", error)
                else:
                    print(model_name, "success")


if __name__ == "__main__":
    run_batch_test()
