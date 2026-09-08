import json
import os
import difflib


def attribute_accuracy(reference, generated, attributes=None):
    if attributes is None:
        attributes = (
            "display name",
            "offset bits",
            "size bits",
            "type",
            "optional",
            "presence condition",
            "constraints",
        )

    documents = []
    for document in (reference, generated):
        if isinstance(document, (str, os.PathLike)):
            with open(document, "r", encoding="utf-8") as file:
                document = json.load(file)
        documents.append(document)

    all_fields = []
    for document in documents:
        fields = []
        stack = [document]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                for key, child in value.items():
                    if key.replace("_", " ").lower() == "fields" and isinstance(child, list):
                        fields.extend(field for field in child if isinstance(field, dict))
                    else:
                        stack.append(child)
            elif isinstance(value, list):
                stack.extend(reversed(value))
        all_fields.append(fields)

    reference_fields, generated_fields = all_fields

    generated_by_name = {}
    for field in generated_fields:
        name = field.get("name") or field.get("display name")
        if isinstance(name, str):
            name = " ".join(name.lower().replace("_", " ").split())
            generated_by_name.setdefault(name, []).append(field)

    correct = 0
    total = 0
    for reference_field in reference_fields:
        name = reference_field.get("name") or reference_field.get("display name")
        if not isinstance(name, str):
            continue
        name = " ".join(name.lower().replace("_", " ").split())

        candidates = generated_by_name.get(name, [])
        if not candidates:
            continue
        generated_field = candidates.pop(0)

        for attribute in attributes:
            total += 1
            if attribute in generated_field and generated_field[attribute] == reference_field.get(attribute):
                correct += 1

    return correct / total if total else 0


def NLS(reference, standardized_test_file):
    import Levenshtein

    with open(reference, "r", encoding="utf-8") as file:
        rf = json.dumps(json.load(file), sort_keys=True) #reference file

    with open(standardized_test_file, "r", encoding="utf-8") as file:
        stf = json.dumps(json.load(file), sort_keys=True) #standardized test file

    #nomalized_levenshtein_similarity = 1 - Levenshtein_Distance(doc_test, doc_real) / max(len(doc_test), len(doc_real))
    nls = 1 - (Levenshtein.distance(rf, stf) / max(len(rf), len(stf)))
    print(standardized_test_file, nls)

    return nls


def extract_path_value_pairs(document_path):
    with open(document_path, "r", encoding="utf-8") as file:
        document = json.load(file)

    pv_pairs = [] #list of path-value pairs

    def recursive_extract(path, value):
        if type(value) == dict:
            for key in value:
                child = value[key]
                new_path = path + "." + key
                recursive_extract(new_path, child)
        elif type(value) == list:
            index = 0
            while index < len(value):
                child = value[index]
                new_path = path + "[" + str(index) + "]"
                recursive_extract(new_path, child)
                index = index + 1
        else:
            pv_pairs.append((path, value))

    recursive_extract("root", document)
    return pv_pairs


def calculate_tp(reference_pairs, generated_pairs): #get value of true positive
    tp = 0

    for path, reference_value in reference_pairs:
        if path not in dict(generated_pairs):
            continue

        generated_value = dict(generated_pairs)[path]

        if type(reference_value) == str and type(generated_value) == str:
            reference_value = reference_value.strip() #eliminate possible interference caused by spaces
            generated_value = generated_value.strip()
            similarity = difflib.SequenceMatcher(None, reference_value, generated_value).ratio()#
            if similarity > 0.5:
                tp += 1
            else:
                tp = tp
            #tp = tp + similarity
        elif type(reference_value) == type(generated_value):
            if reference_value == generated_value:
                tp += 1

    return tp


def calculate_fp(reference_pairs, generated_pairs):#get value of false positive
    tp = calculate_tp(reference_pairs, generated_pairs)
    fp = len(generated_pairs) - tp
    return fp


def calculate_fn(reference_pairs, generated_pairs):#get value of false negative
    tp = calculate_tp(reference_pairs, generated_pairs)
    fn = len(reference_pairs) - tp
    return fn


def soft_precision(tp, fp):
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)


def soft_recall(tp, fn):
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)


def soft_f1(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def score_merge_output(reference, standardized_test_file):
    reference_pairs = extract_path_value_pairs(reference)
    generated_pairs = extract_path_value_pairs(standardized_test_file)

    tp = calculate_tp(reference_pairs, generated_pairs)
    fp = calculate_fp(reference_pairs, generated_pairs)
    fn = calculate_fn(reference_pairs, generated_pairs)

    precision = soft_precision(tp, fp)
    recall = soft_recall(tp, fn)
    f1 = soft_f1(precision, recall)

    print(standardized_test_file, precision, recall, f1)

    return precision, recall, f1


def format_standardized():
    input_folder = "savefile"
    output_folder = "savefile/standardized_file"
    os.makedirs(output_folder, exist_ok=True)

    for name in os.listdir(input_folder):
        if name.endswith(".json"):
            with open(input_folder + "/" + name, "r", encoding="utf-8") as file:
                data = json.load(file)

            data = json.dumps(data)
            data = data.lower().replace("_", " ")

            with open(output_folder + "/" + name, "w", encoding="utf-8") as file:
                file.write(data)


def save_evaluation_results():
    project_folder = os.path.dirname(os.path.abspath(__file__))
    reference_folder = project_folder + "/reference/standardized_file"
    generated_folder = project_folder + "/savefile/standardized_file"
    result_folder = project_folder + "/evaluation_results"
    result_path = result_folder + "/result.txt"
    model_scores = {}

    format_standardized()

    for name in os.listdir(generated_folder):
        if not name.endswith(".json"):
            continue

        file_name = name[:-5]
        document_name = file_name.split("_")[0]

        if file_name.endswith("_llm_ascii"):
            method = "llm_ascii"
        elif file_name.endswith("_ascii"):
            method = "ascii"
        elif file_name.endswith("_full"):
            method = "full"
        else:
            continue

        model_name = file_name[len(document_name) + 1 : -(len(method) + 1)]
        model_method = (model_name, method)

        reference_path = reference_folder + "/" + document_name + "_reference.json"
        generated_path = generated_folder + "/" + name

        if not os.path.exists(reference_path):
            continue

        nls = NLS(reference_path, generated_path)
        precision, recall, f1 = score_merge_output(reference_path, generated_path)
        accuracy = attribute_accuracy(reference_path, generated_path)

        if model_method not in model_scores:
            model_scores[model_method] = []
        model_scores[model_method].append((nls, precision, recall, f1, accuracy))

    with open(result_path, "w", encoding="utf-8") as file:
        file.write(f"{'model_name':<30}{'method':<20}{'anls':<20}{'attribute_accuracy':<20}{'average_precision':<20}{'average_recall':<20}{'average_f1':<20}\n")

        for model_name, method in sorted(model_scores):
            all_scores = model_scores[(model_name, method)]
            count = len(all_scores)

            average_nls = sum(score[0] for score in all_scores) / count
            average_precision = sum(score[1] for score in all_scores) / count
            average_recall = sum(score[2] for score in all_scores) / count
            average_f1 = sum(score[3] for score in all_scores) / count
            average_accuracy = sum(score[4] for score in all_scores) / count

            file.write(f"{model_name:<30}{method:<20}{average_nls:<20.2f}{average_accuracy:<20.2f}{average_precision:<20.2f}{average_recall:<20.2f}{average_f1:<20.2f}\n")

    return 0

if __name__ == "__main__":
    save_evaluation_results()

