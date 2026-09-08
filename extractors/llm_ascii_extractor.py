#use one model to locate packet ascii diagrams in a complete rfc

import json

from ollama_client import run_model
from prompts import build_diagram_location_prompt, build_diagram_repair_prompt


def add_line_numbers(document):
    #add stable source line numbers for the model response, I think this is a way to help LLM to locate
    numbered_lines = []
    for index, line in enumerate(document.splitlines(), start=1):
        numbered_lines.append("{}: {}".format(index, line))
    return "\n".join(numbered_lines)


def parse_diagram_ranges(answer, line_count):
    #parse and validate every returned diagram line range
    try:
        data = json.loads(answer)
    except json.JSONDecodeError as error:
        raise ValueError("diagram location response is not valid json") from error
    if not isinstance(data, dict):
        raise ValueError("diagram location response must be a json object")

    #Small local models sometimes return the right range in a slightly different
    #JSON shape.  Normalize the common variants before validating the contents.
    ranges = data.get("diagram_ranges")
    if ranges is None and "diagram_range" in data:
        ranges = data.get("diagram_range")
    if ranges is None and "ranges" in data:
        ranges = data.get("ranges")
    if ranges is None and "start_line" in data and "end_line" in data:
        ranges = data
    if isinstance(ranges, str):
        try:
            ranges = json.loads(ranges)
        except json.JSONDecodeError as error:
            raise ValueError("diagram_ranges string is not valid json") from error
    if ranges is None:
        ranges = []
    elif isinstance(ranges, dict):
        ranges = [ranges]
    elif not isinstance(ranges, list):
        raise ValueError("diagram_ranges must be an array or an object")

    valid_ranges = []
    for item in ranges:
        if not isinstance(item, dict):
            raise ValueError("every diagram range must be a json object")
        start_line = item.get("start_line")
        end_line = item.get("end_line")
        if not isinstance(start_line, int) or not isinstance(end_line, int):
            raise ValueError("diagram line numbers must be integers")
        if start_line < 1 or end_line < start_line or end_line > line_count:
            raise ValueError("diagram line range is outside the rfc document")
        valid_ranges.append((start_line, end_line))
    return valid_ranges


def locate_diagram_ranges(rfc_number, document, model_name):
    #send the complete numbered rfc to the diagram location model
    numbered_document = add_line_numbers(document)
    prompt = build_diagram_location_prompt(rfc_number, numbered_document)
    answer = run_model(model_name, prompt, json_output=True)
    line_count = len(document.splitlines())
    try:
        return parse_diagram_ranges(answer, line_count)
    except ValueError as first_error:
        #retry once with an explicit range format correction
        repair_prompt = build_diagram_repair_prompt(
            line_count, answer, first_error
        )
        repaired = run_model(model_name, repair_prompt, json_output=True)
        return parse_diagram_ranges(repaired, line_count)


def extract_diagram_evidence(document, ranges, context_lines=12):
    #copy selected ranges with nearby field descriptions
    lines = document.splitlines()
    sections = []
    for index, line_range in enumerate(ranges, start=1):
        start_line = line_range[0]
        end_line = line_range[1]
        context_start = max(0, start_line - 1 - context_lines)
        context_end = min(len(lines), end_line + context_lines)
        context = "\n".join(lines[context_start:context_end])
        section = "<DIAGRAM number=\"{}\" lines=\"{}-{}\">\n{}\n</DIAGRAM>"
        sections.append(section.format(index, start_line, end_line, context))
    return "\n\n".join(sections)


def extract_llm_ascii_evidence(rfc_number, document, model_name=None):
    #use the current representation model to locate its own evidence
    if not model_name:
        raise ValueError("A model name is required for llm_ascii extraction.")
    ranges = locate_diagram_ranges(rfc_number, document, model_name)
    if not ranges:
        return ""
    return extract_diagram_evidence(document, ranges)
