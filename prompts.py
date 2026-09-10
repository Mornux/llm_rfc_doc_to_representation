#store and build prompts used by the extraction pipelines


RFC_EXTRACTION_INSTRUCTION = """
Your entire response MUST be exactly one JSON object conforming to the schema below. The first non-whitespace character MUST be { and the last MUST be }.
NEVER return Markdown, headings, prose, summaries, tables, code fences, XML tags, Python assignments, or analysis outside that JSON object. Do not replace the required representation with a human-readable RFC summary. Every required top-level key MUST be present even when its value is an empty array.
You are a network protocol specification analysis system.
Your task is to analyse an RFC document and convert the packet format described in the RFC into a structured Network Packet Representation.
The representation must describe protocol data units using typed packet structures, field layouts, constraints, optional fields, variable-length data, and parsing dependencies when such information is available.

Core Representation Concepts:
Use the following concepts when constructing the representation:
Protocol: the overall network protocol.
PDU: a protocol data unit or packet format defined by the protocol.
Bit String Type: raw protocol data with a specific bit width and semantic meaning.
Enumerated Type: a value or packet type that can have multiple variants.
Structure Type: an ordered collection of heterogeneous packet fields.
Array Type: a sequence of elements whose length may be fixed or depend on another field.
Presence Condition: an expression determining whether a field is present.
Constraint: a condition that must hold for a packet or field to be valid.
Parsing Context: persistent or external information required to parse packet data.
Helper Function: a function used to calculate values or evaluate constraints.
Transform Function: a function describing how one representation is parsed from or serialized to another representation.

Extraction Rules:
1. Use only information contained in the supplied RFC input.
2. Do not use prior knowledge about the protocol.
3. Do not invent:
packet fields
field sizes
field offsets
constraints
optionality
packet types
parsing dependencies
functions
4. Packet diagrams and field descriptions are the primary evidence for packet structure.
5. Field order must match the order in which fields occur on the wire.
6. offset_bits is measured from the beginning of the current PDU.
7. size_bits may be obtained from:
an explicit field size in the RFC;
packet diagram boundaries;
a deterministic calculation based on preceding fixed-width fields.
8. If a size or offset cannot be determined reliably, use `null`.
9. Use lowercase snake_case for machine-readable field names.
Example:
Source Port
becomes:
"name": "source_port",
"display_name": "Source Port"
10. A field may only have:
"optional": true
when the RFC explicitly defines it as optional or defines a condition under which it is present.
11. If the presence of a field depends on another field, store that dependency in:
"presence_condition"
Example:
"presence_condition": "x == 1"
12. Only generate constraints that are explicitly stated in the RFC or can be deterministically expressed from an RFC requirement.
Examples:
"length >= 8"
"version == 4"
"payload_length == length - 8"
13. If no constraint is available, use:
"constraints": []
14. Variable-length data must not be converted into an arbitrary fixed size.
Represent its size using an expression when possible.
Example:
"size": "length - 8"
15. Each distinct packet-format ASCII diagram must be represented as a separate entry in `"packet"`. All extracted information associated with that diagram, including PDUs, bit string types, enumerated types, structure types, array types, parsing context, helper functions, and transform functions, must be stored only within that diagram's entry. Information from different diagrams must not be merged.
16. If multiple packet formats are alternatives of a common packet type, they may be represented as variants of an enumerated type.
17. Different fields with the same bit width may still represent different semantic types.
18. Create parsing context information only when parsing depends on:
previous packets;
external state;
out-of-band information.
19. Create helper functions or transform functions only when the RFC explicitly describes such processing.
20. Never guess missing information.
When evidence is insufficient, use:
null
an empty array
an empty object where appropriate
Accuracy is more important than completeness.
Required Output Format
Return ONLY one valid JSON object.
Do NOT output:
Markdown
code fences
comments
explanations
Python assignments
text before the JSON
text after the JSON
The JSON must follow this structure:
{
  "protocol": "string",
  "rfc": "string",
  "packet": [
    {
      "name": "string",
      "pdus": [
        "string"
      ],
      "bit_string_types": [
        {
          "name": "string",
          "size_bits": "integer or null"
        }
      ],
      "enumerated_types": [
        {
          "name": "string",
          "variants": [
            {
              "name": "string",
              "type": "string",
              "condition": "string or null"
            }
          ]
        }
      ],
      "structure_types": [
        {
          "name": "string",
          "header_size_bytes": "integer or null",
          "byte_order": "string or null",
          "fields": [
            {
              "name": "string",
              "display_name": "string",
              "offset_bits": "integer or null",
              "size_bits": "integer or null",
              "type": "string or null",
              "optional": "boolean or null",
              "presence_condition": "string or null",
              "constraints": [
                "string"
              ]
            }
          ],
          "constraints": [
            "string"
          ],
          "parse_from": "string or null",
          "serialize_to": "string or null"
        }
      ],
      "array_types": [
        {
          "name": "string",
          "element_type": "string",
          "length": "integer, expression, or null"
        }
      ],
      "parsing_context": [
        {
          "name": "string",
          "type": "string"
        }
      ],
      "helper_functions": [
        {
          "name": "string"
        }
      ],
      "transform_functions": [
        {
          "name": "string",
          "from": "string",
          "to": "string"
        }
      ]
    }
  ]
}

RFC Input:
<RFC_DOCUMENT>
{RFC_TEXT}
</RFC_DOCUMENT>

"""


def build_representation_prompt(evidence):
    #insert one evidence source into the shared representation instruction
    return RFC_EXTRACTION_INSTRUCTION.replace("{RFC_TEXT}", evidence)


REPRESENTATION_REPAIR_INSTRUCTION = """Your previous response violated the mandatory output contract:

{error}

Return the corrected Network Packet Representation now.

Output exactly one JSON object and nothing else. It must contain every required
top-level key shown in the original instruction. Do not summarize the RFC and
do not use Markdown.

<INVALID_RESPONSE>

{invalid_answer}

</INVALID_RESPONSE>

<ORIGINAL_INSTRUCTION>

{original_prompt}

</ORIGINAL_INSTRUCTION>"""


def build_representation_repair_prompt(original_prompt, invalid_answer, error):
    #ask the model to correct one invalid representation response
    return REPRESENTATION_REPAIR_INSTRUCTION.format(
        error=error,
        invalid_answer=invalid_answer,
        original_prompt=original_prompt,
    )


DIAGRAM_LOCATION_INSTRUCTION = """
Read the complete numbered RFC document below.

Find every ASCII diagram that defines the layout of a network packet, message, header, frame, or protocol data unit.

Reject ordinary tables, state machines, timelines, examples, and message flow diagrams.

Return exactly one JSON object in this form:

{{
  "diagram_ranges": [
    {{
      "start_line": 10,
      "end_line": 20
    }}
  ]
}}

Use an empty diagram_ranges array when no packet layout exists. 
Include only line ranges that belong to the ASCII diagram itself.

<RFC number="{rfc_number}">

{numbered_document}

</RFC>
"""


def build_diagram_location_prompt(rfc_number, numbered_document):
    #ask a model to return packet diagram line ranges from a complete rfc
    return DIAGRAM_LOCATION_INSTRUCTION.format(
        rfc_number=rfc_number,
        numbered_document=numbered_document,
    )


DIAGRAM_REPAIR_INSTRUCTION = """
Correct the diagram location response below.

Return exactly one JSON object with a diagram_ranges array.
Every array item must contain integer start_line and end_line values between 1 and {line_count}.

The previous response failed validation because:

{error}

<INVALID_RESPONSE>

{invalid_answer}

</INVALID_RESPONSE>
"""


def build_diagram_repair_prompt(line_count, invalid_answer, error):
    #ask the model to correct an invalid diagram range response
    return DIAGRAM_REPAIR_INSTRUCTION.format(
        line_count=line_count,
        error=error,
        invalid_answer=invalid_answer,
    )
