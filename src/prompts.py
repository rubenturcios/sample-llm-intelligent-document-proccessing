ATC_PROMPT = """You are a Manufacturing Contract reviewer. Given a Manufacturing CONTRACT provided by a customer, \
return a list of formatted requirements from the CONTRACT. Make sure to look at all sections and subsections. \
Make sure to pull the requirements from the CONTRACT ONLY.

Any requirement in a numbered or bulleted list form should be extracted out into its own requirement. \
If a requirement text has a quotation symbol ("), replace it with a single quote (').

Wrap the list elements in <requirement></requirement> XML tags.
The requirements must follow the following JSON format:

{{
    "PAGE NUMBER": Page number of the requirement.
    "SECTION / SUBSECTION": The Section and Subsection of the requirement.
    "EXCEPTIONS / SPECIAL REQUIREMENTS": The requirement itself. (Replace any double quotes with single quotes.)
}}

Use these examples as reference, but do not include them in your final answer:

{examples}

Here is the contract:

<CONTRACT>
{context}
</CONTRACT>

Result:
"""

CLASSIFIER_PROMPT = """You are a manufacturing contract reviewer. \
Classify the following customer REQUIREMENT as either ACCEPT or EXCEPTION using the CONTEXT provided. \
You also need to provide notes to the customer and internal notes using the CONTEXT provided. \
If you do not have enough information from the EXAMPLES and CONTEXT to confidently classify the REQUIREMENT, \
default to EXCEPTION.

The requirement must follow the following JSON format:

{{
    "RECOMMENDATION": The value the requirement is classified as.
    "NOTES TO CUSTOMER": Notes to the customer.
    "INTERNAL NOTES": Notes to internal deparments.
    "DEPT. RESPONSIBLE": Department responsible for handling the requirement.
}}

Use the following examples to help you in your classification: \

{examples}

Here is some context:

<CONTEXT>
{context}
</CONTEXT>

Here is the requirement:

<REQUIREMENT>
{requirements}
</REQUIREMENT>

RESULT:
"""

DEFAULT_SPEC_SHEET_PROMPT = ATC_PROMPT
EXAMPLE_FORMAT = '<EXAMPLE>\n{}\n</EXAMPLE>'