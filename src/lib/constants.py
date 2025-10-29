from itertools import chain
from os import getenv


BEDROCK_REGION = getenv('BEDROCK_REGION', 'us-east-1')
OPENSEARCH_URL = getenv('OPENSEARCH_URL')
KB_ID = getenv('KNOWLEDGEBASE_ID', 'BUQVYSGTXL')

LLM_MODELS = [
    'anthropic.claude-3-sonnet-20240229-v1:0',
    'anthropic.claude-3-haiku-20240307-v1:0',
    'anthropic.claude-v2:1'
]
EMBEDDING_MODEL = getenv('EMBEDDING_MODEL', 'amazon.titan-embed-text-v2:0')
EMBEDDING_MODELS_DIMENSIONS = {
    'amazon.titan-embed-text-v1': 1536,
    'amazon.titan-embed-text-v2:0': 1024
}

RULES_INDEX_NAME = getenv('RULES_INDEX_NAME', 'rules')
SPEC_SHEETS_INDEX = getenv('SPEC_SHEETS_INDEX', 'spec_sheets')
BID_REVIEW_INDEX = getenv('BID_REVIEW_INDEX', 'bid_reviews')
VECTOR_FIELD_NAME = getenv('VECTOR_FIELD_NAME', 'vector_field')
BID_CONTEXT_INDEX = getenv('BID_CONTEXT_INDEX', 'bid_context')

TEMP_SAVE_DIR = getenv('TEMP_SAVE_DIR', './tmp')
UPLOAD_DIR = getenv('UPLOAD_DIR', 'UPLOADED_FILES/')

BUCKET = getenv('BUCKET', 'company-s3-app-bidreviewdocument-input')
SPEC_SHEET_PREFIX = getenv('SPEC_SHEET_PREFIX', 'CUSTOMER SPECIFICATIONS/')
BID_REVIEW_PREFIX = getenv('BID_REVIEW_PREFIX', 'company EXCEPTIONS/')
BID_REVIEWS: list[dict[str, str | list]] = [
    {
        'BID_REVIEW': 'ATC/24-25636 -ATC -SR 09-11-23.xlsx',
        'SPEC_SHEETS': [
            'ATC/SNM-2100.pdf',
            'ATC/SNM-2150.pdf',
        ]
    },
    {
        'BID_REVIEW': 'NATIONAL GRID/24-26502 -NATIONAL GRID - SR 02-15-24.xlsx',
        'SPEC_SHEETS': [
            'NATIONAL GRID/T11_StructureInformation.pdf',
            'NATIONAL GRID/Spec SP.06.01.407 5.0 dated 8.22.23.pdf',
            'NATIONAL GRID/100037-G-2B-M-01.pdf',
        ]
    },
    {
        'BID_REVIEW': 'XCEL/00-00000 -XCEL - SR 08-31-23.xlsx',
        'SPEC_SHEETS': [
            'XCEL/XEL-STD-Specification for Procurement of Tubular Steel Poles version 3.7.docx'
        ]
    },
]
SPEC_SHEETS = list(
    chain.from_iterable(bid_review['SPEC_SHEETS'] for bid_review in BID_REVIEWS)
)
WORKBOOK = getenv('WORKBOOK', 'company EXCEPTIONS/Tech Bible for Spec Review R-7.xlsx')
BATCH_SIZE = getenv('BATCH_SIZE', 5)
