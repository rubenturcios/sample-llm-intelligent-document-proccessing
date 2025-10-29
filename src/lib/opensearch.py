import json
import logging
import os
from pprint import pformat
from typing import Any

import boto3
from opensearchpy import (
    OpenSearch,
    AWSV4SignerAuth,
    RequestsHttpConnection
)
from opensearchpy.helpers import bulk

from .constants import (
    BEDROCK_REGION,
    OPENSEARCH_URL,
)


format = '%(levelname)s:%(filename)s:%(lineno)d:%(asctime)s:%(message)s'
logging.basicConfig(format=format)
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv('LOG_LEVEL', logging.INFO))

session = boto3.Session()
credentials = session.get_credentials()
os_auth = AWSV4SignerAuth(
    credentials=credentials,
    region=BEDROCK_REGION,
    service='aoss'
)
os_client = OpenSearch(
    hosts=[{'host': OPENSEARCH_URL.split("/")[-1], 'port': 443}],
    http_auth=os_auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=300
)
s3_client = session.client('s3')


def default_text_mappings(
    *,
    metadata_field_name: str = 'metadata',
    text_field_name: str = 'text',
    vector_field_name: str = 'vector_field'
) -> dict[str, Any]:
    return {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param": {
                    "ef_search": 512
                },
                "number_of_shards": 2,
                "number_of_replicas": 0
            }
        },
        "mappings": {
            "properties": {
                "id": {
                    "type": "text"
                },
                "x-amz-bedrock-kb-source-uri": {
                    "type": "text"
                },
                metadata_field_name: {
                    "type": "text",
                    "index": False
                },
                text_field_name: {
                    "type": "text"
                },
                vector_field_name: {
                    "type": "knn_vector",
                    "dimension": 1536,
                    "method": {
                        "engine": "faiss",
                        "space_type": "l2",
                        "name": "hnsw",
                    }
                }
            }
        }
    }


def check_index_exist(index_name: str) -> bool:
    return os_client.indices.exists(index_name)


def create_index(index_name: str, *, default_mappings: dict = None) -> None:
    if default_mappings:
        response = os_client.indices.create(index_name, body=default_mappings)
    else:
        response = os_client.indices.create(index_name)

    return logger.info('Create Index Response: %s', response)


def delete_index(index_name: str) -> None:
    logger.warning('Deleting: %s', index_name)
    logger.info(os_client.indices.delete(index_name))


def get_all_os_entries(
    index_name: str,
) -> list[dict]:
    query = {'query': {'match_all': {}}}
    return get_os_entries(index_name, query)


def get_os_entries(
    index_name: str,
    query_args: dict = None
) -> list[dict]:
    response = os_client.search(body=query_args, index=index_name)
    logger.info('Number of hits: %s', response['hits']['total']['value'])

    entries = [hit for hit in response['hits']['hits']]

    for entry in entries:
        logger.debug(pformat(entry))

    return entries


def put_os_entries(index_name: str, entries: list[dict], *, type: str = 'index') -> None:
    for entry in entries:
        entry.update({'_op_type': type, '_index': index_name})

    logger.info(bulk(os_client, entries))


def index_requirements(index_name: str, body: dict) -> None:
    response = os_client.index(
        index=index_name,
        body=body
    )
    return logger.info('Response: %s', response)


def get_sheet_categories(index_name: str) -> list[dict]:
    query = {
        'size': 0,
        'aggs': {
            'sheets': {
                'terms': {
                    'field': 'sheet.keyword',
                    'size': 100
                }
            }
        }
    }
    result = os_client.search(query, index_name)
    buckets = result['aggregations']['sheets']['buckets']
    logger.debug(buckets)
    return buckets


def get_hits_by_msearch(
    index_name: str, queries: list[dict]
) -> list[list[dict[str, str | dict]]]:
    query_string = '\n'.join("".join(json.dumps(query).split()) for query in queries )
    response = os_client.msearch(body=query_string, index=index_name)
    logger.info('Len Responses: %s', len(response['responses']))
    hits = [entry['hits']['hits'] for entry in response['responses']]
    logger.info('Len hits list: %s', len(hits))
    for hit in hits: logger.debug(pformat(hit))
    return hits
