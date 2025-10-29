import ast
import asyncio
from datetime import datetime
import json
from pprint import pformat
import re
import logging
import os
from typing import Optional

import boto3
from botocore.config import Config
from langchain_core.documents import Document
from langchain_community.vectorstores.opensearch_vector_search \
    import OpenSearchVectorSearch
from langchain_community.retrievers.bedrock import AmazonKnowledgeBasesRetriever
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from opensearchpy import AWSV4SignerAuth, RequestsHttpConnection

from .utils import arun_batch, batched, save_output, summarize
from .constants import (
    BEDROCK_REGION,
    OPENSEARCH_URL,
    TEMP_SAVE_DIR,
    KB_ID,
)


logger = logging.getLogger(__name__)
logger.setLevel(level=os.getenv('LOG_LEVEL', logging.INFO))

config = Config(read_timeout=500)
session = boto3.Session()
credentials = session.get_credentials()
bedrock_agent_client = session.client("bedrock-agent-runtime", region_name=BEDROCK_REGION, config=config)
bedrock_client = session.client("bedrock-runtime", region_name=BEDROCK_REGION, config=config)
os_auth = AWSV4SignerAuth(
    credentials=credentials,
    region=BEDROCK_REGION,
    service='aoss'
)


def get_aws_bedrock_retriever(
    knowledgebase_id: str, *, top_rec: int = 4
) -> AmazonKnowledgeBasesRetriever:
    return AmazonKnowledgeBasesRetriever(
        client=bedrock_agent_client,
        knowledge_base_id=knowledgebase_id,
        retrieval_config={
            "vectorSearchConfiguration": {
                "numberOfResults": top_rec
            }
        },
    )


def invoke_model(model_id: str, prompt: str, **model_kwargs) -> str:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        **model_kwargs
    }
    api_response = bedrock_client.invoke_model(
        body=json.dumps(body),
        modelId=model_id,
    )
    response = json.loads(api_response['body'].read())
    logger.debug(pformat(response))
    return response['content'][0]['text']


def get_page_string(doc: Document) -> str:
    #page_number = doc.metadata['page']
    page_content = doc.page_content
    return f'<Page>{page_content}</Page>'
    #return f"<Page {page_number}>\n{page_content}\n</Page {page_number}>"""


def embed_rule_requirements(text: str, model_id: str, **model_kwargs) -> list[float]:
    response = bedrock_client.invoke_model(
        body=json.dumps(
            {'inputText': text, **model_kwargs}
        ).encode('utf-8'),
        modelId=model_id,
    )
    deserialized = json.loads(response['body'].read())
    logger.debug('Embedding Response: %s', deserialized)
    return deserialized['embedding']


async def aembed_rule_requirements(text: str, model_id: str, **model_kwargs) -> list[float]:
    return await asyncio.get_running_loop().run_in_executor(
        None, embed_rule_requirements, text, model_id, **model_kwargs
    )


def get_opensearch_helper(
    model_id: str,
    index_name: str,
) -> OpenSearchVectorSearch:
    embedding_functions = BedrockEmbeddings(
        client=bedrock_client,
        model_id=model_id,
        normalize=True
    )
    return OpenSearchVectorSearch(
        connection_class=RequestsHttpConnection,
        embedding_function=embedding_functions,
        opensearch_url=OPENSEARCH_URL,
        index_name=index_name,
        verify_certs=True,
        http_auth=os_auth,
        engine='faiss',
        use_ssl=True,
        timeout=300
    )


def save_prompts_local(
    prompt: str,
    inputs: list[dict[str, str]]
) -> None:
    for index, input in enumerate(inputs):
        file = open(f'Prompt_{index}_{datetime.now()}.txt', 'w')
        file.write(prompt.format(**input))
        file.close()


def parse_llm_output_to_mapping(
    text: str,
    regex: str,
    *,
    default: Optional[dict] = None
) -> tuple[list[dict], list[str]]:
    matches: list[str] = re.findall(regex, text, flags=re.DOTALL)
    formatted_matches = []
    errors = []

    for match in matches:
        try:
            formatted_matches.append(json.loads(match))
            continue
        except Exception as e:
            pass
        try:
            formatted_matches.append(ast.literal_eval(match))
            continue
        except Exception as e:
            logger.exception(e)

        logger.error('ERROR MATCH: %s', match)
        errors.append(match)
        if default is not None: formatted_matches.append(default)

    return formatted_matches, errors


async def classify(
    model_id: str,
    prompt_template: str,
    examples: list[str],
    requirements: list[dict]
) -> list[str]:
    model_kwargs={'max_tokens': 4096, 'temperature': 0.0}
    retriever = get_aws_bedrock_retriever(KB_ID, top_rec=8)
    contexts: list[list[Document]] = await retriever.abatch(
        list(summarize(text) if len(text) > 1000 else text for text in
        (req['EXCEPTIONS / SPECIAL REQUIREMENTS'] for req in requirements))
    )
    inputs = [
        {
            'model_id': model_id,
            'prompt': prompt_template.format(
                examples=examples_str,
                context=str().join('\n\n'.join(doc.page_content for doc in docs)),
                requirements=req['EXCEPTIONS / SPECIAL REQUIREMENTS']
            ),
            **model_kwargs
        } for docs, examples_str, req in zip(contexts, examples, requirements)
    ]

    logger.info('Len LLM Inputs: %s', len(inputs))
    results = await arun_batch(invoke_model, inputs)    
    regex = '{.*?}'
    text = '\n\n'.join(result for result in results)
    return parse_llm_output_to_mapping(text, regex, default={})[0]


async def pull_requirements(
    model_id: str,
    prompt_template: str,
    docs: list[Document],
    *,
    examples: str | None = None
) -> tuple[list[dict], list[str]]:
    batched_docs: list[list[Document]] = list(batched(docs, 5))
    inputs: list[dict] = []
    model_kwargs={'max_tokens': 4096, 'temperature': 0.0}

    for batch in batched_docs:
        llm_input = {'context': '\n'.join(get_page_string(doc) for doc in batch)}

        if examples:
            llm_input['examples'] = examples

        inputs.append(
            {
                'model_id': model_id,
                'prompt': prompt_template.format(**llm_input),
                **model_kwargs
            }
        )

    #save_prompts_local(prompt, llm_inputs)
    response: list[str] = await arun_batch(invoke_model, inputs)
    regex = '(?<=<requirement>).*?(?=</requirement>)'
    matches = []
    errors = []

    for result in response:
        ms, errs = parse_llm_output_to_mapping(result.replace('\n', ''), regex)
        matches.extend(ms)
        errors.extend(errs)

    if errors:
        logger.error('Errors: %s', errors)
        #save_errors(errors, TEMP_SAVE_DIR)

    return matches, errors
