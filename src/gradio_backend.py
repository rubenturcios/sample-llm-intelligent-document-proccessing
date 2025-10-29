import asyncio
from inspect import get_annotations
from datetime import UTC, datetime
import json
import logging
from functools import partial
from pprint import pformat
from typing import Any, Callable, Optional, TypedDict
from random import sample
import os
from io import BytesIO

import gradio as gr
from gradio.utils import NamedString
import numpy as np
import pandas as pd

from langchain_core.documents import Document

from lib.constants import (
    BID_CONTEXT_INDEX,
    BID_REVIEW_INDEX,
    BID_REVIEW_PREFIX,
    BID_REVIEWS,
    BUCKET,
    EMBEDDING_MODEL,
    EMBEDDING_MODELS_DIMENSIONS,
    RULES_INDEX_NAME,
    SPEC_SHEET_PREFIX,
    SPEC_SHEETS,
    SPEC_SHEETS_INDEX,
    UPLOAD_DIR,
    VECTOR_FIELD_NAME,
    WORKBOOK,
)
from prompts import (
    ATC_PROMPT,
    DEFAULT_SPEC_SHEET_PROMPT,
    EXAMPLE_FORMAT
)
from lib.opensearch import (
    create_index,
    default_text_mappings,
    put_os_entries,
    get_os_entries,
    check_index_exist,
)
from lib.bedrock import (
    classify,
    pull_requirements,
    get_opensearch_helper,
    embed_rule_requirements,
    aembed_rule_requirements
)
from lib.load import (
    Method,
    load_s3_file,
    upload_to_os,
    pdfminer_load,
    get_rules_from_wb,
    get_spec_sheet_docs,
    get_bid_review_docs,
    unstructructed_load,
    get_spec_sheet_matches,
    get_requirements_from_wb,
    get_matches_by_embedding_vector,
)
from lib.utils import Rule, SaveType, save_to_s3


format = '%(levelname)s:%(filename)s:%(lineno)d:%(asctime)s:%(message)s'
logging.basicConfig(format=format)
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv('LOG_LEVEL', logging.INFO))


GenRequirement = TypedDict(
    'GenRequirement',
    {
        'NUM': str,
        'PAGE NUMBER': str,
        'SECTION / SUBSECTION': str,
        'EXCEPTIONS / SPECIAL REQUIREMENTS': str,
    }
)
GenClassification = TypedDict(
    'GenClassification',
    {
        "NUM": str,
        "RECOMMENDATION": str,
        "NOTES TO CUSTOMER": str,
        "INTERNAL NOTES": str,
        "DEPT. RESPONSIBLE": str
    }
)


class Backend:
    def __init__(self):
        self.bucket_name = "walter-p-knowledgebase-test-103985810001"
        self.table_name = "llm_poc_ddb_table"

    def setup(self) -> None:
        vector_field_dimensions = EMBEDDING_MODELS_DIMENSIONS.get(EMBEDDING_MODEL)
        index_mappings = {
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
            'mappings': {
                'properties': {
                    VECTOR_FIELD_NAME: {
                        "type": "knn_vector",
                        "dimension": vector_field_dimensions,
                        "method": {
                            "engine": "faiss",
                            "space_type": "l2",
                            "name": "hnsw",
                        }
                    }
                }
            }
        }

        if not check_index_exist(RULES_INDEX_NAME):
            self._index_rules(index_mappings)

        if not check_index_exist(SPEC_SHEETS_INDEX):
            self._index_spec_sheets()

        if not check_index_exist(BID_REVIEW_INDEX):
            self._index_bid_reviews(index_mappings)

        if not check_index_exist(BID_CONTEXT_INDEX):
            index_mappings = default_text_mappings()
            create_index(BID_CONTEXT_INDEX, default_mappings=index_mappings)

    def _index_bid_reviews(self, index_mappings) -> None:
        create_index(BID_REVIEW_INDEX, default_mappings=index_mappings)

        for bid_review in BID_REVIEWS:
            bid_review_name = BID_REVIEW_PREFIX + bid_review['BID_REVIEW']
            filename = load_s3_file(BUCKET, bid_review_name)
            requirements = get_requirements_from_wb(filename, iter(bid_review['SPEC_SHEETS']))

            for req in requirements:
                if requirement := req.get('EXCEPTIONS / SPECIAL REQUIREMENTS'):
                    req['vector_field'] = embed_rule_requirements(requirement, EMBEDDING_MODEL)
                else:
                    logger.warning('Requirement has no EXCEPTION: %s', pformat(req))

            put_os_entries(BID_REVIEW_INDEX, requirements)

    def _index_spec_sheets(self) -> None:
        for file in SPEC_SHEETS:
            file_name = SPEC_SHEET_PREFIX + file

            if file_name.endswith('.pdf'):
                docs = pdfminer_load(bucket=BUCKET, filename=file_name)
            elif file_name.endswith('.docx') or file_name.endswith('.doc'):
                docs = unstructructed_load(bucket=BUCKET, filename=file_name)

            upload_to_os(docs, SPEC_SHEETS_INDEX)

    def _index_rules(self, index_mappings) -> None:
        filename = load_s3_file(BUCKET, WORKBOOK)
        rules = get_rules_from_wb(filename)

        for rule in rules:
            if requirement := rule.get('REQUIREMENT'):
                rule['vector_field'] = embed_rule_requirements(requirement, EMBEDDING_MODEL)
            else:
                logger.warning('Rule has no REQUIREMENT: %s', pformat(rule))

        create_index(RULES_INDEX_NAME, default_mappings=index_mappings)
        put_os_entries(RULES_INDEX_NAME, rules)

    @staticmethod
    def get_rules_from_os(sheet_name: str, *, size: int = 10000) -> list[dict]:
        query = {
            'query': {
                'term': {'sheet.keyword': sheet_name}
            },
            "sort": {
                "#": "asc"
            },
            'size': size
        }
        return get_os_entries(RULES_INDEX_NAME, query)

    @staticmethod
    def save_results(
        pulled_reqs: pd.DataFrame,
        classified: pd.DataFrame,
        matched_rules: pd.DataFrame,
        matched_errors: list[str],
        file: str | NamedString,
        save_type: SaveType = SaveType.csv
    ) -> tuple[gr.Button, list]:
        new = pd.concat(
            [
                pulled_reqs.drop(['NUM'], axis=1),
                classified.drop(['NUM'], axis=1)
            ],
            axis=1
        )
        df_file_names = (
            'Results_{}' + f'.{save_type}',
            'Rules_Matched_{}' + f'.{save_type}',
            'Parsing_Errors_{}.json'
        )
        dfs = (new, matched_rules)
        prefix = f'{UPLOAD_DIR.strip("/")}/{str(datetime.now(tz=UTC))}/'
        base_name = (
            file.name.split('/')[-1].split('.')[-2]
            if isinstance(file, NamedString) else
            file.split('/')[-1].split('.')[-2]
        )

        for file_name, df in zip(df_file_names, dfs):
            with BytesIO() as output:
                if save_type == 'csv':
                    df.to_csv(output)
                else:
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        df.to_excel(writer)

                save_to_s3(
                    bucket=BUCKET,
                    path_to_file=(prefix + file_name.format(base_name)),
                    stream=output.getvalue()
                )

        if matched_errors:
            save_to_s3(
                bucket=BUCKET,
                path_to_file=(prefix + file_name.format(base_name)),
                stream=json.dumps(matched_errors).encode()
            )

        return gr.Button(value='Saved', interactive=False), []

    @classmethod
    def df_to_dict_list(
        cls,
        values: pd.DataFrame,
        *,
        sheet_name: Optional[str] = None
    ) -> list[dict]:
        keys = list(values.keys())
        rules = []

        for new_rule in values.values:
            rule_template = {key: value for key, value in zip(keys, new_rule)}

            if sheet_name:
                rule = cls.pre_process_rule_entry_for_os(rule_template, sheet_name)
            else:
                rule = rule_template

            rules.append(rule)

        return rules

    @staticmethod
    def highlight_fn(row: pd.Series, score_threshold: float) -> list[str]:
        if float(row['SCORE'] or 0.0) >= score_threshold:
            return [
                'background: green; color: black;' 
                if key == 'SCORE' else 'background: lightgreen; color: black;'
                for key in row.to_dict()
            ]
        else:
            return [
                '' for _ in row.to_dict()
            ]

    @classmethod
    def new_rule_input_function(
        cls,
        input_values: pd.DataFrame,
        *,
        index_name: str,
        sheet_name: str
    ) -> list[list]:
        rules = cls.df_to_dict_list(input_values, sheet_name=sheet_name)

        for rule in rules:
            if requirement := rule.get('REQUIREMENT'):
                rule['vector_field'] = embed_rule_requirements(requirement, EMBEDDING_MODEL)
            else:
                logger.warning('Requirement has no EXCEPTION: %s', pformat(rule))

        put_os_entries(index_name, [rule for rule in rules if rule.get('vector_field')])
        rules_raw = cls.get_rules_from_os(sheet_name)
        rules_processed = cls.post_process_rule_entry_for_ui(rules_raw)
        return [list(rule.values()) for rule in rules_processed]

    @classmethod
    def edit_rule_input_function(
        cls,
        input_values: pd.DataFrame,
        *,
        index_name: str,
        sheet_name: str
    ) -> list[list]:
        modified_rules = cls.df_to_dict_list(input_values, sheet_name=sheet_name)
        entries = [{'doc': rule, '_id': rule.pop('_id')} for rule in modified_rules]
        put_os_entries(
            index_name,
            [entry for entry in entries if entry.get('vector_field')],
            type='update'
        )
        return input_values

    @staticmethod
    def _match_req_to_rule(
        reqs_embeddings: list[list[float]],
        gen_requirements: list[dict]
    ) -> list[dict]:
        os_helper = get_opensearch_helper(EMBEDDING_MODEL, RULES_INDEX_NAME)
        matched_rules = []

        for embedding, req in zip(reqs_embeddings, gen_requirements):
            #text = req['EXCEPTIONS / SPECIAL REQUIREMENTS']
            if embedding:
                relevant_docs_with_score = (
                    os_helper
                    .similarity_search_with_score_by_vector(
                        embedding=embedding, k=1, text_field='REQUIREMENT'
                        #=text, text_field='REQUIREMENT'
                    )
                )
                top_rule_with_score = relevant_docs_with_score[0]
                top_rule, score = top_rule_with_score
                rule = {'NUM': req['NUM'], 'SCORE': score}
                _ = top_rule.metadata.pop('vector_field')
                rule.update(top_rule.metadata)
            else:
                rule = {'NUM': req.get('NUM')}

            logger.info('Matched Rule: %s', rule)
            matched_rules.append(rule)

        return matched_rules

    @staticmethod
    def _msearch_match_rules(
        reqs_embeddings: list[list[float]],
        gen_requirements: list[dict],
        **query_args: Any
    ) -> list[list[dict]]:
        hits = get_matches_by_embedding_vector(
            RULES_INDEX_NAME,
            reqs_embeddings,
            vector_field_name=VECTOR_FIELD_NAME,
            **query_args
        )
        matched_rules = []

        for req, hit in zip(gen_requirements, hits):
            matched_ = []

            for hit_ in hit:
                rule = {
                    'NUM': req['NUM'],
                    'SCORE': float(hit_['_score'])
                }
                source = hit_['_source']
                _ = source.pop('vector_field')
                rule.update(source)
                matched_.append(rule)

            matched_rules.append(matched_)

        return matched_rules

    @classmethod
    async def match_requirements(
        cls, 
        input_values: pd.DataFrame,
        score_threshold: float,
    ) -> tuple[gr.DataFrame, gr.DataFrame]:
        async def empty_list():
            return []

        gen_requirements = cls.df_to_dict_list(input_values)
        reqs_embeddings: list[list[float]] = await asyncio.gather(
            *[
                aembed_rule_requirements(
                    requirement, EMBEDDING_MODEL
                ) if (requirement := req.get('EXCEPTIONS / SPECIAL REQUIREMENTS')) else
                empty_list() for req in gen_requirements 
            ]
        )
        reqs_embedding_df_values = [
            [index, embedding] for index, embedding in enumerate(reqs_embeddings)
        ]
        reqs_embedding_df = gr.DataFrame(
            headers=['NUM', 'vector_field'],
            value=reqs_embedding_df_values
        )
        #matched_rules = cls._match_req_to_rule(reqs_embeddings, gen_requirements)
        matched_rules = [
            rules[0] for rules in cls._msearch_match_rules(reqs_embeddings, gen_requirements)
        ]
        fn = partial(cls.highlight_fn, score_threshold=score_threshold)
        return (
            cls.get_rules_df(
                matched_rules,
                'Matched Rules',
                highlight_fn=fn
            ),
            reqs_embedding_df
        )

    @staticmethod
    def _get_true_positives(
        reqs_embeddings: list[list[float]],
        spec_sheet: str,
        threshold_score: float,
    ) -> int:
        hits = [
            hit[0] for hit in get_spec_sheet_matches(
                BID_REVIEW_INDEX,
                reqs_embeddings,
                spec_sheet,
                score=threshold_score
            ) if hit
        ]
        return len(hits)

    @staticmethod
    def _get_false_positives(
        length_of_true_positives: int,
        length_of_total_requirements: int
    ) -> int:
        return length_of_total_requirements - length_of_true_positives

    @staticmethod
    def _get_false_negatives(
        spec_sheet_embeddings: list[list[float]],
        gen_reqs_embeddings: list[list],
        threshold_score: float
    ) -> int:
        negatives = 0

        for spec_sheet_embedding, gen_req_embedding in zip(
            spec_sheet_embeddings, gen_reqs_embeddings
        ):
            a = np.array(spec_sheet_embedding)
            b = np.array(gen_req_embedding)
            dist = np.linalg.norm(a-b) # euclidean distance (l2 norm)
            score = 1/(1 + dist) # Equation used by Opensearch to determine similarity.
            if score < threshold_score: negatives +=1

        return negatives

    @classmethod
    def calculate_scores(
        cls,
        gen_reqs_embedding_df: pd.DataFrame,
        spec_sheet: str,
        threshold_score: float
    ) -> gr.DataFrame:
        true_positives, false_negatives, false_positives = (
            cls.get_scores(
                gen_reqs_embedding_df,
                spec_sheet,
                threshold_score
            )
        )
        logger.info('True Positives: %s', true_positives)
        logger.info('False Negatives: %s', false_negatives)
        logger.info('False Positives: %s', false_positives)

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = (2 * precision * recall) / (precision + recall)

        return gr.DataFrame(
            headers=['Precision', 'Recall', 'F1'],
            value=[[precision, recall, f1]]
        )

    @classmethod
    def get_scores(
        cls, gen_reqs_embedding_df, spec_sheet, threshold_score
    ) -> tuple[int, int, int]:
        spec_sheet_embeddings: list[list[float]] = [
            value.metadata[VECTOR_FIELD_NAME]
            for value in get_bid_review_docs(BID_REVIEW_INDEX, spec_sheet)
        ]
        gen_reqs_embeddings: list[list[float]] = [
            value[VECTOR_FIELD_NAME] for value in cls.df_to_dict_list(gen_reqs_embedding_df)
        ]
        true_positives = cls._get_true_positives(
            gen_reqs_embeddings, spec_sheet, threshold_score
        )
        false_negatives = cls._get_false_negatives(
            spec_sheet_embeddings, gen_reqs_embeddings, threshold_score
        )
        false_positives = cls._get_false_positives(
            true_positives, len(gen_reqs_embeddings)
        )
        return true_positives, false_negatives, false_positives

    @staticmethod
    def get_example_rule_string(rule: Rule) -> str:
        try:
            return EXAMPLE_FORMAT.format(pformat(rule))
        except Exception as e:
            logger.error('Rule Failed formatting: %s', pformat(rule))
            return ''

    @classmethod
    def get_classify_examples(
        cls,
        gen_reqs: list[dict],
        gen_reqs_embeddings: list[list[float]],
    ) -> list[str]:
        keys = [
            'NUM',
            'REQUIREMENT',
            'RECOMMENDATION',
            'NOTES TO CUSTOMER',
            'INTERNAL NOTES',
            'DEPT. RESPONSIBLE'
        ]
        rules_hits = cls._msearch_match_rules(gen_reqs_embeddings, gen_reqs, k=4)
        examples: list[str] = []

        for rules in rules_hits:
            examples_formatted = (
                {k: v for k, v in rule.items() if k in keys} for rule in rules
            )
            example_string = '\n\n'.join(
                cls.get_example_rule_string(rule) for rule in examples_formatted
            )
            examples.append(example_string)

        return examples

    @classmethod
    async def classify_requirements(
        cls,
        model_id: str,
        prompt: str,
        generated_reqs: pd.DataFrame,
        genereted_reqs_embeddings: pd.DataFrame,
        matched_rules: pd.DataFrame,
        score: float, 
    ) -> gr.DataFrame:
        labels = list(get_annotations(GenClassification).keys())
        reqs = cls.df_to_dict_list(generated_reqs)
        reqs_embeddings = [
            row[VECTOR_FIELD_NAME] for row in 
            cls.df_to_dict_list(genereted_reqs_embeddings)
        ]
        matched = cls.df_to_dict_list(matched_rules)
        default_examples = cls.get_classify_examples(reqs, reqs_embeddings)
        examples = [
            cls.get_example_rule_string(match)
            if float(match['SCORE'] or 0.0) >= score
            else default for match, default
            in zip(matched, default_examples, strict=True)
        ]
        response: list[dict] = await classify(model_id, prompt, examples, reqs)
        logger.info(
            'Len Pulled Reqs: %s, Len Matched: %s, Len Response: %s',
            len(reqs),
            len(matched),
            len(response)
        )
        recommendations = []

        for index, requirement in enumerate(response):
            row = [index] + (
                list(requirement[key] for key in labels[1:])
                if requirement else ['PARSING ERROR' for i in labels[1:]]
            )
            recommendations.append(row)

        return gr.DataFrame(headers=labels, value=recommendations) 

    @staticmethod
    def pre_process_rule_entry_for_os(rule: dict, sheet_name: str) -> dict:
        rule['sheet'] = sheet_name
        return rule

    @staticmethod
    def post_process_rule_entry_for_ui(
        rules: dict | list[dict]
    ) -> Rule | list[Rule]:
        def process(rule: dict) -> Rule:
            if source := rule.get('_source'):
                new_rule = source
            else:
                new_rule = rule
            
            if id := rule.get('_id'):
                new_rule['_id'] = id
            if new_rule.get('sheet'):
                _ = new_rule.pop('sheet')
            
            return new_rule

        if isinstance(rules, list):
            return [process(rule) for rule in rules]
        else:
            return process(rules)

    @classmethod
    def get_rules_df(
        cls, 
        rules_raw: dict | list[dict],
        label: str,
        *,
        highlight_fn: Optional[Callable] = None,

    ) -> gr.DataFrame:
        rules_processed = cls.post_process_rule_entry_for_ui(rules_raw)
        values = [list(rule.values()) for rule in rules_processed]
        headers = tuple(rules_processed[0].keys())

        if not highlight_fn:
            return gr.DataFrame(
                value=values,
                label=label,
                headers=headers,
                interactive=False
            )

        styler = (
            pd.DataFrame(
                data=values,
                columns=headers,
            )
            .replace(np.nan, None)
            .style
            .apply(highlight_fn, axis=1)
        )
        df = gr.DataFrame(styler)
        df.label = label
        df.headers = headers
        return df

    @staticmethod
    def get_prompt(spec_sheet: str) -> str:
        if 'ATC' in spec_sheet:
            return ATC_PROMPT
        else:
            return DEFAULT_SPEC_SHEET_PROMPT

    @staticmethod
    def show_upload_button(prompt: str, model_id: str) -> tuple[str, gr.UploadButton]:
        if prompt and model_id:
            button = gr.UploadButton(visible=True)
        else:
            button = gr.UploadButton(visible=False)

        return prompt, button

    @staticmethod
    def get_sample_docs_string(
        spec_sheets: list[Document],
        file_name: str, 
        *,
        docs: list[Document] | None = None
    ):
        def get_doc_string(doc: Document) -> str:
            keys = ['PAGE', 'SECTION / SUB', 'EXCEPTIONS / SPECIAL REQUIREMENTS']
            formatted_doc = pformat({k: v for k, v in doc.metadata.items() if k in keys})
            return EXAMPLE_FORMAT.format(formatted_doc).replace('\'', '\"')

        example_docs = [] if not docs else docs
        logger.info('FILENAME: %s', file_name)

        for spec_sheet in spec_sheets:
            if spec_sheet == file_name:
                logger.info('SKIPPING: %s', spec_sheet)
            else:
                logger.info('ADDING: %s', spec_sheet)
                docs = get_bid_review_docs(BID_REVIEW_INDEX, spec_sheet)
                example_docs.extend(docs)

        return '\n\n'.join(get_doc_string(doc) for doc in sample(example_docs, k=3))

    @classmethod
    async def pull_reqs(
        cls, model_id: str, prompt: str, file_name: str, state: Any
    ) -> tuple[gr.Row, gr.DataFrame, list[str]]:
        labels = list(get_annotations(GenRequirement).keys())
        filename = SPEC_SHEET_PREFIX + file_name

        if file_name.endswith('.doc') or file_name.endswith('.docx'):
            docs = get_spec_sheet_docs(SPEC_SHEETS_INDEX, filename, method=Method.UNSTRUCTUED)
        else:
            docs = get_spec_sheet_docs(SPEC_SHEETS_INDEX, filename)
        
        examples = cls.get_sample_docs_string(SPEC_SHEETS, file_name)
        matches, state = await pull_requirements(model_id, prompt, docs, examples=examples)
        requirements = []

        for index, requirement in enumerate(matches):
            try:
                row = [index] + (
                    list(requirement[key] for key in labels[1:])
                    if requirement else []
                )
            except Exception as e:
                logger.error(requirement)
                raise e
            requirements.append(row)

        row = gr.Row(visible=True)
        df = gr.DataFrame(
            headers=labels,
            value=requirements
        )
        return row, df, state

    @classmethod
    async def dynamic_pull_reqs(
        cls,
        model_id: str,
        prompt: str,
        file: NamedString,
        state: Any
    ) -> tuple[gr.Row, gr.DataFrame, list]:
        labels = list(get_annotations(GenRequirement).keys())
        with open(file, 'rb') as f:
            if file.name.endswith('.pdf'):
                logger.info('Loading PDF...')
                docs = pdfminer_load(bucket=BUCKET, stream=f.read())
            elif file.name.endswith('.docx') or file.name.endswith('.doc'):
                logger.info('Loading DOC/DOCX')
                docs = unstructructed_load(bucket=BUCKET, stream=f.read())

        examples = cls.get_sample_docs_string(SPEC_SHEETS, file_name=f.name)
        matches, state = await pull_requirements(model_id, prompt, docs, examples=examples)
        requirements = []

        for index, requirement in enumerate(matches):
            try:
                row = [index] + (
                    list(requirement[key] for key in labels[1:])
                    if requirement else []
                )
            except Exception as e:
                logger.error(requirement)
                raise e
            requirements.append(row)

        row = gr.Row(visible=True)
        df = gr.DataFrame(
            headers=labels,
            value=requirements
        )
        return row, df, state
