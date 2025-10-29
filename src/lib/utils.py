import asyncio
from enum import StrEnum, auto
from datetime import datetime
from functools import partial
from io import BytesIO
from itertools import islice
import logging
import json
import os
from typing import Any, Callable, Iterable, TypedDict

import boto3
import pandas as pd
from nltk import word_tokenize, sent_tokenize, download
from nltk.data import path

from .constants import TEMP_SAVE_DIR


logger = logging.getLogger(__name__)
logger.setLevel(level=os.getenv('LOG_LEVEL', logging.INFO))

session = boto3.Session()
credentials = session.get_credentials()
s3_client = session.client('s3')


Rule = TypedDict(
    'Rule',
    {
        '#': int,
        'COMM0N ITEM': str,
        'TOPIC': str,
        'REQUIREMENT': str,
        'RECOMMENDATION': str,
        'NOTES TO CUSTOMER': str | None,
        'INTERNAL NOTES': str | None,
        'DEPT. RESPONSIBLE': str | None,
        'sheet': str | None,
        '_id': int | None,
    }
)


class SaveType(StrEnum):
    csv = auto()
    xlsx = auto()


def load_s3_file(bucket: str, path_to_file: str) -> BytesIO:
    s3_data = s3_client.get_object(Bucket=bucket, Key=path_to_file)
    return BytesIO(s3_data['Body'].read())


def save_to_s3(bucket: str, path_to_file: str, stream: str | bytes) -> None:
    put_object_response = s3_client.put_object(
        Bucket=bucket,
        Key=path_to_file,
        Body=stream
    )
    logger.info(put_object_response)


def batched(iterable, n) -> Iterable:
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)

    while batch := tuple(islice(it, n)):
        yield batch


async def arun_batch(fn: Callable, inputs: list[dict]) -> list[Any]:
    loop = asyncio.get_event_loop()
    return await asyncio.gather(
        *[loop.run_in_executor(None, partial(fn, **i)) for i in inputs]
    )


def save_df(
    df: pd.DataFrame,
    file_name: str,
    dir_: str,
    *,
    save_type: SaveType = SaveType.csv
) -> None:
    with BytesIO() as output:
        if save_type == 'csv':
            df.to_csv(output)
        else:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer)

        data = output.getvalue()

    if not os.path.exists(dir_):
        os.makedirs(dir_)

    file_name = '/'.join((dir_, file_name))
    file = open(file_name, 'w')
    file.write(data)
    file.close()


def save_errors(errors: list[str], dir_: str) -> None:
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    file_name = '/'.join((dir_, f'Error_Matches_{datetime.now()}.json'))
    file = open(file_name, 'w')
    json.dump(errors, file)
    return file.close()


def save_output(output: str, dir_) -> None:
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    file_name = '/'.join((dir_, f'Output_{datetime.now()}.txt'))
    file = open(file_name, 'w')
    file.write(output)
    return file.close()


def summarize(text: str, *, max_len: int = 1000) -> str:
    path.append(TEMP_SAVE_DIR)
    download('punkt', quiet=True, download_dir=TEMP_SAVE_DIR)
    download('stopwords', quiet=True, download_dir=TEMP_SAVE_DIR)
    from nltk.corpus import stopwords

    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    freq_table = dict()

    for word in words:
        word = word.lower()
        if word in stop_words:
            continue
        if word in freq_table:
            freq_table[word] += 1
        else:
            freq_table[word] = 1

    sentences: list[str] = sent_tokenize(text)
    sentence_value = dict()

    for sentence in sentences:
        for word, freq in freq_table.items():
            if word in sentence.lower():
                if sentence in sentence_value:
                    sentence_value[sentence] += freq
                else:
                    sentence_value[sentence] = freq

    sum_values = 0

    for sentence in sentence_value: sum_values += sentence_value[sentence]

    average = int(sum_values / len(sentence_value))
    summary = ''

    for sentence in sentences:
        if (
            (sentence in sentence_value) and
            (sentence_value[sentence] > (1.2 * average))
        ):
            summary += ' ' + sentence

    logger.info('Len Summary: %s', len(summary))

    if not len(summary):
        return text[:max_len]
    elif len(summary) > max_len:
        return summary[:max_len]
    else:
        return summary
