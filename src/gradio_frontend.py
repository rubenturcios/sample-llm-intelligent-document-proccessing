import logging
from functools import partial
import os

from fastapi import FastAPI
import gradio as gr
from gradio_backend import Backend

from lib.opensearch import (
    get_sheet_categories
)
from lib.constants import (
    RULES_INDEX_NAME,
    LLM_MODELS,
    SPEC_SHEETS,
)
from prompts import DEFAULT_SPEC_SHEET_PROMPT, CLASSIFIER_PROMPT


format = '%(levelname)s:%(filename)s:%(lineno)d:%(asctime)s:%(message)s'
logging.basicConfig(format=format)
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv('LOG_LEVEL', logging.INFO))


class Frontend:
    def __init__(self):
        self.backend = Backend()
        self.backend.setup()
        self.setup_ui()

    def launch(self, **kwargs):
        self.demo.launch(**kwargs)

    def setup_ui(self):
        with gr.Blocks() as self.demo:
            error_state = gr.State([])
            sheets = get_sheet_categories(RULES_INDEX_NAME)
            llm_tab = self.get_llm_tab(error_state)
            upload_tab = self.get_upload_tab(error_state)
            rules_tab = self.get_all_rules_tab(sheets)

        self.app = gr.mount_gradio_app(FastAPI(), self.demo, '/')

    def get_llm_tab(self, error_state: gr.State) -> gr.Tab:
        with gr.Tab(label='LLM') as tab:
            with gr.Row():
                model = gr.Dropdown(LLM_MODELS, label='Model', value=LLM_MODELS[0])
                spec_sheet = gr.Dropdown(SPEC_SHEETS, label='Spec Sheet', value=SPEC_SHEETS[0])
                score = gr.Number(label='Matching Threshold Score', value=0.5, minimum=0.1, maximum=0.9, step=0.1)
            with gr.Row():
                pull_prompt = gr.Textbox(label='Pull Requirements Prompt', value=DEFAULT_SPEC_SHEET_PROMPT)
            with gr.Row():
                classifier_prompt = gr.Textbox(label='Classify Requirements Prompt', value=CLASSIFIER_PROMPT)
            with gr.Row():
                submit_button = gr.Button(value='Submit', size='sm')
            with gr.Row(visible=False) as results_row:
                with gr.Column() as pull_reqs_col:
                    pulled_reqs = gr.DataFrame(label='Results', interactive=False)
                    pulled_reqs_embeddings = gr.DataFrame(visible=False)
                with gr.Column(visible=False) as class_col:
                    classifications = gr.DataFrame(
                    label='Recommended Classifications',
                    interactive=False
                )
            with gr.Row(visible=False) as matched_row:
                relevant_rules = gr.DataFrame(label='Matched Rules', interactive=False)
            with gr.Row(visible=False) as save_row:
                save_button = gr.Button(value='Save')
                save_button.click(
                    fn=self.backend.save_results,
                    inputs=[
                        pulled_reqs,
                        classifications,
                        relevant_rules,
                        error_state,
                        spec_sheet
                    ],
                    outputs=[save_button, error_state]
                )
            metrics_row = self.get_metrics_row(pulled_reqs_embeddings, spec_sheet)

            model.input(self.backend.get_prompt, inputs=[spec_sheet], outputs=[pull_prompt])
            spec_sheet.input(self.backend.get_prompt, inputs=[spec_sheet], outputs=[pull_prompt])
            pull_prompt.change(fn=lambda x: x, inputs=[pull_prompt], outputs=[pull_prompt])
            classifier_prompt.change(fn=lambda x: x, inputs=[classifier_prompt], outputs=[classifier_prompt])
            score.input(fn=lambda x: x, inputs=[score], outputs=[score])

            submit_button.click(
                fn=self.show_hidden_row,
                inputs=None,
                outputs=[results_row]
            ).then(
                fn=self.backend.pull_reqs,
                inputs=[model, pull_prompt, spec_sheet, error_state],
                outputs=[results_row, pulled_reqs, error_state]
            ).then(
                fn=self.show_hidden_row,
                inputs=None,
                outputs=[matched_row]
            ).then(
                fn=self.backend.match_requirements,
                inputs=[pulled_reqs, score],
                outputs=[relevant_rules, pulled_reqs_embeddings]
            ).then(
                fn=self.show_hidden_col,
                inputs=None,
                outputs=[class_col]
            ).then(
                fn=self.backend.classify_requirements,
                inputs=[
                    model,
                    classifier_prompt,
                    pulled_reqs,
                    pulled_reqs_embeddings,
                    relevant_rules,
                    score
                ],
                outputs=[classifications]
            ).then(
                fn=self.show_hidden_row,
                inputs=None,
                outputs=[save_row]
            ).then(
                fn=self.show_hidden_row,
                inputs=None,
                outputs=[metrics_row]
            )

        return tab

    def get_upload_tab(self, error_state: gr.State) -> list[dict]:
        with gr.Tab(label='Upload Spec Sheet') as tab:
            with gr.Row():
                model = gr.Dropdown(LLM_MODELS, label='Model', value=LLM_MODELS[0])
                score = gr.Number(label='Matching Threshold Score', value=0.5, minimum=0.1, maximum=0.9, step=0.1)
            with gr.Row():
                pull_prompt = gr.Textbox(label='Pull Requirements Prompt', value=DEFAULT_SPEC_SHEET_PROMPT)
            with gr.Row():
                classifier_prompt = gr.Textbox(label='Classify Requirements Prompt', value=CLASSIFIER_PROMPT)
            with gr.Row():
                upload_button = gr.UploadButton()
            with gr.Row(visible=False) as results_row:
                with gr.Column() as pull_reqs_col:
                    pulled_reqs = gr.DataFrame(label='Results', interactive=False)
                    pulled_reqs_embeddings = gr.DataFrame(visible=False)
                with gr.Column(visible=False) as class_col:
                    classifications = gr.DataFrame(
                    label='Recommended Classifications',
                    interactive=False
                )
            with gr.Row(visible=False) as matched_row:
                relevant_rules = gr.DataFrame(label='Matched Rules', interactive=False)
            with gr.Row(visible=False) as save_row:
                save_button = gr.Button(value='Save')
                save_button.click(
                    fn=self.backend.save_results,
                    inputs=[
                        pulled_reqs,
                        classifications,
                        relevant_rules,
                        error_state,
                        upload_button
                    ],
                    outputs=[save_button, error_state]
                )

            model.input(fn=lambda x: x, inputs=[model], outputs=[model])
            pull_prompt.change(fn=lambda x: x, inputs=[pull_prompt], outputs=[pull_prompt])
            classifier_prompt.change(fn=lambda x: x, inputs=[classifier_prompt], outputs=[classifier_prompt])
            score.input(fn=lambda x: x, inputs=[score], outputs=[score])

            upload_button.upload(
                fn=self.show_hidden_row,
                inputs=None,
                outputs=[results_row]
            ).then(
                fn=self.backend.dynamic_pull_reqs,
                inputs=[model, pull_prompt, upload_button, error_state],
                outputs=[results_row, pulled_reqs, error_state]
            ).then(
                fn=self.show_hidden_row,
                inputs=None,
                outputs=[matched_row]
            ).then(
                fn=self.backend.match_requirements,
                inputs=[pulled_reqs, score],
                outputs=[relevant_rules, pulled_reqs_embeddings]
            ).then(
                fn=self.show_hidden_col,
                inputs=None,
                outputs=[class_col]
            ).then(
                fn=self.backend.classify_requirements,
                inputs=[
                    model,
                    classifier_prompt,
                    pulled_reqs,
                    pulled_reqs_embeddings,
                    relevant_rules,
                    score
                ],
                outputs=[classifications]
            ).then(
                fn=self.show_hidden_row,
                inputs=None,
                outputs=[save_row]
            )

        return tab

    def get_rules_tabs(self, index_name:str, sheets: list[dict]) -> list[gr.Tab]:
        tabs = []

        for sheet in sheets:
            new_fn = partial(
                self.backend.new_rule_input_function,
                index_name=index_name,
                sheet_name=sheet['key']
            )
            edit_fn = partial(
                self.backend.edit_rule_input_function,
                index_name=index_name,
                sheet_name=sheet['key']
            )

            with gr.Tab() as tab:
                tab.label = sheet['key']
                rules = self.backend.get_rules_from_os(sheet['key'])
                df = self.backend.get_rules_df(rules, sheet['key'])
                df.interactive = True
                update_rules_button = gr.Button(value='Update', size='sm')
                update_rules_button.click(
                    fn=edit_fn,
                    inputs=[df],
                    outputs=[df]
                )

                with gr.Accordion(label='Add Rule', open=False):
                    input_row = gr.DataFrame(headers=df.headers)
                    add_rule_button = gr.Button(value='Add', size='sm')
                    add_rule_button.click(
                        fn=new_fn,
                        inputs=[input_row],
                        outputs=df
                    )

            tabs.append(tab)

        return tabs

    def get_all_rules_tab(self, sheets: list[dict]) -> gr.Tab:
        with gr.Tab(label='Rules') as tab:
            ...
            _ = self.get_rules_tabs(RULES_INDEX_NAME, sheets)
        
        return tab

    def get_metrics_row(
        self,
        pulled_reqs_embeddings: gr.DataFrame,
        spec_sheet_name: gr.Dropdown,
    ) -> gr.Row:
        with gr.Row(visible=False) as scores_row:
            threshold_score = gr.Number(
                label='Metrics Threshold Score',
                value=0.5,
                minimum=0.1,
                maximum=0.9,
                step=0.1
            )
            scores_button = gr.Button(value='Get Scores')
            scores = gr.DataFrame(label='Scores', visible=False)

        fn1 = lambda: (gr.Number(visible=False), gr.Button(visible=False))
        fn2 = lambda: gr.DataFrame(visible=True)

        scores_button.click(
            fn=fn1,
            inputs=None,
            outputs=[threshold_score, scores_button]
        ).then(
            fn=fn2,
            inputs=None,
            outputs=[scores]
        ).then(
            fn=self.backend.calculate_scores,
            inputs=[pulled_reqs_embeddings, spec_sheet_name, threshold_score],
            outputs=[scores]
        )

        return scores_row

    @staticmethod
    def show_hidden_row() -> gr.Row:
        return gr.Row(visible=True)

    @staticmethod
    def show_hidden_col() -> gr.Column:
        return gr.Column(visible=True)
