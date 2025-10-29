import os
import aws_cdk as cdk
import aws_cdk.aws_codecommit as codecommit
from constructs import Construct
from aws_cdk.pipelines import (
    CodePipeline,
    CodePipelineSource,
    ShellStep,
    ManualApprovalStep,
)

from src.cdk_resources.llm_poc_accelerator_stack import LLMPocAcceleratorStack
from src.cdk_resources.opensearch_serverless_stack import OpensearchCollectionStack
import config


class LLMPocPipelineAppStage(cdk.Stage):
    """
    This class represents a stage in the CDK pipeline, responsible for deploying the
    LLMPocAcceleratorStack.
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        """
        Initializes a new instance of the LLMPocPipelineAppStage.

        :param scope: The scope in which this stage is defined.
        :param construct_id: The ID of this stage.
        :param kwargs: Additional keyword arguments.
        """
        super().__init__(scope, construct_id, **kwargs)

        # Initialize and deploy the LLMPocAcceleratorStack within this stage.
        accelerator_stack = LLMPocAcceleratorStack(self, "LLMPocAcceleratorStack")
        _opensearch_stack = OpensearchCollectionStack(
            self,
            "OpenSearchCollectionStack",
            collection_name="llm-poc-vector-db",
            iam_access_arns=[
                f"arn:aws:iam::{config.ACCOUNT}:user/TensorIOT-rturcios",
                accelerator_stack.gradio_app_parameter.string_value
            ],
            env=cdk.Environment(account=config.ACCOUNT, region=config.BEDROCK_REGION)
        )


class LLMPocPipelineStack(cdk.Stack):
    """
    This class defines the main pipeline stack for the LLM POC Accelerator. It includes the
    setup for a CodeCommit repository and the steps necessary for deploying the application.
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        """
        Initializes a new instance of the LLMPocPipelineStack.

        :param scope: The scope in which this stack is defined.
        :param construct_id: The ID of this stack.
        :param kwargs: Additional keyword arguments.
        """
        super().__init__(scope, construct_id, **kwargs)

        # Determine the path for the code directory and the zip file containing the repository code.
        code_directory = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        code_zip = os.path.join(code_directory, "latest.zip")

        # Create a new CodeCommit repository using the zip file as the initial commit.
        repo = codecommit.Repository(
            self,
            config.NEW_REPO_NAME,
            repository_name=config.NEW_REPO_NAME,
            code=codecommit.Code.from_zip_file(code_zip),
        )

        # Define the pipeline with a single synth step that installs dependencies and synthesizes the CDK app.
        pipeline = CodePipeline(
            self,
            "AcceleratorPipeline",
            publish_assets_in_parallel=False,
            pipeline_name="AcceleratorPipeline",
            synth=ShellStep(
                "Synth",
                input=CodePipelineSource.code_commit(
                    repo, config.REPO_NOTEBOOK_BRANCH, code_build_clone_output=True
                ),
                commands=[
                    "npm install -g aws-cdk",
                    "python -m pip install -r requirements.txt",
                    "cdk synth --all",
                ],
            ),
        )

        # Add the application stage to the pipeline, specifying the deployment environment.
        stage = pipeline.add_stage(
            LLMPocPipelineAppStage(
                self,
                "POCDeploymentStage",
                env=cdk.Environment(account=config.ACCOUNT, region=config.REGION),
            )
        )

        # If manual approval is required, add a manual approval step before the deployment.
        if config.MANUAL_APPROVAL:
            stage.add_pre(ManualApprovalStep("ApprovalBeforeDeploy"))
