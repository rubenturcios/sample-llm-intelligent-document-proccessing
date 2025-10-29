#!/usr/bin/env python3
import config
import aws_cdk as cdk

from src.cdk_resources.llm_poc_pipeline_stack import LLMPocPipelineStack


app = cdk.App()
LLMPocPipelineStack(
    app,
    "LLMPocPipelineStack",
    # Uncomment the next line to specialize this stack for the AWS Account
    # and Region that are implied by the current CLI configuration.
    # env=cdk.Environment(account=os.getenv('CDK_DEFAULT_ACCOUNT'), region=os.getenv('CDK_DEFAULT_REGION')),
    # Stack configured to account and region in config file. (stage is configured the same in pipeline stack)
    env=cdk.Environment(account=config.ACCOUNT, region=config.REGION),
)

app.synth()
