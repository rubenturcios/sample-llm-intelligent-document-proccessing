from aws_cdk import (
    Stack,
    aws_iam as iam,
    aws_sagemaker as sagemaker,
    aws_codecommit as codecommit,
    Duration,
    CfnOutput,
    Fn,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_s3 as s3,
    aws_ssm as ssm
)
from constructs import Construct

from aws_cdk.aws_lambda import (
    DockerImageFunction,
    DockerImageCode,
    Architecture,
    FunctionUrlAuthType,
)
import config


class LLMPocAcceleratorStack(Stack):
    """
    Defines the AWS Cloud Development Kit (CDK) stack for deploying the LLM POC Accelerator.
    This stack includes AWS resources necessary for running SageMaker instances, Lambda functions,
    ECS services, and optionally, Amazon OpenSearch Serverless for knowledge base functionalities.
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        """
        Initializes a new instance of the LLMPocAcceleratorStack.

        :param scope: The scope in which this stack is defined, typically an App or a Stage.
        :param construct_id: A unique identifier for this stack.
        :param kwargs: Additional keyword arguments passed to the base class.
        """

        # call parent init
        super().__init__(scope, construct_id, **kwargs)

        # Deploy SageMaker notebook
        #self.deploy_notebook()

        # Deploy Gradio interface on AWS Lambda if configured
        if config.DEPLOY_GRADIO_ON_LAMBDA:
            self.deploy_gradio_lambda()
        # Deploy Gradio interface on Amazon ECS if configured
        if config.DEPLOY_GRADIO_ON_ECS:
            self.deploy_ecs()
        

    ############################################################################################
    # Class methods below.
    ############################################################################################

    def deploy_notebook(self):
        """
        Creates and configures the AWS SageMaker notebook instance, including its associated IAM role,
        policies for necessary AWS service access, and a CodeCommit repository for storing notebook files.
        Also sets up a lifecycle configuration to pre-install Python libraries.
        """

        # Define the SageMaker execution role
        self.sagemaker_execution_role = iam.Role(
            self,
            "SageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="Role for SageMaker to access services",
        )

        # Attach necessary managed policies to the role
        self.sagemaker_execution_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonS3FullAccess")
        )
        self.sagemaker_execution_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess")
        )

        # Create an inline policy for additional service permissions
        self.notebook_policy_statement = iam.PolicyStatement(
            actions=["bedrock:*", "transcribe:*"],
            resources=["*"],
            effect=iam.Effect.ALLOW,
        )

        # Add the inline policy to the role
        self.sagemaker_execution_role.add_to_policy(self.notebook_policy_statement)

        # Create a reference to the CodeCommit repository
        repo = codecommit.Repository.from_repository_name(
            self, config.NEW_REPO_NAME, config.NEW_REPO_NAME
        )
        repo.grant_pull_push(self.sagemaker_execution_role)

        # Register the repository with SageMaker for use with the notebook instance
        self.code_repository = sagemaker.CfnCodeRepository(
            self,
            "LLMPocAcceleratorCodeRepository",
            git_config=sagemaker.CfnCodeRepository.GitConfigProperty(
                repository_url=repo.repository_clone_url_http,
                branch=config.REPO_NOTEBOOK_BRANCH,
            ),
            code_repository_name="LLMPocAcceleratorCodeRepository",
        )

        # Define and create a Notebook Lifecycle Configuration for pre-installing libraries
        lifecycle_config_content = f"""#!/bin/bash
set -e

# The Activate the conda python environment in SageMaker
sudo -u ec2-user -i <<'EOF'
source /home/ec2-user/anaconda3/bin/activate python3
pip install -r /home/ec2-user/SageMaker/{repo.repository_name}/requirements.txt
pip install -r /home/ec2-user/SageMaker/{repo.repository_name}/requirements-dev.txt
source /home/ec2-user/anaconda3/bin/deactivate
EOF
"""
        # Create the Lifecycle Configuration using the script above
        self.lifecycle_config = sagemaker.CfnNotebookInstanceLifecycleConfig(
            self,
            "LifecycleConfig",
            on_start=[
                sagemaker.CfnNotebookInstanceLifecycleConfig.NotebookInstanceLifecycleHookProperty(
                    content=Fn.base64(lifecycle_config_content)
                )
            ],
        )

        # Create the SageMaker notebook instance with the specified configuration
        self.notebook_instance = sagemaker.CfnNotebookInstance(
            self,
            "LLMPocAcceleratorNotebook",
            instance_type=config.NOTEBOOK_INSTANCE_SIZE,
            notebook_instance_name="LLMPocAcceleratorNotebook",
            role_arn=self.sagemaker_execution_role.role_arn,
            lifecycle_config_name=self.lifecycle_config.attr_notebook_instance_lifecycle_config_name,
            default_code_repository=self.code_repository.attr_code_repository_name,
        )

    def deploy_gradio_lambda(self):
        """
        Creates and configures an AWS Lambda function with a Docker image for hosting a Gradio interface.
        It includes setting up necessary permissions and an HTTPS URL for accessing the Gradio UI.
        """
        # Define the policy for Lambda function permissions
        lambda_policy_statement = iam.PolicyStatement(
            actions=["bedrock:*", "transcribe:*"],
            resources=["*"],
            effect=iam.Effect.ALLOW,
        )

        # Create the Docker-based Lambda function for the Gradio application
        self.gradio_lambda_fn = DockerImageFunction(
            self,
            "GradioApp",
            code=DockerImageCode.from_image_asset("./src", file="GradioLambdaDockerfile"),
            architecture=Architecture.X86_64,
            memory_size=3008,
            timeout=Duration.minutes(5)
        )
        # Enable an HTTPS URL for the Lambda function without authentication
        self.gradio_lambda_fn_url = self.gradio_lambda_fn.add_function_url(
            auth_type=FunctionUrlAuthType.NONE
        )
        self.gradio_app_parameter = ssm.StringParameter(
            self,
            'GradioAppServiceRoleArn',
            string_value=self.gradio_lambda_fn.role.role_arn
        )
        CfnOutput(self, "FunctionUrl", value=self.gradio_lambda_fn_url.url)

        _ = s3.Bucket.from_bucket_arn(
            self, 'InputBucket', config.INPUT_BUCKET_ARN
        ).grant_read_write(self.gradio_lambda_fn)
        # Add the policy to the Lambda function's role for necessary permissions
        self.gradio_lambda_fn.add_to_role_policy(lambda_policy_statement)

    def deploy_ecs(self):
        """
        Creates and configures an Amazon ECS (Elastic Container Service) infrastructure for hosting a Gradio application.
        It includes setting up a VPC, an ECS cluster, a Fargate task definition with the Gradio container, and an
        Application Load Balanced Fargate Service for public access.
        """

        # Create a VPC
        vpc = ec2.Vpc(self, "Vpc", max_azs=2)

        # Create an ECS cluster
        cluster = ecs.Cluster(self, "GradioCluster", vpc=vpc)

        # Create an IAM role for the Fargate task with admin rights
        task_role = iam.Role(
            self,
            "GradioTaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AdministratorAccess")
            ],
        )

        # Define the Fargate task definition with a single container
        task_definition = ecs.FargateTaskDefinition(
            self, "GradioTaskDef", task_role=task_role
        )

        container = task_definition.add_container(
            "GradioContainer",
            image=ecs.ContainerImage.from_asset("./src", file="GradioECSDockerfile"),
            environment={"GRADIO_SERVER_PORT": "8080"},
            logging=ecs.LogDriver.aws_logs(stream_prefix="Gradio"),
        )

        container.add_port_mappings(ecs.PortMapping(container_port=8080))

        # Create a Fargate service running on the cluster, exposed via a public load balancer
        fargate_service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            "GradioService",
            cluster=cluster,
            task_definition=task_definition,
            public_load_balancer=True,
        )

        # Access the target group created by the Fargate service
        target_group = fargate_service.target_group

        # Enable stickiness on the target group
        target_group.enable_cookie_stickiness(duration=Duration.hours(1))
