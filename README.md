# Sample LLM Intelligent Document Processing

A proof-of-concept project demonstrating intelligent document processing for contract bid reviews using AWS services and Large Language Models.

## Overview

This sample project showcases a real-world use case where organizations need to efficiently review and process bid documents for ongoing contracts. The solution leverages AWS infrastructure, OpenSearch for document indexing and retrieval, and LLM capabilities through Amazon Bedrock to provide intelligent document analysis.

## Key Components

### Infrastructure (CDK)

- **llm_poc_accelerator_stack.py** - Main infrastructure stack for the LLM POC
- **llm_poc_pipeline_stack.py** - CI/CD pipeline configuration
- **opensearch_serverless_stack.py** - OpenSearch Serverless setup for document indexing

### Application Layer

- **gradio_frontend.py** - User interface built with Gradio
- **gradio_backend.py** - Backend logic for processing requests
- **gradio_main.py** - Main application entry point

### Core Libraries

- **bedrock.py** - Integration with Amazon Bedrock LLM services
- **opensearch.py** - OpenSearch operations for document storage and retrieval
- **load.py** - Document loading and preprocessing utilities
- **utils.py** - Common utility functions
- **prompts.py** - LLM prompt templates and configurations

### Deployment

- **GradioECSDockerfile** - Docker configuration for ECS deployment
- **GradioLambdaDockerfile** - Docker configuration for Lambda deployment

## Features

- Intelligent document processing and analysis
- Natural language querying of contract documents
- Vector search capabilities through OpenSearch
- Interactive web interface built with Gradio
- Scalable deployment options (ECS or Lambda)
- Infrastructure as Code using AWS CDK

## Prerequisites

- AWS Account with appropriate permissions
- AWS CDK installed and configured
- Python 3.11+
- Docker (for containerized deployments)

## Installation

1. Clone the repository
2. Install dependencies:
```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
```

3. Configure AWS credentials:
```bash
   aws configure
```

4. Deploy the infrastructure:
```bash
   cdk deploy
```

## Usage

The application provides an interactive interface for:
- Uploading contract and bid documents
- Querying document contents using natural language
- Extracting key information from documents
- Comparing and analyzing multiple documents

## Architecture

This solution uses:
- **Amazon Bedrock** for LLM capabilities
- **OpenSearch Serverless** for vector storage and semantic search
- **AWS Lambda/ECS** for compute
- **Gradio** for the user interface
- **AWS CDK** for infrastructure management
