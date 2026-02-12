# Terraform configuration for Physics-Informed ML infrastructure
# Deploys backend API to AWS with auto-scaling

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    bucket = "physics-ml-terraform-state"
    key    = "prod/terraform.tfstate"
    region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
}

# Variables
variable "aws_region" {
  description = "AWS region"
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  default     = "production"
}

variable "api_memory" {
  description = "Lambda memory in MB"
  default     = 2048
}

variable "api_timeout" {
  description = "Lambda timeout in seconds"
  default     = 60
}

# Lambda function for API
resource "aws_lambda_function" "physics_ml_api" {
  filename      = "../lambda-deployment.zip"
  function_name = "physics-ml-api-${var.environment}"
  role          = aws_iam_role.lambda_exec.arn
  handler       = "handler.lambda_handler"
  runtime       = "python3.11"
  
  memory_size = var.api_memory
  timeout     = var.api_timeout
  
  environment {
    variables = {
      ENVIRONMENT = var.environment
      LOG_LEVEL   = "INFO"
    }
  }
  
  tags = {
    Environment = var.environment
    Project     = "Physics-Informed-ML"
  }
}

# IAM role for Lambda
resource "aws_iam_role" "lambda_exec" {
  name = "physics-ml-lambda-exec-${var.environment}"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_policy" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# API Gateway
resource "aws_apigatewayv2_api" "physics_ml" {
  name          = "physics-ml-api-${var.environment}"
  protocol_type = "HTTP"
  
  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["GET", "POST", "OPTIONS"]
    allow_headers = ["*"]
  }
}

resource "aws_apigatewayv2_integration" "lambda" {
  api_id             = aws_apigatewayv2_api.physics_ml.id
  integration_type   = "AWS_PROXY"
  integration_uri    = aws_lambda_function.physics_ml_api.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "default" {
  api_id    = aws_apigatewayv2_api.physics_ml.id
  route_key = "$default"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_apigatewayv2_stage" "prod" {
  api_id      = aws_apigatewayv2_api.physics_ml.id
  name        = "$default"
  auto_deploy = true
}

resource "aws_lambda_permission" "apigw" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.physics_ml_api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.physics_ml.execution_arn}/*/*"
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "api_logs" {
  name              = "/aws/lambda/${aws_lambda_function.physics_ml_api.function_name}"
  retention_in_days = 7
}

# Outputs
output "api_endpoint" {
  value       = aws_apigatewayv2_api.physics_ml.api_endpoint
  description = "API Gateway endpoint URL"
}

output "lambda_function_name" {
  value       = aws_lambda_function.physics_ml_api.function_name
  description = "Lambda function name"
}
