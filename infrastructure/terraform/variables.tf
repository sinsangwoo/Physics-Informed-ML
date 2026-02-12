# Terraform variables for Physics-Informed ML

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "production"
  
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be dev, staging, or production."
  }
}

variable "api_memory" {
  description = "Lambda function memory in MB"
  type        = number
  default     = 2048
  
  validation {
    condition     = var.api_memory >= 128 && var.api_memory <= 10240
    error_message = "Memory must be between 128 and 10240 MB."
  }
}

variable "api_timeout" {
  description = "Lambda function timeout in seconds"
  type        = number
  default     = 60
  
  validation {
    condition     = var.api_timeout >= 1 && var.api_timeout <= 900
    error_message = "Timeout must be between 1 and 900 seconds."
  }
}

variable "enable_auto_scaling" {
  description = "Enable auto-scaling for Lambda"
  type        = bool
  default     = true
}

variable "min_instances" {
  description = "Minimum number of Lambda instances"
  type        = number
  default     = 1
}

variable "max_instances" {
  description = "Maximum number of Lambda instances"
  type        = number
  default     = 10
}

variable "allowed_cors_origins" {
  description = "Allowed CORS origins"
  type        = list(string)
  default     = ["*"]
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 7
}

variable "tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default = {
    Project     = "Physics-Informed-ML"
    ManagedBy   = "Terraform"
  }
}
