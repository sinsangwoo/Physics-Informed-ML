"""AWS Lambda handler for Physics-Informed ML API.

Serverless deployment handler that wraps FastAPI application.
Supports:
- HTTP API Gateway v2
- Lambda function URLs
- Application Load Balancer
"""

import json
import base64
from mangum import Mangum
from physics_informed_ml.api.main import app

# Create Lambda handler
handler = Mangum(app, lifespan="off")


def lambda_handler(event, context):
    """Main Lambda entry point.
    
    Args:
        event: Lambda event (API Gateway, ALB, etc.)
        context: Lambda context
        
    Returns:
        API Gateway response
    """
    # Log request for debugging
    print(f"Event: {json.dumps(event)}")
    
    # Handle binary content (images, etc.)
    if event.get('isBase64Encoded', False):
        event['body'] = base64.b64decode(event['body'])
    
    # Process request
    response = handler(event, context)
    
    # Log response
    print(f"Response status: {response.get('statusCode')}")
    
    return response
