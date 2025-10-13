from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Dict, Any

from services.auth_service import AuthService

# Pydantic models for responses
class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class TokenVerifyRequest(BaseModel):
    token: str

class TokenVerifyResponse(BaseModel):
    username: str
    token_valid: bool
    expires_at: int

class ErrorResponse(BaseModel):
    detail: str

# Create router and service instance
auth_router = APIRouter()
auth_service = AuthService()

@auth_router.post("/login", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login endpoint that accepts username and password via form-data.
    Returns a JWT token on successful authentication.
    
    Credentials:
    - username: admin
    - password: meter123
    """
    try:
        result = await auth_service.login_user(form_data.username, form_data.password)
        return TokenResponse(**result)
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login error: {str(e)}"
        )

@auth_router.post("/verify-token", response_model=TokenVerifyResponse)
async def verify_token(request: TokenVerifyRequest):
    """
    Verify token endpoint that takes a token as input and verifies its validity.
    Returns 401 if invalid or expired.
    Returns user info if valid.
    """
    try:
        result = await auth_service.verify_user_token(request.token)
        return TokenVerifyResponse(**result)
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Token verification error: {str(e)}"
        )
