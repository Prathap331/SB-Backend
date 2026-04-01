from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from supabase import create_client, Client
from dotenv import load_dotenv
import os

load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    raise ValueError("Supabase credentials not found in .env")

supabase: Client = create_client(supabase_url, supabase_key)

# Swagger will call POST /token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# 🔐 LOGIN ENDPOINT LOGIC (for Swagger form)
async def login_user(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Takes username + password from Swagger form.
    Uses Supabase to authenticate.
    Returns access_token for Swagger.
    """
    try:
        response = supabase.auth.sign_in_with_password({
            "email": form_data.username,  # Swagger uses 'username' field
            "password": form_data.password
        })

        session = response.session

        if not session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return {
            "access_token": session.access_token,
            "refresh_token": getattr(session, "refresh_token", None),
            "expires_in": getattr(session, "expires_in", None),
            "token_type": "bearer",
        }

    except Exception as e:
        print(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def refresh_access_token(refresh_token: str) -> dict:
    """
    Exchange a valid refresh_token for a new access_token.
    """
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="refresh_token is required",
        )
    try:
        # supabase-py accepts refresh token as argument in current SDK versions.
        response = supabase.auth.refresh_session(refresh_token)
        session = getattr(response, "session", None)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return {
            "access_token": session.access_token,
            "refresh_token": getattr(session, "refresh_token", refresh_token),
            "expires_in": getattr(session, "expires_in", None),
            "token_type": "bearer",
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Refresh token error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# 🔐 VERIFY TOKEN FOR PROTECTED ROUTES
async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Validates Bearer token sent from Swagger or frontend.
    """
    try:
        user_response = supabase.auth.get_user(token)
        user = user_response.user

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return user

    except Exception as e:
        print(f"Token validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
