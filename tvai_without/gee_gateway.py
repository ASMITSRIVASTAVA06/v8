import os
import json
import ee
from dotenv import load_dotenv

# Load environment variables from the .env file when running locally
# On Streamlit Cloud, these come from the Secrets manager instead
load_dotenv()


def _get_secret(key: str):
    """
    Look up a config value from, in order:
    1. Streamlit Secrets (when running inside a deployed Streamlit app)
    2. Environment variables (.env file when running locally)
    """
    try:
        import streamlit as st
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key)


def connect_gee():
    """
    Initialises Google Earth Engine.

    Cloud (Streamlit Secrets): authenticates as a service account using
    GEE_SERVICE_ACCOUNT_JSON (the full JSON key, pasted as one secret) plus
    GEE_PROJECT_ID. No browser login needed, so this works headless.

    Local (.env): falls back to the credentials cached by running
    `earthengine authenticate` once on your machine. Only GEE_PROJECT_ID
    is needed in that case.
    """
    project_id = _get_secret("GEE_PROJECT_ID")
    if not project_id:
        raise ValueError(
            "GEE_PROJECT_ID not found. "
            "Create a .env file locally or add it to Streamlit Secrets."
        )

    service_account_json = _get_secret("GEE_SERVICE_ACCOUNT_JSON")
    if service_account_json is not None and not isinstance(service_account_json, str):
        # A key pasted as a TOML table arrives as a mapping rather than a string
        service_account_json = json.dumps(dict(service_account_json))
    service_account_json = (service_account_json or "").strip()

    # Fallback for organisations that forbid service-account keys: the contents of the
    # credentials file that `earthengine authenticate` writes on your own computer.
    user_credentials_json = _get_secret("GEE_CREDENTIALS_JSON")
    if user_credentials_json is not None and not isinstance(user_credentials_json, str):
        user_credentials_json = json.dumps(dict(user_credentials_json))
    user_credentials_json = (user_credentials_json or "").strip()

    try:
        if not service_account_json and user_credentials_json:
            from google.oauth2.credentials import Credentials
            info = json.loads(user_credentials_json)
            credentials = Credentials(
                None,
                refresh_token=info["refresh_token"],
                token_uri="https://oauth2.googleapis.com/token",
                client_id=info.get("client_id") or ee.oauth.CLIENT_ID,
                client_secret=info.get("client_secret") or ee.oauth.CLIENT_SECRET,
                scopes=info.get("scopes") or ee.oauth.SCOPES,
            )
            ee.Initialize(credentials, project=project_id)
            print(f"Connected to Earth Engine project {project_id} with a user refresh token")
        elif service_account_json:
            key_dict = json.loads(service_account_json)
            credentials = ee.ServiceAccountCredentials(
                key_dict["client_email"], key_data=service_account_json
            )
            ee.Initialize(credentials, project=project_id)
            print(f"Connected to Earth Engine as service account: {key_dict['client_email']}")
        else:
            # ee.Authenticate() is not called here because authentication
            # is done once manually on each machine using the CLI.
            ee.Initialize(project=project_id)
            print(f"Connected to Earth Engine project: {project_id}")
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"GEE_SERVICE_ACCOUNT_JSON is present but is not valid JSON ({e}). In Streamlit "
            "Secrets wrap the whole key file in triple SINGLE quotes so the \\n sequences "
            "inside private_key stay intact."
        )
    except Exception as e:
        if not service_account_json:
            raise RuntimeError(
                "No GEE_SERVICE_ACCOUNT_JSON secret was found, so the app tried a personal "
                "login, which does not exist on a server. Add it as a top-level entry (not "
                "under a [section]) in Settings > Secrets, save, then reboot the app. "
                f"Underlying error: {e}"
            )
        raise RuntimeError(f"Earth Engine initialisation failed: {e}")


if __name__ == "__main__":
    connect_gee()
    print("Connection test passed.")