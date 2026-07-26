"""Small resilience helpers for Hugging Face model downloads."""


def retry_closed_hf_client(loader):
    """Retry once if Hugging Face retries through a closed HTTP client.

    Some Hugging Face Hub versions can close their shared ``httpx`` client
    after a transient connection failure, then reuse that client for the next
    retry. Resetting the session lets the following attempt create a new one.
    """
    try:
        return loader()
    except RuntimeError as error:
        if "client has been closed" not in str(error).lower():
            raise

        from huggingface_hub.utils import close_session

        close_session()
        return loader()
