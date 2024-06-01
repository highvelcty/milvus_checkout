from openai import OpenAI, AsyncOpenAI
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.core.callbacks.base import CallbackManager
from typing import Optional, Dict, Any, List
from datetime import datetime
import requests
import json


class NeMoEmbedding(NVIDIAEmbedding):
    def __init__(
        self,
        model: str = "NV-Embed-QA",
        embed_batch_size: int = 100,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_retries: int = 10,
        timeout: float = 60.0,
        callback_manager: Optional[CallbackManager] = None,
        default_headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ):
        if embed_batch_size > 259:
            raise ValueError("The batch size should not be larger than 259.")

        self._client = OpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers
        )

        self._aclient = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers
        )

        super(NVIDIAEmbedding, self).__init__(
            model=model,
            embed_batch_size=embed_batch_size,
            callback_manager=callback_manager,
            **kwargs,
        )

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        assert len(texts) <= 259, "The batch size should not be larger than 259."

        data = self._client.embeddings.create(
            input=texts, model=self.model, extra_body={"input_type": "passage", "truncate": self.truncate}
        ).data
        return [d.embedding for d in data]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Asynchronously get query embedding."""
        return (
            (
                await self._aclient.embeddings.create(
                    input=[query],
                    model=self.model,
                    extra_body={"input_type": "query", "truncate": self.truncate},
                )
            )
            .data[0]
            .embedding
        )


def is_before_date(date1: str, date2: str) -> bool:
    """
    Check if date1 is before date2.

    Args:
    date1 (str): First date string in "yyyy-mm-dd" format.
    date2 (str): Second date string in "yyyy-mm-dd" format.

    Returns:
    bool: True if date1 is before date2, False otherwise.
    """

    datetime1 = datetime.strptime(date1, '%Y-%m-%d')
    datetime2 = datetime.strptime(date2, '%Y-%m-%d')
    return datetime1 < datetime2


def days_after_1900(date_string: str) -> int:
    """
    Convert a date string in "yyyy-mm-dd" format to the number of days after 1900-01-01.

    Args:
    date_string (str): Date string in "yyyy-mm-dd" format.

    Returns:
    int: Number of days after 1900-01-01.
    """
    from datetime import datetime

    date = datetime.strptime(date_string, "%Y-%m-%d")
    base_date = datetime(1900, 1, 1)
    delta = date - base_date
    return delta.days

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def login_aioli(aioli_host, username, password):
    endpoint = f"http://{aioli_host}:80/api/v1/login"
    response = requests.post(endpoint, json={"username": username, "password": password})
    assert response.status_code == 200, f"Failed to login to Aioli: {response.text}"
    result = json.loads(response.text)
    return result["token"]

def get_deployments(aioli_host, token):
    endpoint = f"http://{aioli_host}:80/api/v1/deployments"
    header = {"Authorization": f"Bearer {token}"}
    response = requests.get(endpoint, headers=header)
    assert response.status_code == 200, f"Failed to get deployments from Aioli: {response.text}"
    return json.loads(response.text)

def get_models(aioli_host, token):
    endpoint = f"http://{aioli_host}:80/api/v1/models"
    header = {"Authorization": f"Bearer {token}"}
    response = requests.get(endpoint, headers=header)
    assert response.status_code == 200, f"Failed to get models from Aioli: {response.text}"
    return json.loads(response.text)

def get_api_key(aioli_host, token, model_name):
    models = get_models(aioli_host, token)
    model_id = None
    for m in models:
        if m["name"] == model_name:
            print(m)
            model_id = m["id"]
            break
    if model_id is None:
        raise ValueError(f"Model {model_name} not found.")

    endpoint = f"http://{aioli_host}:80/api/v1/models/{model_id}/token"
    header = {"Authorization": f"Bearer {token}"}
    response = requests.get(endpoint, headers=header)
    assert response.status_code == 200, f"Failed to get API key from Aioli: {response.text}"
    result = json.loads(response.text)
    return result["token"]

def get_model_hostname(aioli_host, token, model_name):
    endpoint = f"http://{aioli_host}:80/api/v1/deployments"
    header = {"Authorization": f"Bearer {token}"}
    response = requests.get(endpoint, headers=header)
    assert response.status_code == 200, f"Failed to get deployments from Aioli: {response.text}"
    deployments = json.loads(response.text)
    for m in deployments:
        if m["name"] == model_name:
            try:
                return remove_prefix(m["state"]["endpoint"], "http://")
            except Exception as e:
                raise ValueError(f"Failed to get hostname for model {model_name}: {e}")
    raise ValueError(f"Model {model_name} not found.")
