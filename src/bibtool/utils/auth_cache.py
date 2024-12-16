import json
import time
from pathlib import Path
from typing import Optional

from rich import print

CACHE_FILE = Path.home() / "auth_cache.json"


def request_user_auth(service_name: str, prompt_message: Optional[str] = None, overwrite=False) -> str:
    """Request user authentication for the given service.

    Args:
        service_name: The name of the service for which the user is being authenticated.

    Returns:
        The user's authentication token for the service.

    """
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f:
            auth_cache = json.load(f)

        if not overwrite and service_name in auth_cache:
            return auth_cache[service_name]
    else:
        auth_cache = {}

    if prompt_message:
        print(prompt_message)
    time.sleep(0.5)
    token = input(f"Enter authentification token for {service_name}:")
    auth_cache[service_name] = token
    with open(CACHE_FILE, "w") as f:
        json.dump(auth_cache, f)

    return token
