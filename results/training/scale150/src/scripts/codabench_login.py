#!/usr/bin/env python3
"""Authenticate to Codabench from an interactive terminal and save a private API token.

Passwords/tokens are entered with hidden prompts, never as command arguments.
Only the API token is saved; the password is not persisted.
"""
import argparse
import getpass
import json
import os
from pathlib import Path
import sys
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token", action="store_true", help="Enter an existing API token instead of username/password")
    parser.add_argument("--output", type=Path, default=Path.home() / ".config/fusion/codabench_token")
    args = parser.parse_args()
    if not sys.stdin.isatty():
        parser.exit(1, "Run this helper yourself in an interactive server terminal.\n")
    if args.token:
        token = getpass.getpass("Codabench API token (hidden): ").strip()
    else:
        username = input("Codabench username: ").strip()
        password = getpass.getpass("Codabench password (hidden): ")
        if not username or not password:
            parser.exit(1, "Username and password are required; nothing was saved.\n")
        request = Request("https://www.codabench.org/api/api-token-auth/",
                          data=json.dumps({"username": username, "password": password}).encode(),
                          headers={"Content-Type": "application/json", "Accept": "application/json"},
                          method="POST")
        try:
            with urlopen(request, timeout=30) as response:
                data = json.load(response)
        except HTTPError as exc:
            parser.exit(1, f"Codabench login failed (HTTP {exc.code}); nothing was saved.\n")
        except (URLError, TimeoutError):
            parser.exit(1, "Could not reach Codabench; nothing was saved.\n")
        del password, request
        token = data.get("token", "")
    if not isinstance(token, str) or not token or any(c.isspace() for c in token):
        parser.exit(1, "No valid token was provided or returned; nothing was saved.\n")
    output = args.output.expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=output.parent, prefix=".codabench_", delete=False) as stream:
            temporary = Path(stream.name)
            os.fchmod(stream.fileno(), 0o600)
            stream.write(token + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"API token saved privately to {output}")
    print("Tell the assistant that Codabench login is ready. Do not send the token in chat.")


if __name__ == "__main__":
    main()
