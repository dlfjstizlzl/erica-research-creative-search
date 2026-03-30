"""Minimal Ollama chat and embedding client."""

from __future__ import annotations

import json
import socket
import time
from typing import Any
from urllib import error, request

from config import OLLAMA_HOST, OLLAMA_MODEL, REQUEST_TIMEOUT


class OllamaClient:
    """Small wrapper around Ollama's API."""

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        host: str = OLLAMA_HOST,
        timeout: int = REQUEST_TIMEOUT,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def chat_json(
        self,
        user_prompt: str,
        system_prompt: str = "",
        debug_label: str | None = None,
        num_predict: int | None = None,
    ) -> Any:
        payload = {
            "model": self.model,
            "stream": True,
            "format": "json",
            "messages": [
                {"role": "system", "content": system_prompt or "Return only valid JSON."},
                {"role": "user", "content": user_prompt},
            ],
        }
        if num_predict is not None:
            payload["options"] = {"num_predict": num_predict}

        body = self._post(payload, stream=True, debug_label=debug_label)
        content = self._extract_message_content(body)
        try:
            return self._parse_json_content(content)
        except RuntimeError:
            if num_predict is None:
                raise

        retry_num_predict = max(num_predict * 2, num_predict + 200)
        retry_payload = dict(payload)
        retry_payload["options"] = {"num_predict": retry_num_predict}
        if debug_label:
            print(
                f"[stream] {debug_label} retrying with larger num_predict="
                f"{retry_num_predict} after JSON parse failure"
            )
        retry_body = self._post(
            retry_payload,
            stream=True,
            debug_label=f"{debug_label} retry" if debug_label else None,
        )
        retry_content = self._extract_message_content(retry_body)
        return self._parse_json_content(retry_content)

    def embed(self, inputs: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        payload = {
            "model": self.model,
            "input": inputs,
        }
        body = self._post(payload, endpoint="/api/embed")

        embeddings = body.get("embeddings")
        if not isinstance(embeddings, list):
            raise RuntimeError("Unexpected Ollama embedding response format.")
        return [
            [float(value) for value in vector]
            for vector in embeddings
            if isinstance(vector, list)
        ]

    def _post(
        self,
        payload: dict[str, Any],
        *,
        endpoint: str = "/api/chat",
        stream: bool = False,
        debug_label: str | None = None,
    ) -> dict[str, Any]:
        req = request.Request(
            f"{self.host}{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                if stream:
                    return self._read_streaming_response(response, debug_label=debug_label)
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Could not reach Ollama: {exc.reason}") from exc
        except socket.timeout as exc:
            raise RuntimeError(
                f"Ollama request timed out after {self.timeout}s. "
                f"Consider increasing OLLAMA_TIMEOUT or using a smaller/faster model."
            ) from exc

    def _extract_message_content(self, body: dict[str, Any]) -> str:
        try:
            return str(body["message"]["content"])
        except (KeyError, TypeError) as exc:
            raise RuntimeError("Unexpected Ollama response format.") from exc

    def _read_streaming_response(
        self,
        response: Any,
        *,
        debug_label: str | None = None,
    ) -> dict[str, Any]:
        content_parts: list[str] = []
        started_at = time.perf_counter()
        last_reported_chars = 0
        first_token_logged = False

        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            chunk = json.loads(line)
            message = chunk.get("message", {})
            piece = message.get("content", "")
            if piece:
                content_parts.append(str(piece))
                if debug_label and not first_token_logged:
                    elapsed = time.perf_counter() - started_at
                    print(f"[stream] {debug_label} first token after {elapsed:.1f}s")
                    first_token_logged = True
                if debug_label:
                    current_chars = sum(len(part) for part in content_parts)
                    if current_chars - last_reported_chars >= 200:
                        elapsed = time.perf_counter() - started_at
                        print(
                            f"[stream] {debug_label} "
                            f"received {current_chars} chars in {elapsed:.1f}s"
                        )
                        last_reported_chars = current_chars
            if chunk.get("done"):
                break

        if debug_label:
            elapsed = time.perf_counter() - started_at
            total_chars = sum(len(part) for part in content_parts)
            print(f"[stream] {debug_label} completed with {total_chars} chars in {elapsed:.1f}s")
        return {"message": {"content": "".join(content_parts)}}

    def _parse_json_content(self, content: str) -> Any:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            for open_char, close_char in (("[", "]"), ("{", "}")):
                start = content.find(open_char)
                end = content.rfind(close_char)
                if start != -1 and end != -1 and end > start:
                    return json.loads(content[start : end + 1])
        raise RuntimeError("Model did not return valid JSON.")
