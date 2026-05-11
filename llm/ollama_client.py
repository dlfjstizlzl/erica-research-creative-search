"""Robust Async Ollama chat and embedding client."""

from __future__ import annotations

import ast
import asyncio
import json
import re
from typing import Any

import aiohttp

from config import OLLAMA_HOST, OLLAMA_MODEL, REQUEST_TIMEOUT


class AsyncOllamaClient:
    """Async wrapper around Ollama's API with robust JSON parsing and self-correction."""

    _semaphore: asyncio.Semaphore | None = None

    @classmethod
    def _get_semaphore(cls) -> asyncio.Semaphore:
        if cls._semaphore is None:
            # Increase to 2 for better speed, but watch out for VRAM issues
            cls._semaphore = asyncio.Semaphore(2)
        return cls._semaphore

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        host: str = OLLAMA_HOST,
        timeout: int = REQUEST_TIMEOUT,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def chat_json(
        self,
        user_prompt: str,
        system_prompt: str = "",
        debug_label: str | None = None,
        num_predict: int | None = None,
        max_retries: int = 2,
    ) -> Any:
        async with self._get_semaphore():
            full_prompt = (
                f"{system_prompt}\n\n{user_prompt}\n\n"
                f"[CRITICAL: Return ONLY valid JSON. Do NOT include any explanations or markdown blocks like ```json. "
                "Do NOT use unescaped double quotes (\") inside strings; use single quotes (') instead.]"
            )

            payload = {
                "model": self.model,
                "stream": True,
                "messages": [
                    {"role": "user", "content": full_prompt.strip()},
                ],
            }
            if num_predict is not None:
                payload["options"] = {"num_predict": num_predict}

            content = ""
            for attempt in range(max_retries + 1):
                try:
                    content = await self._stream_post(
                        payload, endpoint="/api/chat", debug_label=debug_label
                    )
                    return self._parse_json_content(content)
                except Exception as exc:
                    if attempt == max_retries:
                        print(
                            f"[client] Final JSON parsing failed after {max_retries} retries for {debug_label}: {exc}"
                        )
                        print(f"         Raw Output Snippet: {repr(content[:1000])}")
                        return {}
                    
                    print(
                        f"[client] JSON parsing failed on attempt {attempt+1}/{max_retries} for {debug_label}: {exc}. "
                        "Retrying with self-correction..."
                    )
                    
                    correction_prompt = (
                        f"The previous output failed to parse as JSON. Error: {exc}. "
                        "Fix the format and return ONLY valid JSON."
                    )
                    payload["messages"] = [
                        {"role": "user", "content": full_prompt.strip()},
                        {"role": "assistant", "content": f"```json\n{content}\n```"},
                        {"role": "user", "content": correction_prompt},
                    ]
                    if num_predict:
                        payload["options"]["num_predict"] = num_predict + 200
            return {}

    async def embed(self, inputs: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        async with self._get_semaphore():
            payload = {
                "model": self.model,
                "input": inputs,
            }
            body = await self._post(payload, endpoint="/api/embed")

        embeddings = body.get("embeddings")
        if not isinstance(embeddings, list):
            raise RuntimeError("Unexpected Ollama embedding response format.")
        return [
            [float(value) for value in vector]
            for vector in embeddings
            if isinstance(vector, list)
        ]

    async def _stream_post(
        self,
        payload: dict[str, Any],
        *,
        endpoint: str = "/api/chat",
        debug_label: str | None = None,
    ) -> str:
        url = f"{self.host}{endpoint}"
        full_content = ""
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.content:
                        if line:
                            chunk = json.loads(line.decode('utf-8'))
                            
                            # 'message' 필드가 있으면 content 추출
                            msg = chunk.get("message", {})
                            content = msg.get("content", "")
                            
                            # 만약 'thinking' 모델인 경우, 생각하는 도중에는 content가 없을 수 있음
                            # 하지만 JSON 답변은 보통 content에 들어오므로 일단 누적
                            full_content += content
                            
                            if chunk.get("done"):
                                if not full_content:
                                    # 만약 'thinking' 필드만 있고 content가 비어있다면 경고
                                    if msg.get("thinking"):
                                        print(f"⚠️ [client] {self.model} spent tokens on thinking, but output content is empty.")
                                break
                    return full_content
            except aiohttp.ClientResponseError as exc:
                raise RuntimeError(f"Ollama HTTP {exc.status}: {exc.message}") from exc
            except aiohttp.ClientError as exc:
                raise RuntimeError(f"Could not reach Ollama: {exc}") from exc
            except asyncio.TimeoutError as exc:
                raise RuntimeError(f"Ollama request timed out after {self.timeout.total}s.") from exc

    async def _post(
        self,
        payload: dict[str, Any],
        *,
        endpoint: str = "/api/chat",
        debug_label: str | None = None,
    ) -> dict[str, Any]:
        url = f"{self.host}{endpoint}"
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    return await response.json()
            except aiohttp.ClientResponseError as exc:
                raise RuntimeError(f"Ollama HTTP {exc.status}: {exc.message}") from exc
            except aiohttp.ClientError as exc:
                raise RuntimeError(f"Could not reach Ollama: {exc}") from exc
            except asyncio.TimeoutError as exc:
                raise RuntimeError(f"Ollama request timed out after {self.timeout.total}s.") from exc

    def _extract_message_content(self, body: dict[str, Any]) -> str:
        try:
            return str(body["message"]["content"])
        except (KeyError, TypeError) as exc:
            raise RuntimeError("Unexpected Ollama response format.") from exc

    def _parse_json_content(self, content: str) -> Any:
        try:
            # First try standard JSON
            return json.loads(content)
        except json.JSONDecodeError:
            # Clean common markdown artifacts
            clean_content = content.strip()
            if "```json" in clean_content:
                match = re.search(r"```json\s*([\s\S]*?)\s*```", clean_content)
                if match:
                    clean_content = match.group(1)
            elif "```" in clean_content:
                match = re.search(r"```\s*([\s\S]*?)\s*```", clean_content)
                if match:
                    clean_content = match.group(1)

            # Attempt extraction if there's surrounding text
            obj_match = re.search(r"\{[\s\S]*\}", clean_content)
            arr_match = re.search(r"\[[\s\S]*\]", clean_content)
            
            for match in (obj_match, arr_match):
                if match:
                    json_str = match.group(0)
                    # Some models (like Llama 3.1) output ""string"" instead of "string"
                    json_str = re.sub(r'""(.*?)""', r'"\1"', json_str)
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        # Fallback: Many LLMs return single-quoted dictionaries (pseudo-JSON)
                        # ast.literal_eval is excellent for parsing these safely
                        try:
                            # Also try to fix double-double quotes for ast.literal_eval
                            fixed_str = re.sub(r"'' (.*?) ''", r"'\1'", json_str)
                            return ast.literal_eval(fixed_str)
                        except Exception:  # noqa: BLE001
                            continue
                        
            raise RuntimeError("Model did not return valid JSON. Regex extraction also failed.")
