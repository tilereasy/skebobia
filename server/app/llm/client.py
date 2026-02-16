from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from json import JSONDecodeError, JSONDecoder
from typing import Any

from openai import OpenAI


def _is_enabled(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class LLMClient:
    enabled: bool
    base_url: str
    model: str
    api_key: str | None
    timeout_sec: float = 6.0
    max_output_tokens: int = 900
    max_retries: int = 0
    debug: bool = False
    _sdk_client: OpenAI | None = None

    @classmethod
    def from_env(cls) -> "LLMClient":
        enabled = _is_enabled(os.getenv("LLM_DECIDER_ENABLED", "0"))
        base_url = os.getenv("LLM_DECIDER_BASE_URL", "https://bothub.chat/api/v2/openai/v1").strip()
        model = os.getenv("LLM_DECIDER_MODEL", "").strip()
        api_key = os.getenv("LLM_DECIDER_API_KEY", "").strip() or None

        try:
            timeout_sec = float(os.getenv("LLM_DECIDER_TIMEOUT_SEC", "6"))
        except ValueError:
            timeout_sec = 6.0
        timeout_sec = max(1.0, min(timeout_sec, 180.0))

        try:
            max_output_tokens = int(os.getenv("LLM_DECIDER_MAX_OUTPUT_TOKENS", "900"))
        except ValueError:
            max_output_tokens = 900
        max_output_tokens = max(64, min(max_output_tokens, 9000))

        try:
            max_retries = int(os.getenv("LLM_DECIDER_MAX_RETRIES", "0"))
        except ValueError:
            max_retries = 0
        max_retries = max(0, min(max_retries, 5))

        debug = _is_enabled(os.getenv("LLM_DECIDER_DEBUG", "0"))

        return cls(
            enabled=enabled and bool(base_url) and bool(model) and bool(api_key),
            base_url=base_url.rstrip("/"),
            model=model,
            api_key=api_key,
            timeout_sec=timeout_sec,
            max_output_tokens=max_output_tokens,
            max_retries=max_retries,
            debug=debug,
        )

    def request_json_object(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        temperature: float = 0.2,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        response_obj = self._responses_create(
            system_prompt=system_prompt,
            user_payload=user_payload,
            temperature=temperature,
            json_schema=json_schema,
        )
        if response_obj is None:
            self._debug("LLM request failed: no JSON response")
            return None

        content = self._extract_message_content(response_obj)
        if not content:
            self._debug("LLM response has no assistant text content")
            return None

        parsed = self._extract_json_object(content)
        if parsed is None:
            self._debug(f"LLM returned non-JSON text prefix: {content[:280]!r}")
        return parsed

    def _build_input_text(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
    ) -> str:
        user_text = json.dumps(user_payload, ensure_ascii=False)
        return f"{system_prompt}\n\nUSER_CONTEXT_JSON:\n{user_text}"

    def _get_sdk_client(self) -> OpenAI:
        if self._sdk_client is None:
            self._sdk_client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout_sec,
                max_retries=self.max_retries,
            )
        return self._sdk_client

    def _responses_create(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        temperature: float,
        json_schema: dict[str, Any] | None = None,
    ) -> Any | None:
        text_format: dict[str, Any] = {"type": "json_object"}
        if json_schema:
            text_format = {
                "type": "json_schema",
                "name": "tick_decision_envelope",
                "strict": True,
                "schema": json_schema,
            }

        try:
            response = self._get_sdk_client().responses.create(
                model=self.model,
                input=self._build_input_text(system_prompt=system_prompt, user_payload=user_payload),
                temperature=max(0.0, min(float(temperature), 1.0)),
                max_output_tokens=self.max_output_tokens,
                text={"format": text_format},
            )
        except Exception as exc:
            self._debug(f"LLM SDK error type={type(exc).__name__} detail={exc!r}")
            return None

        try:
            as_dict = response.model_dump()
        except Exception:
            as_dict = {"response_repr": repr(response)}
        self._debug(
            "LLM SDK response "
            f"status={as_dict.get('status')!r} "
            f"incomplete={as_dict.get('incomplete_details')!r} "
            f"output_types={self._output_types(as_dict)!r} "
            f"prefix={json.dumps(as_dict, ensure_ascii=False)[:280]!r}"
        )
        return response

    def _debug(self, message: str) -> None:
        if self.debug:
            logging.getLogger("app.llm.client").warning(message)

    def _extract_message_content(self, response_obj: Any) -> str | None:
        output_text = getattr(response_obj, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        as_dict = None
        try:
            as_dict = response_obj.model_dump()
        except Exception:
            as_dict = None

        if isinstance(as_dict, dict):
            output = as_dict.get("output")
            if isinstance(output, list):
                response_text = self._extract_from_responses_output(output)
                if response_text:
                    return response_text

            status = as_dict.get("status")
            incomplete = as_dict.get("incomplete_details")
            self._debug(
                "LLM response has no extractable text "
                f"status={status!r} incomplete={incomplete!r} output_types={self._output_types(as_dict)!r}"
            )

        # Backward-compatible parser for chat.completions-like dicts.
        if isinstance(response_obj, dict):
            choices = response_obj.get("choices")
            if not isinstance(choices, list) or not choices:
                return None

            first_choice = choices[0]
            if not isinstance(first_choice, dict):
                return None

            message = first_choice.get("message")
            if not isinstance(message, dict):
                return None
            content = message.get("content")

            if isinstance(content, str):
                return content.strip()

            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                merged = "".join(parts).strip()
                return merged if merged else None

        return None

    def _output_types(self, response_dict: dict[str, Any]) -> list[str]:
        output = response_dict.get("output")
        if not isinstance(output, list):
            return []
        items: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if isinstance(item_type, str):
                items.append(item_type)
        return items

    def _extract_from_responses_output(self, output: list[Any]) -> str | None:
        chunks: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for content_item in content:
                if not isinstance(content_item, dict):
                    continue
                content_type = content_item.get("type")
                text = content_item.get("text")
                if content_type in {"output_text", "text"} and isinstance(text, str):
                    chunks.append(text)
        merged = "".join(chunks).strip()
        return merged if merged else None

    def _extract_json_object(self, content: str) -> dict[str, Any] | None:
        normalized = content.strip()
        if not normalized:
            return None

        if normalized.startswith("```"):
            lines = [line for line in normalized.splitlines() if not line.strip().startswith("```")]
            normalized = "\n".join(lines).strip()

        try:
            parsed = json.loads(normalized)
            if isinstance(parsed, dict):
                return parsed
        except JSONDecodeError:
            pass

        start = normalized.find("{")
        if start < 0:
            return None
        decoder = JSONDecoder()
        try:
            parsed, _idx = decoder.raw_decode(normalized[start:])
        except JSONDecodeError as exc:
            self._debug(
                "JSON decode failed "
                f"msg={exc.msg!r} pos={exc.pos} len={len(normalized)} "
                f"prefix={normalized[:180]!r} suffix={normalized[-180:]!r}"
            )
            return None
        return parsed if isinstance(parsed, dict) else None
