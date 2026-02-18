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
    timeout_sec: float = 30.0
    max_output_tokens: int = 350
    max_retries: int = 0
    debug: bool = False
    strict_json_schema: bool = False
    json_schema_parse_retries: int = 1
    _sdk_client: OpenAI | None = None
    _json_schema_supported: bool = True

    @classmethod
    def from_env(cls) -> "LLMClient":
        enabled = _is_enabled(os.getenv("LLM_DECIDER_ENABLED", "0"))
        base_url = os.getenv("LLM_DECIDER_BASE_URL", "https://bothub.chat/api/v2/openai/v1").strip()
        model = os.getenv("LLM_DECIDER_MODEL", "").strip()
        api_key = os.getenv("LLM_DECIDER_API_KEY", "").strip() or None

        try:
            timeout_sec = float(os.getenv("LLM_DECIDER_TIMEOUT_SEC", "30"))
        except ValueError:
            timeout_sec = 30.0
        timeout_sec = max(1.0, min(timeout_sec, 180.0))

        try:
            max_output_tokens = int(os.getenv("LLM_DECIDER_MAX_OUTPUT_TOKENS", "350"))
        except ValueError:
            max_output_tokens = 350
        max_output_tokens = max(64, min(max_output_tokens, 9000))

        try:
            max_retries = int(os.getenv("LLM_DECIDER_MAX_RETRIES", "0"))
        except ValueError:
            max_retries = 0
        max_retries = max(0, min(max_retries, 5))

        debug = _is_enabled(os.getenv("LLM_DECIDER_DEBUG", "0"))
        strict_json_schema_raw = os.getenv("LLM_DECIDER_STRICT_JSON_SCHEMA")
        strict_json_schema = (
            model.startswith("gpt-4o") if strict_json_schema_raw is None else _is_enabled(strict_json_schema_raw)
        )
        try:
            json_schema_parse_retries = int(os.getenv("LLM_DECIDER_JSON_SCHEMA_PARSE_RETRIES", "1"))
        except ValueError:
            json_schema_parse_retries = 1
        json_schema_parse_retries = max(0, min(json_schema_parse_retries, 3))

        return cls(
            enabled=enabled and bool(base_url) and bool(model) and bool(api_key),
            base_url=base_url.rstrip("/"),
            model=model,
            api_key=api_key,
            timeout_sec=timeout_sec,
            max_output_tokens=max_output_tokens,
            max_retries=max_retries,
            debug=debug,
            strict_json_schema=strict_json_schema,
            json_schema_parse_retries=json_schema_parse_retries,
        )

    def request_json_object(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        temperature: float = 0.2,
        json_schema: dict[str, Any] | None = None,
        minimum_output_tokens: int = 0,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        effective_max_output_tokens = max(
            self.max_output_tokens,
            max(0, int(minimum_output_tokens)),
        )
        effective_max_output_tokens = max(64, min(effective_max_output_tokens, 9000))

        effective_schema: dict[str, Any] | None = None
        if json_schema is not None and self._json_schema_supported:
            effective_schema = self._provider_compatible_schema(json_schema)
        if json_schema is not None and self.strict_json_schema and effective_schema is None:
            self._debug("LLM strict_json_schema enabled but json_schema is unavailable for provider")
            return None

        if effective_schema is not None:
            attempts = max(1, self.json_schema_parse_retries + 1)
            for attempt in range(attempts):
                response_obj = self._responses_create(
                    system_prompt=system_prompt,
                    user_payload=user_payload,
                    temperature=temperature,
                    json_schema=effective_schema,
                    max_output_tokens=effective_max_output_tokens,
                )
                if response_obj is None:
                    break
                parsed = self._extract_json_from_response(response_obj, prefer_structured=True)
                if parsed is not None:
                    return parsed
                self._debug(f"LLM responses json_schema parse failed attempt={attempt + 1}/{attempts}")

            for attempt in range(attempts):
                response_obj = self._chat_completions_create(
                    system_prompt=system_prompt,
                    user_payload=user_payload,
                    temperature=temperature,
                    json_schema=effective_schema,
                    max_output_tokens=effective_max_output_tokens,
                )
                if response_obj is None:
                    break
                parsed = self._extract_json_from_response(response_obj, prefer_structured=True)
                if parsed is not None:
                    return parsed
                self._debug(f"LLM chat.completions json_schema parse failed attempt={attempt + 1}/{attempts}")

            if self.strict_json_schema:
                self._debug("LLM strict_json_schema enabled: skip json_object fallback")
                return None

        self._debug("LLM responses fallback: retry with json_object format")
        response_obj = self._responses_create(
            system_prompt=system_prompt,
            user_payload=user_payload,
            temperature=temperature,
            json_schema=None,
            max_output_tokens=effective_max_output_tokens,
        )
        if response_obj is not None:
            parsed = self._extract_json_from_response(response_obj, prefer_structured=False)
            if parsed is not None:
                return parsed

        self._debug("LLM responses fallback: try chat.completions json_object format")
        response_obj = self._chat_completions_create(
            system_prompt=system_prompt,
            user_payload=user_payload,
            temperature=temperature,
            json_schema=None,
            max_output_tokens=effective_max_output_tokens,
        )
        if response_obj is not None:
            parsed = self._extract_json_from_response(response_obj, prefer_structured=False)
            if parsed is not None:
                return parsed

        self._debug("LLM request failed: no valid JSON object extracted")
        return None

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
        max_output_tokens: int | None = None,
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
            max_tokens = self.max_output_tokens if max_output_tokens is None else max_output_tokens
            response = self._get_sdk_client().responses.create(
                model=self.model,
                input=self._build_input_text(system_prompt=system_prompt, user_payload=user_payload),
                temperature=max(0.0, min(float(temperature), 1.0)),
                max_output_tokens=max_tokens,
                text={"format": text_format},
            )
        except Exception as exc:
            if json_schema is not None and self._is_response_format_schema_error(exc):
                self._json_schema_supported = False
                self._debug("LLM provider rejected json_schema in responses API; disabled for next requests")
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

    def _chat_completions_create(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        temperature: float,
        json_schema: dict[str, Any] | None = None,
        max_output_tokens: int | None = None,
    ) -> Any | None:
        response_format: dict[str, Any] = {"type": "json_object"}
        if json_schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "tick_decision_envelope",
                    "strict": True,
                    "schema": json_schema,
                },
            }

        try:
            max_tokens = self.max_output_tokens if max_output_tokens is None else max_output_tokens
            response = self._get_sdk_client().chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                ],
                temperature=max(0.0, min(float(temperature), 1.0)),
                max_tokens=max_tokens,
                response_format=response_format,
            )
        except Exception as exc:
            if json_schema is not None and self._is_response_format_schema_error(exc):
                self._json_schema_supported = False
                self._debug("LLM provider rejected json_schema in chat.completions API; disabled for next requests")
            self._debug(f"LLM chat.completions error type={type(exc).__name__} detail={exc!r}")
            return None

        try:
            as_dict = response.model_dump()
        except Exception:
            as_dict = {"response_repr": repr(response)}

        choices = as_dict.get("choices")
        choices_count = len(choices) if isinstance(choices, list) else 0
        finish_reason = None
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            finish_reason = choices[0].get("finish_reason")
        self._debug(
            "LLM chat.completions response "
            f"choices={choices_count} finish_reason={finish_reason!r} "
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

            chat_text = self._extract_from_chat_choices_dict(as_dict)
            if chat_text:
                return chat_text

            status = as_dict.get("status")
            incomplete = as_dict.get("incomplete_details")
            self._debug(
                "LLM response has no extractable text "
                f"status={status!r} incomplete={incomplete!r} output_types={self._output_types(as_dict)!r}"
            )

        # Парсер с обратной совместимостью для словарей в стиле chat.completions.
        if isinstance(response_obj, dict):
            return self._extract_from_chat_choices_dict(response_obj)

        return None

    def _extract_json_from_response(self, response_obj: Any, *, prefer_structured: bool) -> dict[str, Any] | None:
        if prefer_structured:
            structured = self._extract_structured_json(response_obj)
            if structured is not None:
                return structured

        content = self._extract_message_content(response_obj)
        if isinstance(content, str) and content:
            parsed = self._extract_json_object(content)
            if parsed is not None:
                return parsed
            self._debug(f"LLM returned non-JSON text prefix: {content[:280]!r}")
        else:
            self._debug("LLM response has no assistant text content")

        if not prefer_structured:
            structured = self._extract_structured_json(response_obj)
            if structured is not None:
                return structured
        return None

    def _extract_structured_json(self, response_obj: Any) -> dict[str, Any] | None:
        for attr in ("output_parsed", "parsed"):
            payload = self._coerce_json_payload(getattr(response_obj, attr, None))
            if payload is not None:
                return payload

        as_dict: dict[str, Any] | None = None
        if isinstance(response_obj, dict):
            as_dict = response_obj
        else:
            try:
                dumped = response_obj.model_dump()
                if isinstance(dumped, dict):
                    as_dict = dumped
            except Exception:
                as_dict = None
        if not isinstance(as_dict, dict):
            return None

        for key in ("output_parsed", "parsed", "json", "data"):
            payload = self._coerce_json_payload(as_dict.get(key))
            if payload is not None:
                return payload

        choices = as_dict.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict):
                payload = self._coerce_json_payload(message.get("parsed"))
                if payload is not None:
                    return payload
                content = message.get("content")
                if isinstance(content, list):
                    for item in content:
                        if not isinstance(item, dict):
                            continue
                        for key in ("json", "parsed", "value", "data"):
                            payload = self._coerce_json_payload(item.get(key))
                            if payload is not None:
                                return payload

        output = as_dict.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                for key in ("json", "parsed", "value", "data"):
                    payload = self._coerce_json_payload(item.get(key))
                    if payload is not None:
                        return payload
                content = item.get("content")
                if not isinstance(content, list):
                    continue
                for content_item in content:
                    if not isinstance(content_item, dict):
                        continue
                    for key in ("json", "parsed", "value", "data"):
                        payload = self._coerce_json_payload(content_item.get(key))
                        if payload is not None:
                            return payload
        return None

    def _coerce_json_payload(self, payload: Any) -> dict[str, Any] | None:
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list):
            return {"decisions": payload}
        return None

    def _extract_from_chat_choices_dict(self, response_dict: dict[str, Any]) -> str | None:
        choices = response_dict.get("choices")
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
            trimmed = content.strip()
            return trimmed if trimmed else None

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
            if isinstance(parsed, list):
                return {"decisions": parsed}
        except JSONDecodeError:
            pass

        obj_start = normalized.find("{")
        arr_start = normalized.find("[")
        starts = [idx for idx in (obj_start, arr_start) if idx >= 0]
        if not starts:
            return None
        start = min(starts)
        decoder = JSONDecoder()
        try:
            parsed, _idx = decoder.raw_decode(normalized[start:])
        except JSONDecodeError as exc:
            self._debug(
                "JSON decode failed "
                f"msg={exc.msg!r} pos={exc.pos} len={len(normalized)} "
                f"prefix={normalized[:180]!r} suffix={normalized[-180:]!r}"
            )
            partial = self._extract_partial_decisions(normalized)
            if partial is not None:
                self._debug(
                    "Recovered partial JSON decisions "
                    f"count={len(partial.get('decisions', []))}"
                )
                return partial
            return None
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"decisions": parsed}
        return None

    def _extract_partial_decisions(self, content: str) -> dict[str, Any] | None:
        key_idx = content.find('"decisions"')
        if key_idx >= 0:
            arr_idx = content.find("[", key_idx)
            if arr_idx >= 0:
                items = self._parse_partial_object_array(content, arr_idx)
                if items:
                    return {"decisions": items}

        first_arr = content.find("[")
        if first_arr >= 0:
            items = self._parse_partial_object_array(content, first_arr)
            if items:
                return {"decisions": items}
        return None

    def _parse_partial_object_array(self, content: str, array_start: int) -> list[dict[str, Any]]:
        decoder = JSONDecoder()
        idx = array_start + 1
        length = len(content)
        items: list[dict[str, Any]] = []

        while idx < length:
            while idx < length and content[idx] in " \n\r\t,":
                idx += 1
            if idx >= length:
                break
            if content[idx] == "]":
                break
            if content[idx] != "{":
                idx += 1
                continue

            try:
                parsed, consumed = decoder.raw_decode(content[idx:])
            except JSONDecodeError:
                break
            if isinstance(parsed, dict):
                items.append(parsed)
            idx += consumed
        return items

    def _is_response_format_schema_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        if "response_format" not in msg and "json_schema" not in msg and "schema" not in msg:
            return False
        schema_problem = any(
            marker in msg
            for marker in (
                "invalid schema",
                "schema is invalid",
                "unknown parameter",
                "unsupported",
                "not supported",
                "invalid type for",
            )
        )
        missing_required = "json_schema" in msg and "required" in msg
        return schema_problem or missing_required

    def _provider_compatible_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        return self._normalize_schema_node(schema)

    def _normalize_schema_node(self, node: Any) -> Any:
        if isinstance(node, list):
            return [self._normalize_schema_node(item) for item in node]
        if not isinstance(node, dict):
            return node

        normalized: dict[str, Any] = {}
        for key, value in node.items():
            if key == "default":
                continue
            normalized[key] = self._normalize_schema_node(value)

        node_type = normalized.get("type")
        props = normalized.get("properties")
        if node_type == "object" and isinstance(props, dict):
            normalized["required"] = list(props.keys())
            normalized.setdefault("additionalProperties", False)
        return normalized
