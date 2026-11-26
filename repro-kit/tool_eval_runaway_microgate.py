#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded, fail-closed tool-call streaming microgate for QAD5500.

The gate sends one deterministic request twice: first as an SSE stream and
then as a blocking chat completion.  It preserves the raw responses and
requires byte-exact text parity, one expected tool call, exact usage, a
terminal ``[DONE]`` marker, and completion before a shared deadline.

This is deliberately a microgate, not an end-to-end tool benchmark.  It does
not execute tools and it does not launch or mutate a server.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import http.client
import json
import math
import os
import re
import secrets
import socket
import ssl
import subprocess
import sys
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA = "flux-tool-eval-runaway-manifest-v1"
FIXTURE_SCHEMA = "flux-tool-eval-runaway-fixture-v1"
RECEIPT_SCHEMA = "flux-tool-eval-runaway-receipt-v1"
INTERVENTION_LEDGER_SCHEMA = "atlas.interventions.v1"
INTERVENTION_LEDGER_POLICY = "report_only"
RENDERED_PROMPT_SCHEMA = "atlas.rendered_prompt.v1"
LIVE_PROVENANCE_SCHEMA = "flux-tool-eval-live-provenance-v1"
ATLAS_SYSTEM_FINGERPRINT = "fp_atlas"

ATLAS_RESPONSE_ID_PATTERN = re.compile(
    r"chatcmpl-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)
ATLAS_TOOL_CALL_ID_PATTERN = re.compile(r"call_[0-9a-f]{16}")

SEALED_TOOL_SCHEMAS = {
    "get_weather": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
        "additionalProperties": False,
    }
}
SEALED_TOOL_ARGUMENTS = {"get_weather": {"city": "Paris"}}
SEALED_USER_MESSAGE = {
    "role": "user",
    "content": "Use get_weather once to look up Paris.",
}

HARD_TIMEOUT_SECONDS = 45.0
HARD_TOKEN_CAP = 512
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_SSE_LINE_BYTES = 2 * 1024 * 1024
GIT_EXECUTABLE = "/usr/bin/git"

HERMES_MARKERS = (
    "You are a function calling AI model.",
    "return a JSON object with function name and arguments",
    '{"name": <function-name>, "arguments": <args-json-object>}',
)
NATIVE_QWEN_MARKER = "# Tools\n\nYou have access to the following functions:"
F33_BASH_RETRY_SUFFIX = (
    ' | After ONE failure with "command not found" or exit code 127, do NOT '
    "retry the same command — the binary is permanently unavailable in this "
    "environment. Choose a different approach or tell the user the dependency "
    "is missing."
)
NATIVE_QWEN_TOOL_SUFFIX = """</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>

<ATLAS_TOOL_SAFETY>
Tools are immediately executable. Once you decide to use a tool, emit the function call directly. Put commands, file content, and every other tool input only in the call arguments; do not pre-render them as prose or markdown fences.
For Bash/bash, do not stage the command in a bash fence or narrate that you are about to run it. For Write/Edit, do not print the file body before calling the tool.
Use separate calls for multiple tools. Never simulate a tool response, file listing, or other result that the tool has not returned.
You must call at least one offered function. Do not answer with ordinary text instead.
</ATLAS_TOOL_SAFETY>"""


class GateError(ValueError):
    """A fail-closed contract or wire-format violation."""


class RawPromptLeakError(GateError):
    """A response attempted to expose prompt material outside its digest."""


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def strict_json_loads(value: str, label: str) -> Any:
    """Decode JSON while rejecting duplicate keys and non-finite numbers."""

    def reject_constant(constant: str) -> None:
        raise GateError(f"{label} contains non-finite number {constant!r}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise GateError(f"{label} contains duplicate key {key!r}")
            result[key] = item
        return result

    def reject_nonfinite_tree(item: Any, path: str = "$") -> None:
        if type(item) is float and not math.isfinite(item):
            raise GateError(f"{label} contains non-finite number at {path}")
        if type(item) is dict:
            for key, child in item.items():
                reject_nonfinite_tree(child, f"{path}.{key}")
        elif type(item) is list:
            for index, child in enumerate(item):
                reject_nonfinite_tree(child, f"{path}[{index}]")

    try:
        decoded = json.loads(
            value,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
        reject_nonfinite_tree(decoded)
        return decoded
    except GateError:
        raise
    except json.JSONDecodeError as error:
        raise GateError(f"{label} is malformed JSON") from error
    except (OverflowError, RecursionError, ValueError):
        raise GateError(f"{label} exceeds JSON resource limits") from None


def strict_response_json_loads(value: str, label: str) -> Any:
    """Decode an untrusted response without exposing undecoded bytes."""

    try:
        return strict_json_loads(value, label)
    except GateError:
        raise RawPromptLeakError(
            f"{label} failed strict privacy-safe JSON decoding"
        ) from None


def _object(value: object, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise GateError(f"{label} must be an object")
    return value


def _array(value: object, label: str) -> list[Any]:
    if type(value) is not list:
        raise GateError(f"{label} must be an array")
    return value


def _string(value: object, label: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        qualifier = "non-empty " if nonempty else ""
        raise GateError(f"{label} must be a {qualifier}string")
    return value


def _exact_int(value: object, expected: int, label: str) -> int:
    if type(value) is not int or value != expected:
        raise GateError(f"{label} must equal {expected}, got {value!r}")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise GateError(f"{label} must be a positive integer")
    return value


def _exact_bool(value: object, expected: bool, label: str) -> bool:
    if type(value) is not bool or value is not expected:
        raise GateError(f"{label} must be {expected!r}, got {value!r}")
    return value


def _exact_number(value: object, expected: float, label: str) -> float:
    if type(value) not in (int, float) or type(value) is bool:
        raise GateError(f"{label} must be numeric")
    number = float(value)
    if not (number == expected):
        raise GateError(f"{label} must equal {expected}, got {value!r}")
    return number


def _empty_intervention_ledger(value: object, label: str) -> dict[str, Any]:
    ledger = _object(value, label)
    expected_keys = {"schema", "policy", "events"}
    if set(ledger) != expected_keys:
        raise GateError(
            f"{label} keys differ: expected {sorted(expected_keys)!r}, "
            f"got {sorted(ledger)!r}"
        )
    if ledger.get("schema") != INTERVENTION_LEDGER_SCHEMA:
        raise GateError(f"{label}.schema must equal {INTERVENTION_LEDGER_SCHEMA!r}")
    if ledger.get("policy") != INTERVENTION_LEDGER_POLICY:
        raise GateError(f"{label}.policy must equal {INTERVENTION_LEDGER_POLICY!r}")
    events = _array(ledger.get("events"), f"{label}.events")
    if events:
        raise GateError(f"{label}.events must be empty, got {events!r}")
    return ledger


def _nonnegative_number(value: object, label: str) -> float:
    if type(value) not in (int, float) or type(value) is bool:
        raise GateError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise GateError(f"{label} must be a finite non-negative number")
    return number


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise GateError(
            f"{label} keys differ: missing={sorted(expected - actual)!r} "
            f"extra={sorted(actual - expected)!r}"
        )


def _json_file(path: Path, label: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise GateError(f"{label} is not readable JSON") from error
    value = strict_json_loads(text, label)
    return _object(value, label)


def _text_evidence(value: str) -> dict[str, Any]:
    raw = value.encode("utf-8")
    return {
        "utf8": value,
        "utf8_hex": raw.hex(),
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
    }


def max_consecutive_blank_lines(value: str) -> int:
    """Return the longest run of complete blank lines in ``value``."""

    longest = 0
    current = 0
    for line in value.splitlines():
        if not line.strip():
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _captured_tool_matches_request(captured: object, requested: object) -> bool:
    if type(captured) is not dict or type(requested) is not dict:
        return False
    captured_copy = copy.deepcopy(captured)
    requested_copy = copy.deepcopy(requested)
    try:
        captured_function = captured_copy["function"]
        requested_function = requested_copy["function"]
        name = requested_function["name"]
    except (KeyError, TypeError):
        return False
    if name in {"Bash", "bash"}:
        captured_description = captured_function.get("description")
        requested_description = requested_function.get("description")
        if requested_description is None:
            if not (
                type(captured_description) is str
                and captured_description.startswith("[atlas-f33]")
            ):
                return False
        elif captured_description != requested_description:
            if not (
                type(captured_description) is str
                and captured_description.startswith(
                    requested_description + "\n[atlas-f33]"
                )
            ):
                return False
        if "description" in requested_function:
            captured_function["description"] = requested_description
        else:
            captured_function.pop("description", None)
    return captured_copy == requested_copy


def _expected_server_tool(requested: object) -> dict[str, Any]:
    tool = copy.deepcopy(_object(requested, "request tool"))
    function = _object(tool.get("function"), "request tool.function")
    name = function.get("name")
    if name in {"Bash", "bash"}:
        description = function.get("description")
        if description is None:
            function["description"] = "[atlas-f33]" + F33_BASH_RETRY_SUFFIX
        elif "[atlas-f33]" not in description:
            function["description"] = (
                description + "\n[atlas-f33]" + F33_BASH_RETRY_SUFFIX
            )
    return tool


def expected_rendered_prompt(request: dict[str, Any]) -> str:
    messages = _array(request.get("messages"), "request.messages")
    if len(messages) != 1:
        raise GateError("microgate request must contain exactly one user message")
    message = _object(messages[0], "request.messages[0]")
    _exact_keys(message, {"role", "content"}, "request.messages[0]")
    if message.get("role") != "user":
        raise GateError("microgate request message must have role=user")
    content = _string(message.get("content"), "request.messages[0].content")
    if content != content.strip():
        raise GateError("microgate user content must not rely on template trimming")
    tools = _array(request.get("tools"), "request.tools")
    tool_lines = "\n".join(
        json.dumps(
            _expected_server_tool(tool),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for tool in tools
    )
    return (
        "<|im_start|>system\n"
        + NATIVE_QWEN_MARKER
        + "\n\n<tools>\n"
        + tool_lines
        + "\n"
        + NATIVE_QWEN_TOOL_SUFFIX
        + "<|im_end|>\n<|im_start|>user\n"
        + content
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def validate_rendered_prompt(rendered: str, request: dict[str, Any]) -> dict[str, Any]:
    """Bind one captured final Qwen rendering to the exact microgate request."""

    if type(rendered) is not str or not rendered:
        raise GateError("server-rendered prompt must be a non-empty string")
    native_count = rendered.count(NATIVE_QWEN_MARKER)
    tools_open_count = rendered.count("<tools>")
    tools_close_count = rendered.count("</tools>")
    safety_open_count = rendered.count("<ATLAS_TOOL_SAFETY>")
    safety_close_count = rendered.count("</ATLAS_TOOL_SAFETY>")
    hermes_hits = [marker for marker in HERMES_MARKERS if marker in rendered]
    if native_count != 1:
        raise GateError(
            "rendered prompt must contain the native Qwen tool header exactly once, "
            f"got {native_count}"
        )
    if tools_open_count != 1 or tools_close_count != 1:
        raise GateError(
            "rendered prompt must contain exactly one <tools>...</tools> schema block"
        )
    if hermes_hits:
        raise GateError("rendered prompt contains a contradictory Hermes tool prompt")
    if safety_open_count != 1 or safety_close_count != 1:
        raise GateError(
            "rendered prompt must contain exactly one Atlas native-tool safety block"
        )
    if rendered.count("<|im_start|>assistant\n") != 1:
        raise GateError(
            "rendered prompt must contain exactly one assistant generation prefix"
        )

    tools_start = rendered.index("<tools>") + len("<tools>")
    tools_end = rendered.index("</tools>", tools_start)
    tool_lines = [line for line in rendered[tools_start:tools_end].splitlines() if line]
    requested_tools = _array(request.get("tools"), "request.tools")
    if len(tool_lines) != len(requested_tools):
        raise GateError(
            "rendered prompt tool count does not match the request: "
            f"{len(tool_lines)} != {len(requested_tools)}"
        )
    captured_tools = [
        strict_json_loads(line, f"rendered prompt tool[{index}]")
        for index, line in enumerate(tool_lines)
    ]
    for index, (captured, requested) in enumerate(
        zip(captured_tools, requested_tools, strict=True)
    ):
        if not _captured_tool_matches_request(captured, requested):
            raise GateError(
                f"rendered prompt tool[{index}] is not the requested tool schema"
            )

    messages = _array(request.get("messages"), "request.messages")
    if len(messages) != 1:
        raise GateError("microgate request must contain exactly one user message")
    message = _object(messages[0], "request.messages[0]")
    _exact_keys(message, {"role", "content"}, "request.messages[0]")
    if message.get("role") != "user":
        raise GateError("microgate request message must have role=user")
    content = _string(message.get("content"), "request.messages[0].content")
    if content != content.strip():
        raise GateError("microgate user content must not rely on template trimming")

    # The JSON tool lines are compared semantically above.  Everything around
    # them is byte-exact, which rejects an extra message, injected instruction,
    # duplicate protocol, trailing text, or an alternate generation prefix.
    expected_prefix = "<|im_start|>system\n" + NATIVE_QWEN_MARKER + "\n\n<tools>"
    expected_suffix = (
        NATIVE_QWEN_TOOL_SUFFIX
        + "<|im_end|>\n<|im_start|>user\n"
        + content
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    captured_prefix = rendered[:tools_start]
    captured_suffix = rendered[tools_end:]
    if captured_prefix != expected_prefix or captured_suffix != expected_suffix:
        raise GateError(
            "rendered prompt differs from the exact native Qwen microgate envelope"
        )
    if rendered != expected_rendered_prompt(request):
        raise GateError("rendered prompt differs from the exact server serialization")

    tool_choice = request.get("tool_choice")

    binding = {
        "model": request.get("model"),
        "messages": request.get("messages"),
        "tools": request.get("tools"),
        "tool_choice": tool_choice,
    }
    raw = rendered.encode("utf-8")
    return {
        "observed": True,
        "sha256": sha256_bytes(raw),
        "native_qwen_header_count": native_count,
        "tools_open_count": tools_open_count,
        "tools_close_count": tools_close_count,
        "safety_block_count": safety_open_count,
        "hermes_marker_count": 0,
        "request_binding_sha256": sha256_bytes(canonical_bytes(binding)),
    }


def _rendered_prompt_evidence(
    value: object, label: str, *, expected_nonce: str, response_id: str
) -> dict[str, Any]:
    evidence = _object(value, label)
    _exact_keys(
        evidence,
        {"schema", "nonce", "response_id", "prompt_sha256"},
        label,
    )
    if evidence.get("schema") != RENDERED_PROMPT_SCHEMA:
        raise GateError(f"{label}.schema must equal {RENDERED_PROMPT_SCHEMA!r}")
    if evidence.get("nonce") != expected_nonce:
        raise GateError(f"{label}.nonce is not bound to this live request")
    if evidence.get("response_id") != response_id:
        raise GateError(f"{label}.response_id is not bound to the response")
    prompt_sha256 = _string(evidence.get("prompt_sha256"), f"{label}.prompt_sha256")
    if len(prompt_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in prompt_sha256
    ):
        raise GateError(f"{label}.prompt_sha256 must be 64 lowercase hex characters")
    return copy.deepcopy(evidence)


def validate_launch_contract(value: object, request: dict[str, Any]) -> dict[str, Any]:
    contract = _object(value, "launch_contract")
    expected = {
        "profile",
        "concurrency",
        "max_batch_size",
        "max_num_seqs",
        "prefill_chunk_tokens",
        "kv_cache_dtype",
        "speculation",
        "tool_call_parser",
        "chat_template",
    }
    _exact_keys(contract, expected, "launch_contract")

    if contract["profile"] != "interactive-w4096":
        raise GateError("launch_contract.profile must be interactive-w4096")
    _exact_int(contract["concurrency"], 1, "launch_contract.concurrency")
    _exact_int(contract["max_batch_size"], 1, "launch_contract.max_batch_size")
    _exact_int(contract["max_num_seqs"], 1, "launch_contract.max_num_seqs")
    _exact_int(
        contract["prefill_chunk_tokens"],
        4096,
        "launch_contract.prefill_chunk_tokens",
    )
    if contract["kv_cache_dtype"] != "bf16":
        raise GateError("launch_contract.kv_cache_dtype must be bf16")

    spec = _object(contract["speculation"], "launch_contract.speculation")
    _exact_keys(spec, {"mode", "num_drafts"}, "launch_contract.speculation")
    if spec["mode"] != "none":
        raise GateError("launch_contract.speculation.mode must be none")
    _exact_int(spec["num_drafts"], 0, "launch_contract.speculation.num_drafts")

    if contract["tool_call_parser"] != "qwen3_coder":
        raise GateError("launch_contract.tool_call_parser must be qwen3_coder")
    template = _object(contract["chat_template"], "launch_contract.chat_template")
    _exact_keys(
        template,
        {
            "source",
            "tool_format",
            "native_tool_render_count",
            "hermes_prompt_injection_count",
        },
        "launch_contract.chat_template",
    )
    if template["source"] != "checkpoint-native":
        raise GateError(
            "launch_contract.chat_template.source must be checkpoint-native"
        )
    if template["tool_format"] != "qwen-xml":
        raise GateError("launch_contract.chat_template.tool_format must be qwen-xml")
    _exact_int(
        template["native_tool_render_count"],
        1,
        "launch_contract.chat_template.native_tool_render_count",
    )
    _exact_int(
        template["hermes_prompt_injection_count"],
        0,
        "launch_contract.chat_template.hermes_prompt_injection_count",
    )

    expected_prompt = expected_rendered_prompt(request)
    audit = validate_rendered_prompt(expected_prompt, request)
    return {
        **audit,
        "observed": False,
        "source": "server-terminal-response-digest",
        "schema": RENDERED_PROMPT_SCHEMA,
    }


def _tool_names(request: dict[str, Any]) -> list[str]:
    tools = _array(request.get("tools"), "request.tools")
    if not tools:
        raise GateError("request.tools must not be empty")
    names: list[str] = []
    for index, raw in enumerate(tools):
        tool = _object(raw, f"request.tools[{index}]")
        if tool.get("type") != "function":
            raise GateError(f"request.tools[{index}].type must be function")
        function = _object(tool.get("function"), f"request.tools[{index}].function")
        name = _string(function.get("name"), f"request.tools[{index}].function.name")
        _object(
            function.get("parameters"),
            f"request.tools[{index}].function.parameters",
        )
        if name in names:
            raise GateError(f"request contains duplicate tool name {name!r}")
        names.append(name)
    return names


def validate_manifest(value: object) -> dict[str, Any]:
    manifest = _object(value, "manifest")
    _exact_keys(
        manifest,
        {
            "schema",
            "launch_contract",
            "request",
            "expected_tool_names",
            "token_cap",
            "timeout_seconds",
        },
        "manifest",
    )
    if manifest["schema"] != MANIFEST_SCHEMA:
        raise GateError(f"manifest.schema must be {MANIFEST_SCHEMA}")
    token_cap = _positive_int(manifest["token_cap"], "manifest.token_cap")
    if token_cap > HARD_TOKEN_CAP:
        raise GateError(
            f"manifest.token_cap exceeds hard microgate cap {HARD_TOKEN_CAP}"
        )
    timeout = manifest["timeout_seconds"]
    if type(timeout) not in (int, float) or type(timeout) is bool:
        raise GateError("manifest.timeout_seconds must be numeric")
    timeout_seconds = float(timeout)
    if not 0.0 < timeout_seconds <= HARD_TIMEOUT_SECONDS:
        raise GateError(
            f"manifest.timeout_seconds must be in (0, {HARD_TIMEOUT_SECONDS}]"
        )

    request = _object(manifest["request"], "manifest.request")
    runner_fields = {
        "stream",
        "stream_options",
        "rendered_prompt_evidence_nonce",
    }
    if runner_fields.intersection(request):
        raise GateError(
            "runner owns request.stream, request.stream_options, and "
            "request.rendered_prompt_evidence_nonce"
        )
    _string(request.get("model"), "request.model")
    messages = _array(request.get("messages"), "request.messages")
    if not messages:
        raise GateError("request.messages must not be empty")
    for index, raw in enumerate(messages):
        message = _object(raw, f"request.messages[{index}]")
        if message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise GateError(f"request.messages[{index}].role is invalid")
        if "content" not in message:
            raise GateError(f"request.messages[{index}] is missing content")
        _string(message["content"], f"request.messages[{index}].content")
    if messages != [SEALED_USER_MESSAGE]:
        raise GateError("request.messages must equal the sealed Paris test prompt")

    available_tools = _tool_names(request)
    if available_tools != list(SEALED_TOOL_SCHEMAS):
        raise GateError(
            "request.tools must contain exactly the sealed get_weather tool"
        )
    for index, raw_tool in enumerate(request["tools"]):
        function = _object(
            _object(raw_tool, f"request.tools[{index}]").get("function"),
            f"request.tools[{index}].function",
        )
        name = _string(function.get("name"), f"request.tools[{index}].function.name")
        if function.get("parameters") != SEALED_TOOL_SCHEMAS[name]:
            raise GateError(
                f"request.tools[{index}].function.parameters differs from the "
                "sealed city-required schema"
            )
    expected_tools = _array(
        manifest["expected_tool_names"], "manifest.expected_tool_names"
    )
    if not expected_tools:
        raise GateError("manifest.expected_tool_names must not be empty")
    if any(type(name) is not str or not name for name in expected_tools):
        raise GateError(
            "manifest.expected_tool_names entries must be non-empty strings"
        )
    if len(set(expected_tools)) != len(expected_tools):
        raise GateError("manifest.expected_tool_names contains duplicates")
    if expected_tools != list(SEALED_TOOL_SCHEMAS):
        raise GateError(
            "manifest.expected_tool_names must equal the sealed tool-name list"
        )
    unexpected = set(expected_tools) - set(available_tools)
    if unexpected:
        raise GateError(
            f"expected tools are absent from request.tools: {sorted(unexpected)!r}"
        )

    _exact_int(request.get("max_tokens"), token_cap, "request.max_tokens")
    _exact_number(request.get("temperature"), 0.0, "request.temperature")
    _exact_int(request.get("top_k"), 1, "request.top_k")
    _exact_number(request.get("top_p"), 1.0, "request.top_p")
    _exact_bool(
        request.get("parallel_tool_calls"), False, "request.parallel_tool_calls"
    )
    _exact_bool(request.get("return_token_ids"), True, "request.return_token_ids")
    _exact_bool(
        request.get("report_interventions"), True, "request.report_interventions"
    )
    chat_template_kwargs = _object(
        request.get("chat_template_kwargs"), "request.chat_template_kwargs"
    )
    _exact_keys(
        chat_template_kwargs,
        {"enable_thinking"},
        "request.chat_template_kwargs",
    )
    _exact_bool(
        chat_template_kwargs.get("enable_thinking"),
        False,
        "request.chat_template_kwargs.enable_thinking",
    )
    _exact_int(request.get("n"), 1, "request.n")
    seed = request.get("seed")
    if type(seed) is not int or seed < 0:
        raise GateError("request.seed must be a non-negative integer")

    if request.get("tool_choice") not in ("auto", "required"):
        raise GateError("request.tool_choice must be auto or required")

    prompt_audit = validate_launch_contract(manifest["launch_contract"], request)

    validated = copy.deepcopy(manifest)
    validated["timeout_seconds"] = timeout_seconds
    validated["prompt_audit"] = prompt_audit
    return validated


@dataclass(frozen=True)
class RequestPair:
    stream: dict[str, Any]
    blocking: dict[str, Any]
    stream_nonce: str
    blocking_nonce: str


def request_bodies(
    manifest: dict[str, Any],
    *,
    stream_nonce: str | None = None,
    blocking_nonce: str | None = None,
) -> RequestPair:
    stream_nonce = stream_nonce or secrets.token_hex(16)
    blocking_nonce = blocking_nonce or secrets.token_hex(16)
    for nonce, label in (
        (stream_nonce, "stream nonce"),
        (blocking_nonce, "blocking nonce"),
    ):
        if (
            type(nonce) is not str
            or len(nonce) != 32
            or any(char not in "0123456789abcdef" for char in nonce)
        ):
            raise GateError(f"{label} must be exactly 32 lowercase hex characters")
    if stream_nonce == blocking_nonce:
        raise GateError("stream and blocking prompt-evidence nonces must differ")
    base = copy.deepcopy(manifest["request"])
    stream = copy.deepcopy(base)
    stream["stream"] = True
    stream["rendered_prompt_evidence_nonce"] = stream_nonce
    stream["stream_options"] = {
        "include_usage": True,
        "continuous_usage_stats": False,
        "include_obfuscation": False,
    }
    blocking = copy.deepcopy(base)
    blocking["stream"] = False
    blocking["rendered_prompt_evidence_nonce"] = blocking_nonce
    return RequestPair(stream, blocking, stream_nonce, blocking_nonce)


@dataclass(frozen=True)
class HttpCapture:
    status: int
    content_type: str
    raw: bytes
    elapsed_ms: float
    first_body_byte_ms: float
    line_timings_ms: tuple[float, ...] = ()


def _normalize_content_type(value: object, expected: str, label: str) -> str:
    if type(value) is not str or not value.isascii() or value.lower() != expected:
        raise RawPromptLeakError(
            f"{label} Content-Type is not the exact approved media type"
        )
    return expected


def _immutable_hex(value: object, label: str, lengths: tuple[int, ...]) -> str:
    if (
        type(value) is not str
        or len(value) not in lengths
        or any(char not in "0123456789abcdef" for char in value)
    ):
        joined = " or ".join(str(length) for length in lengths)
        raise GateError(f"{label} must be immutable lowercase hex ({joined} chars)")
    return value


def _git_output(arguments: list[str]) -> str:
    clean_env = {
        "PATH": os.defpath,
        "LC_ALL": "C",
        "LANG": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    try:
        completed = subprocess.run(
            [GIT_EXECUTABLE, "--no-replace-objects", *arguments],
            cwd=Path(__file__).resolve().parent,
            env=clean_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        raise GateError(
            "live provenance cannot resolve the local Git identity"
        ) from None
    value = completed.stdout.strip()
    if not value:
        raise GateError("live provenance resolved an empty local Git identity")
    return value


def _local_runner_identity() -> dict[str, str]:
    root = Path(_git_output(["rev-parse", "--show-toplevel"]))
    runner = Path(__file__).resolve()
    try:
        relative_runner = runner.relative_to(root).as_posix()
    except ValueError:
        raise GateError("runner is outside the bound source Git tree") from None
    git_tree = _immutable_hex(
        _git_output(["rev-parse", "HEAD^{tree}"]), "local source Git tree", (40, 64)
    )
    tracked_blob = _immutable_hex(
        _git_output(["rev-parse", f"HEAD:{relative_runner}"]),
        "tracked runner Git blob",
        (40, 64),
    )
    working_blob = _immutable_hex(
        _git_output(["hash-object", "--no-filters", str(runner)]),
        "working runner Git blob",
        (40, 64),
    )
    if working_blob != tracked_blob:
        raise GateError("live runner differs from its committed Git blob")
    try:
        runner_sha256 = sha256_bytes(runner.read_bytes())
    except OSError:
        raise GateError("live runner bytes are not readable") from None
    return {
        "git_tree": git_tree,
        "git_blob": tracked_blob,
        "sha256": runner_sha256,
    }


def _validate_live_provenance(value: object) -> dict[str, Any]:
    provenance = _object(value, "live_provenance")
    _exact_keys(
        provenance,
        {"schema", "runner", "source", "execution", "checkpoint"},
        "live_provenance",
    )
    if provenance["schema"] != LIVE_PROVENANCE_SCHEMA:
        raise GateError(f"live_provenance.schema must be {LIVE_PROVENANCE_SCHEMA}")

    runner = _object(provenance["runner"], "live_provenance.runner")
    _exact_keys(
        runner,
        {"git_tree", "git_blob", "sha256"},
        "live_provenance.runner",
    )
    normalized_runner = {
        "git_tree": _immutable_hex(
            runner["git_tree"], "live_provenance.runner.git_tree", (40, 64)
        ),
        "git_blob": _immutable_hex(
            runner["git_blob"], "live_provenance.runner.git_blob", (40, 64)
        ),
        "sha256": _immutable_hex(
            runner["sha256"], "live_provenance.runner.sha256", (64,)
        ),
    }
    local_runner = _local_runner_identity()
    if normalized_runner != local_runner:
        raise GateError("live provenance does not match the committed runner identity")

    source = _object(provenance["source"], "live_provenance.source")
    _exact_keys(source, {"git_tree"}, "live_provenance.source")
    source_tree = _immutable_hex(
        source["git_tree"], "live_provenance.source.git_tree", (40, 64)
    )
    if source_tree != local_runner["git_tree"]:
        raise GateError("live source Git tree differs from the runner source tree")

    execution = _object(provenance["execution"], "live_provenance.execution")
    _exact_keys(
        execution,
        {"mode", "binary_sha256", "image_digest"},
        "live_provenance.execution",
    )
    mode = execution["mode"]
    if mode not in {"host-binary", "container-image"}:
        raise GateError("live_provenance.execution.mode is invalid")
    binary_sha256 = _immutable_hex(
        execution["binary_sha256"],
        "live_provenance.execution.binary_sha256",
        (64,),
    )
    image_digest = execution["image_digest"]
    if mode == "host-binary":
        if image_digest is not None:
            raise GateError("host-binary provenance must not claim an image digest")
    elif (
        type(image_digest) is not str
        or not image_digest.startswith("sha256:")
        or _immutable_hex(
            image_digest.removeprefix("sha256:"),
            "live_provenance.execution.image_digest",
            (64,),
        )
        != image_digest.removeprefix("sha256:")
    ):
        raise GateError("container-image provenance requires an immutable digest")

    checkpoint = _object(provenance["checkpoint"], "live_provenance.checkpoint")
    _exact_keys(
        checkpoint,
        {"id", "revision", "served_model"},
        "live_provenance.checkpoint",
    )
    checkpoint_id = checkpoint["id"]
    if (
        type(checkpoint_id) is not str
        or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}(?:/[A-Za-z0-9][A-Za-z0-9._-]{0,127})?",
            checkpoint_id,
        )
        is None
    ):
        raise GateError("live_provenance.checkpoint.id is not a safe public identifier")
    checkpoint_revision = _immutable_hex(
        checkpoint["revision"], "live_provenance.checkpoint.revision", (40, 64)
    )
    served_model = checkpoint["served_model"]
    if (
        type(served_model) is not str
        or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}(?:/[A-Za-z0-9][A-Za-z0-9._-]{0,127})?",
            served_model,
        )
        is None
    ):
        raise GateError(
            "live_provenance.checkpoint.served_model is not a safe public identifier"
        )

    normalized = {
        "schema": LIVE_PROVENANCE_SCHEMA,
        "runner": normalized_runner,
        "source": {"git_tree": source_tree},
        "execution": {
            "mode": mode,
            "binary_sha256": binary_sha256,
            "image_digest": image_digest,
        },
        "checkpoint": {
            "id": checkpoint_id,
            "revision": checkpoint_revision,
            "served_model": served_model,
        },
    }
    return normalized


def _endpoint_parts(endpoint: str) -> tuple[str, str, int, str, str]:
    try:
        parsed = urllib.parse.urlsplit(endpoint)
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        raise GateError("endpoint is not a valid URL") from None
    if parsed.scheme not in {"http", "https"}:
        raise GateError("endpoint scheme must be http or https")
    if not parsed.hostname or parsed.username or parsed.password:
        raise GateError("endpoint must have a host and must not contain credentials")
    if parsed.query or parsed.fragment:
        raise GateError("endpoint must not contain a query or fragment")
    path = parsed.path or "/v1/chat/completions"
    if path.endswith("/"):
        path += "v1/chat/completions"
    if not path.endswith("/v1/chat/completions"):
        raise GateError("endpoint path must end in /v1/chat/completions")
    hostname = parsed.hostname.lower()
    display_host = f"[{hostname}]" if ":" in hostname else hostname
    normalized = f"{parsed.scheme}://{display_host}:{port}{path}"
    return parsed.scheme, hostname, port, path, normalized


def _capture_binding(
    mode: str,
    *,
    endpoint: str | None = None,
    live_provenance: object | None = None,
) -> dict[str, Any]:
    if mode == "offline-fixture":
        if endpoint is not None or live_provenance is not None:
            raise GateError("offline fixture capture cannot claim live provenance")
        return {
            "mode": "offline-fixture",
            "authoritative": False,
            "authority_scope": "offline-parser-only",
            "endpoint": None,
            "identity": None,
            "identity_sha256": None,
            "declared_identity": None,
            "declared_identity_sha256": None,
            "runtime_binding": {
                "listener_executable_observed": False,
                "container_image_observed": False,
                "checkpoint_inventory_observed": False,
            },
        }
    if mode != "live-http" or endpoint is None or live_provenance is None:
        raise GateError("live capture requires endpoint and a provenance declaration")
    normalized_provenance = _validate_live_provenance(live_provenance)
    normalized_endpoint = _endpoint_parts(endpoint)[4]
    return {
        "mode": "live-http",
        # The document proves the committed runner, but its execution and
        # checkpoint fields are operator declarations.  This client cannot
        # observe which executable owns the listener, which container (if any)
        # backs it, or which checkpoint files that process opened.  Never turn
        # those declarations into release authority.
        "authoritative": False,
        "authority_scope": "live-parser-only-unverified-runtime",
        "endpoint": normalized_endpoint,
        "identity": None,
        "identity_sha256": None,
        "declared_identity": normalized_provenance,
        "declared_identity_sha256": sha256_bytes(
            canonical_bytes(normalized_provenance)
        ),
        "runtime_binding": {
            "listener_executable_observed": False,
            "container_image_observed": False,
            "checkpoint_inventory_observed": False,
        },
    }


def _bind_capture_to_manifest(
    capture_binding: dict[str, Any], manifest: dict[str, Any]
) -> None:
    if capture_binding["mode"] == "offline-fixture":
        return
    served_model = capture_binding["declared_identity"]["checkpoint"]["served_model"]
    if served_model != manifest["request"]["model"]:
        raise GateError(
            "live checkpoint served_model differs from the sealed request model"
        )


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise GateError("shared request deadline exceeded")
    return remaining


def _connection_for(
    endpoint: str, timeout: float
) -> tuple[http.client.HTTPConnection, str]:
    scheme, hostname, port, path, _ = _endpoint_parts(endpoint)
    if scheme == "https":
        connection: http.client.HTTPConnection = http.client.HTTPSConnection(
            hostname,
            port,
            timeout=timeout,
            context=ssl.create_default_context(),
        )
    else:
        connection = http.client.HTTPConnection(hostname, port, timeout=timeout)
    return connection, path


def _set_socket_timeout(connection: http.client.HTTPConnection, timeout: float) -> None:
    if connection.sock is not None:
        connection.sock.settimeout(max(0.001, timeout))


def http_post(
    endpoint: str,
    body: dict[str, Any],
    api_key: str | None,
    deadline: float,
    *,
    stream: bool,
) -> HttpCapture:
    encoded = canonical_bytes(body)
    started = time.monotonic()
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
        "Connection": "close",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    connection: http.client.HTTPConnection | None = None
    try:
        connection, path = _connection_for(endpoint, _remaining(deadline))
        connection.request("POST", path, body=encoded, headers=headers)
        _set_socket_timeout(connection, _remaining(deadline))
        response = connection.getresponse()
        content_type = response.getheader("Content-Type", "")
        first_body_byte_ms = -1.0
        line_timings: list[float] = []
        chunks: list[bytes] = []
        total = 0
        if stream:
            while True:
                _set_socket_timeout(connection, _remaining(deadline))
                line = response.readline(MAX_SSE_LINE_BYTES + 1)
                now = time.monotonic()
                if not line:
                    break
                if len(line) > MAX_SSE_LINE_BYTES:
                    raise GateError("SSE line exceeds safety limit")
                if first_body_byte_ms < 0:
                    first_body_byte_ms = (now - started) * 1000.0
                total += len(line)
                if total > MAX_RESPONSE_BYTES:
                    raise GateError("SSE response exceeds safety limit")
                chunks.append(line)
                line_timings.append((now - started) * 1000.0)
        else:
            while True:
                _set_socket_timeout(connection, _remaining(deadline))
                chunk = response.read(min(64 * 1024, MAX_RESPONSE_BYTES - total + 1))
                now = time.monotonic()
                if not chunk:
                    break
                if first_body_byte_ms < 0:
                    first_body_byte_ms = (now - started) * 1000.0
                total += len(chunk)
                if total > MAX_RESPONSE_BYTES:
                    raise GateError("blocking response exceeds safety limit")
                chunks.append(chunk)
        elapsed_ms = (time.monotonic() - started) * 1000.0
        if time.monotonic() > deadline:
            raise GateError("shared request deadline exceeded")
        return HttpCapture(
            status=response.status,
            content_type=content_type,
            raw=b"".join(chunks),
            elapsed_ms=elapsed_ms,
            first_body_byte_ms=first_body_byte_ms,
            line_timings_ms=tuple(line_timings),
        )
    except (OSError, http.client.HTTPException, socket.timeout, ValueError):
        # A malformed peer-controlled status line can be embedded in an
        # HTTPException string.  Keep the process-boundary diagnostic constant.
        raise GateError("HTTP request failed") from None
    finally:
        if connection is not None:
            connection.close()


def _decode_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise RawPromptLeakError(
            f"{label} failed strict privacy-safe UTF-8 decoding"
        ) from error
    value = strict_response_json_loads(text, label)
    return _privacy_object(value, label)


def _token_id_occurrences(value: object, path: str = "$") -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if type(value) is dict:
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in {"token_id", "token_ids"}:
                if key == "token_id":
                    valid = type(child) is int and child >= 0
                else:
                    valid = type(child) is list and all(
                        type(item) is int and item >= 0 for item in child
                    )
                if not valid:
                    raise GateError(f"{child_path} has invalid token IDs")
                found.append({"path": child_path, "value": copy.deepcopy(child)})
            found.extend(_token_id_occurrences(child, child_path))
    elif type(value) is list:
        for index, child in enumerate(value):
            found.extend(_token_id_occurrences(child, f"{path}[{index}]"))
    return found


def _field_occurrences(
    value: object, field: str, path: str = "$"
) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if type(value) is dict:
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key == field:
                found.append({"path": child_path, "value": child})
            found.extend(_field_occurrences(child, field, child_path))
    elif type(value) is list:
        for index, child in enumerate(value):
            found.extend(_field_occurrences(child, field, f"{path}[{index}]"))
    return found


def _privacy_object(value: object, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise RawPromptLeakError(
            f"{label} violates the pre-artifact response privacy shape"
        )
    return value


def _privacy_array(value: object, label: str) -> list[Any]:
    if type(value) is not list:
        raise RawPromptLeakError(
            f"{label} violates the pre-artifact response privacy shape"
        )
    return value


def _privacy_keys(value: dict[str, Any], allowed: set[str], label: str) -> None:
    extras = set(value) - allowed
    if extras:
        raise RawPromptLeakError(
            f"{label} keys differ: unapproved pre-artifact response fields "
            f"{sorted(extras)!r}"
        )


def _privacy_leaf(
    value: object,
    allowed_types: tuple[type, ...],
    label: str,
    expected: str,
) -> None:
    """Type one response-schema leaf before any raw capture can be persisted."""

    if type(value) not in allowed_types:
        raise RawPromptLeakError(
            f"{label} {expected}; violates the pre-artifact response privacy shape"
        )


def _privacy_leaf_fields(
    value: dict[str, Any],
    fields: dict[str, tuple[tuple[type, ...], str]],
    label: str,
) -> None:
    """Apply exact built-in types to every present scalar schema leaf."""

    for key, (allowed_types, expected) in fields.items():
        if key in value:
            _privacy_leaf(value[key], allowed_types, f"{label}.{key}", expected)


def _privacy_exact_if_present(
    value: dict[str, Any], key: str, expected: object, label: str
) -> None:
    if key in value and value[key] != expected:
        raise RawPromptLeakError(f"{label}.{key} is not the exact safe metadata value")


def _privacy_none_if_present(value: dict[str, Any], key: str, label: str) -> None:
    _privacy_exact_if_present(value, key, None, label)


def _privacy_safe_identifier(value: object, label: str) -> None:
    if (
        type(value) is not str
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", value) is None
    ):
        raise RawPromptLeakError(f"{label} is not a bounded response identifier")


def _privacy_atlas_response_id(value: object, label: str) -> None:
    """Accept only IDs minted by Atlas's chat-completion UUID generator."""

    if type(value) is not str or ATLAS_RESPONSE_ID_PATTERN.fullmatch(value) is None:
        raise RawPromptLeakError(f"{label} is not an Atlas response identifier")


def _privacy_atlas_tool_call_id(value: object, label: str) -> None:
    """Accept only IDs minted by Atlas's qwen3_coder call counter."""

    if type(value) is not str or ATLAS_TOOL_CALL_ID_PATTERN.fullmatch(value) is None:
        raise RawPromptLeakError(f"{label} is not an Atlas tool-call identifier")


def _privacy_atlas_fingerprint(value: dict[str, Any], label: str) -> None:
    if value.get("system_fingerprint") != ATLAS_SYSTEM_FINGERPRINT:
        raise RawPromptLeakError(
            f"{label}.system_fingerprint is not the exact Atlas fingerprint"
        )


def _privacy_safe_model(value: object, label: str) -> None:
    if (
        type(value) is not str
        or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}(?:/[A-Za-z0-9][A-Za-z0-9._-]{0,127})?",
            value,
        )
        is None
    ):
        raise RawPromptLeakError(f"{label} is not a bounded public model identifier")


def _privacy_created_if_present(value: dict[str, Any], label: str) -> None:
    if "created" in value and not (
        type(value["created"]) is int and 0 <= value["created"] < 2**63
    ):
        raise RawPromptLeakError(
            f"{label}.created is not a bounded non-negative integer"
        )


def _privacy_token_ids_shape(value: object, label: str) -> None:
    token_ids = _privacy_array(value, label)
    for index, token_id in enumerate(token_ids):
        if type(token_id) is not int:
            raise RawPromptLeakError(
                f"{label}[{index}] has invalid token IDs; violates the "
                "pre-artifact response privacy shape"
            )


def _privacy_usage_shape(value: object, label: str) -> None:
    usage = _privacy_object(value, label)
    _privacy_keys(
        usage,
        {
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prompt_tokens_details",
            "completion_tokens_details",
            "time_to_first_token_ms",
            "response_token/s",
        },
        label,
    )
    _privacy_leaf_fields(
        usage,
        {
            "prompt_tokens": ((int,), "must be a non-negative integer"),
            "completion_tokens": ((int,), "must be a non-negative integer"),
            "total_tokens": ((int,), "must be a non-negative integer"),
            "time_to_first_token_ms": ((int, float), "must be numeric"),
            "response_token/s": ((int, float), "must be numeric"),
        },
        label,
    )
    if "prompt_tokens_details" in usage:
        details = _privacy_object(
            usage["prompt_tokens_details"], f"{label}.prompt_tokens_details"
        )
        _privacy_keys(
            details,
            {"cached_tokens", "audio_tokens"},
            f"{label}.prompt_tokens_details",
        )
        _privacy_leaf_fields(
            details,
            {
                "cached_tokens": ((int,), "must be a non-negative integer"),
                "audio_tokens": ((int,), "must be a non-negative integer"),
            },
            f"{label}.prompt_tokens_details",
        )
    if "completion_tokens_details" in usage:
        details = _privacy_object(
            usage["completion_tokens_details"],
            f"{label}.completion_tokens_details",
        )
        _privacy_keys(
            details,
            {
                "reasoning_tokens",
                "audio_tokens",
                "accepted_prediction_tokens",
                "rejected_prediction_tokens",
            },
            f"{label}.completion_tokens_details",
        )
        _privacy_leaf_fields(
            details,
            {
                "reasoning_tokens": ((int,), "must be a non-negative integer"),
                "audio_tokens": ((int,), "must be a non-negative integer"),
                "accepted_prediction_tokens": (
                    (int,),
                    "must be a non-negative integer",
                ),
                "rejected_prediction_tokens": (
                    (int,),
                    "must be a non-negative integer",
                ),
            },
            f"{label}.completion_tokens_details",
        )


def _privacy_intervention_shape(value: object, label: str) -> None:
    ledger = _privacy_object(value, label)
    _privacy_keys(ledger, {"schema", "policy", "events"}, label)
    _privacy_leaf_fields(
        ledger,
        {
            "schema": ((str,), "must be a string"),
            "policy": ((str,), "must be a string"),
        },
        label,
    )
    if "events" not in ledger:
        return
    for index, raw_event in enumerate(
        _privacy_array(ledger["events"], f"{label}.events")
    ):
        event = _privacy_object(raw_event, f"{label}.events[{index}]")
        _privacy_keys(
            event,
            {"code", "action", "count"},
            f"{label}.events[{index}]",
        )
        _privacy_leaf_fields(
            event,
            {
                "code": ((str,), "must be a string"),
                "action": ((str,), "must be a string"),
                "count": ((int,), "must be an integer"),
            },
            f"{label}.events[{index}]",
        )


def _privacy_prompt_evidence_shape(value: object, label: str) -> None:
    evidence = _privacy_object(value, label)
    _privacy_keys(
        evidence,
        {"schema", "nonce", "response_id", "prompt_sha256"},
        label,
    )
    _privacy_leaf_fields(
        evidence,
        {
            "schema": ((str,), "must be a string"),
            "nonce": ((str,), "must be a string"),
            "response_id": ((str,), "must be a string"),
            "prompt_sha256": ((str,), "must be a string"),
        },
        label,
    )


def _privacy_tool_function_shape(value: object, label: str) -> None:
    function = _privacy_object(value, label)
    _privacy_keys(function, {"name", "arguments"}, label)
    _privacy_leaf_fields(
        function,
        {
            "name": ((str,), "must be a string"),
            "arguments": ((str,), "must be a string"),
        },
        label,
    )


def _privacy_tool_shape(value: object, label: str, *, streaming: bool) -> None:
    tool = _privacy_object(value, label)
    allowed = (
        {"index", "id", "type", "function"}
        if streaming
        else {
            "id",
            "type",
            "function",
        }
    )
    _privacy_keys(tool, allowed, label)
    tool_fields = {
        "id": ((str,), "must be a string"),
        "type": ((str,), "must be a string"),
    }
    if streaming:
        tool_fields["index"] = ((int,), "must be an integer")
    _privacy_leaf_fields(tool, tool_fields, label)
    if "id" in tool:
        _privacy_atlas_tool_call_id(tool["id"], f"{label}.id")
    _privacy_exact_if_present(tool, "type", "function", label)
    if "function" in tool:
        _privacy_tool_function_shape(tool["function"], f"{label}.function")
        function = _privacy_object(tool["function"], f"{label}.function")
        if "name" in function:
            _privacy_safe_identifier(function["name"], f"{label}.function.name")


def _privacy_stream_event_shape(value: object, label: str) -> None:
    event = _privacy_object(value, label)
    _privacy_keys(
        event,
        {
            "id",
            "object",
            "created",
            "model",
            "system_fingerprint",
            "choices",
            "usage",
            "interventions",
            "rendered_prompt_evidence",
        },
        label,
    )
    if "id" not in event:
        raise RawPromptLeakError(f"{label}.id is required for an Atlas stream event")
    _privacy_leaf_fields(
        event,
        {
            "id": ((str,), "must be a string"),
            "object": ((str,), "must be a string"),
            "created": ((int,), "must be an integer"),
            "model": ((str,), "must be a string"),
            "system_fingerprint": ((str, type(None)), "must be a string or null"),
        },
        label,
    )
    if "id" in event:
        _privacy_atlas_response_id(event["id"], f"{label}.id")
    if "model" in event:
        _privacy_safe_model(event["model"], f"{label}.model")
    _privacy_exact_if_present(event, "object", "chat.completion.chunk", label)
    _privacy_atlas_fingerprint(event, label)
    _privacy_created_if_present(event, label)
    if "choices" in event:
        for choice_index, raw_choice in enumerate(
            _privacy_array(event["choices"], f"{label}.choices")
        ):
            choice_label = f"{label}.choices[{choice_index}]"
            choice = _privacy_object(raw_choice, choice_label)
            _privacy_keys(
                choice,
                {"index", "delta", "finish_reason", "logprobs", "token_ids"},
                choice_label,
            )
            _privacy_leaf_fields(
                choice,
                {
                    "index": ((int,), "must be an integer"),
                    "finish_reason": (
                        (str, type(None)),
                        "must be a string or null",
                    ),
                },
                choice_label,
            )
            if "finish_reason" in choice and choice["finish_reason"] not in {
                None,
                "length",
                "stop",
                "tool_calls",
            }:
                raise RawPromptLeakError(
                    f"{choice_label}.finish_reason is not an approved value"
                )
            if choice.get("logprobs") is not None:
                raise RawPromptLeakError(
                    f"{choice_label}.logprobs is an unapproved nested payload"
                )
            if "token_ids" in choice:
                _privacy_token_ids_shape(
                    choice["token_ids"], f"{choice_label}.token_ids"
                )
            if "delta" in choice:
                delta_label = f"{choice_label}.delta"
                delta = _privacy_object(choice["delta"], delta_label)
                _privacy_keys(
                    delta,
                    {
                        "role",
                        "reasoning_content",
                        "reasoning",
                        "content",
                        "tool_calls",
                        "refusal",
                    },
                    delta_label,
                )
                _privacy_leaf_fields(
                    delta,
                    {
                        "role": ((str, type(None)), "must be a string or null"),
                        "reasoning_content": (
                            (str, type(None)),
                            "must be a string or null",
                        ),
                        "reasoning": (
                            (str, type(None)),
                            "must be a string or null",
                        ),
                        "content": ((str, type(None)), "must be a string or null"),
                        "refusal": ((str, type(None)), "must be a string or null"),
                    },
                    delta_label,
                )
                if "role" in delta and delta["role"] not in {None, "assistant"}:
                    raise RawPromptLeakError(
                        f"{delta_label}.role is not an approved value"
                    )
                _privacy_none_if_present(delta, "refusal", delta_label)
                if "tool_calls" in delta and delta["tool_calls"] is not None:
                    for tool_index, tool in enumerate(
                        _privacy_array(delta["tool_calls"], f"{delta_label}.tool_calls")
                    ):
                        _privacy_tool_shape(
                            tool,
                            f"{delta_label}.tool_calls[{tool_index}]",
                            streaming=True,
                        )
    if "usage" in event and event["usage"] is not None:
        _privacy_usage_shape(event["usage"], f"{label}.usage")
    if "interventions" in event:
        _privacy_intervention_shape(event["interventions"], f"{label}.interventions")
    if "rendered_prompt_evidence" in event:
        _privacy_prompt_evidence_shape(
            event["rendered_prompt_evidence"],
            f"{label}.rendered_prompt_evidence",
        )


def _privacy_blocking_response_shape(value: object, label: str) -> None:
    response = _privacy_object(value, label)
    _privacy_keys(
        response,
        {
            "id",
            "object",
            "created",
            "model",
            "system_fingerprint",
            "choices",
            "usage",
            "service_tier",
            "metadata",
            "interventions",
            "rendered_prompt_evidence",
        },
        label,
    )
    _privacy_leaf_fields(
        response,
        {
            "id": ((str,), "must be a string"),
            "object": ((str,), "must be a string"),
            "created": ((int,), "must be an integer"),
            "model": ((str,), "must be a string"),
            "system_fingerprint": ((str, type(None)), "must be a string or null"),
            "service_tier": ((str, type(None)), "must be a string or null"),
        },
        label,
    )
    if "id" in response:
        _privacy_atlas_response_id(response["id"], f"{label}.id")
    if "model" in response:
        _privacy_safe_model(response["model"], f"{label}.model")
    _privacy_exact_if_present(response, "object", "chat.completion", label)
    _privacy_atlas_fingerprint(response, label)
    _privacy_created_if_present(response, label)
    if "service_tier" in response and response["service_tier"] not in {
        None,
        "auto",
        "default",
    }:
        raise RawPromptLeakError(f"{label}.service_tier is not an approved value")
    if response.get("metadata") is not None:
        raise RawPromptLeakError(f"{label}.metadata is an unapproved nested payload")
    if "choices" in response:
        for choice_index, raw_choice in enumerate(
            _privacy_array(response["choices"], f"{label}.choices")
        ):
            choice_label = f"{label}.choices[{choice_index}]"
            choice = _privacy_object(raw_choice, choice_label)
            _privacy_keys(
                choice,
                {"index", "message", "finish_reason", "logprobs", "token_ids"},
                choice_label,
            )
            _privacy_leaf_fields(
                choice,
                {
                    "index": ((int,), "must be an integer"),
                    "finish_reason": (
                        (str, type(None)),
                        "must be a string or null",
                    ),
                },
                choice_label,
            )
            if "finish_reason" in choice and choice["finish_reason"] not in {
                "length",
                "stop",
                "tool_calls",
            }:
                raise RawPromptLeakError(
                    f"{choice_label}.finish_reason is not an approved value"
                )
            if choice.get("logprobs") is not None:
                raise RawPromptLeakError(
                    f"{choice_label}.logprobs is an unapproved nested payload"
                )
            if "token_ids" in choice:
                _privacy_token_ids_shape(
                    choice["token_ids"], f"{choice_label}.token_ids"
                )
            if "message" in choice:
                message_label = f"{choice_label}.message"
                message = _privacy_object(choice["message"], message_label)
                _privacy_keys(
                    message,
                    {
                        "role",
                        "reasoning_content",
                        "reasoning",
                        "content",
                        "tool_calls",
                        "annotations",
                        "refusal",
                    },
                    message_label,
                )
                _privacy_leaf_fields(
                    message,
                    {
                        "role": ((str, type(None)), "must be a string or null"),
                        "reasoning_content": (
                            (str, type(None)),
                            "must be a string or null",
                        ),
                        "reasoning": (
                            (str, type(None)),
                            "must be a string or null",
                        ),
                        "content": ((str, type(None)), "must be a string or null"),
                        "refusal": ((str, type(None)), "must be a string or null"),
                    },
                    message_label,
                )
                _privacy_exact_if_present(message, "role", "assistant", message_label)
                _privacy_none_if_present(message, "refusal", message_label)
                if message.get("annotations") is not None:
                    raise RawPromptLeakError(
                        f"{message_label}.annotations is an unapproved nested payload"
                    )
                if "tool_calls" in message and message["tool_calls"] is not None:
                    for tool_index, tool in enumerate(
                        _privacy_array(
                            message["tool_calls"], f"{message_label}.tool_calls"
                        )
                    ):
                        _privacy_tool_shape(
                            tool,
                            f"{message_label}.tool_calls[{tool_index}]",
                            streaming=False,
                        )
    if "usage" in response:
        _privacy_usage_shape(response["usage"], f"{label}.usage")
    if "interventions" in response:
        _privacy_intervention_shape(response["interventions"], f"{label}.interventions")
    if "rendered_prompt_evidence" in response:
        _privacy_prompt_evidence_shape(
            response["rendered_prompt_evidence"],
            f"{label}.rendered_prompt_evidence",
        )


def _reject_raw_prompt_fields(value: object, label: str, path: str = "$") -> None:
    """Reject any response field that could carry rendered prompt material."""

    if type(value) is dict:
        for key, child in value.items():
            child_path = f"{path}.{key}"
            normalized = key.lower().replace("-", "_")
            allowed_prompt_metadata = (
                normalized == "rendered_prompt_evidence"
                or (
                    normalized == "prompt_sha256"
                    and path.endswith(".rendered_prompt_evidence")
                )
                or (
                    normalized in {"prompt_tokens", "prompt_tokens_details"}
                    and path.endswith(".usage")
                )
            )
            if "prompt" in normalized and not allowed_prompt_metadata:
                raise RawPromptLeakError(
                    f"{label} contains forbidden raw-prompt field {child_path}"
                )
            _reject_raw_prompt_fields(child, label, child_path)
    elif type(value) is list:
        for index, child in enumerate(value):
            _reject_raw_prompt_fields(child, label, f"{path}[{index}]")


def _usage(value: object, label: str) -> dict[str, Any]:
    usage = _object(value, label)
    _exact_keys(
        usage,
        {
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prompt_tokens_details",
            "completion_tokens_details",
            "time_to_first_token_ms",
            "response_token/s",
        },
        label,
    )
    counts: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        raw = usage.get(key)
        if type(raw) is not int or raw < 0:
            raise GateError(f"{label}.{key} must be a non-negative integer")
        counts[key] = raw
    if counts["total_tokens"] != counts["prompt_tokens"] + counts["completion_tokens"]:
        raise GateError(f"{label}.total_tokens is inconsistent")

    prompt_details = _object(
        usage.get("prompt_tokens_details"), f"{label}.prompt_tokens_details"
    )
    _exact_keys(
        prompt_details,
        {"cached_tokens", "audio_tokens"},
        f"{label}.prompt_tokens_details",
    )
    for key in ("cached_tokens", "audio_tokens"):
        raw = prompt_details.get(key)
        if type(raw) is not int or raw < 0:
            raise GateError(
                f"{label}.prompt_tokens_details.{key} must be a non-negative integer"
            )
        if raw > counts["prompt_tokens"]:
            raise GateError(
                f"{label}.prompt_tokens_details.{key} exceeds prompt_tokens"
            )

    completion_details = _object(
        usage.get("completion_tokens_details"),
        f"{label}.completion_tokens_details",
    )
    completion_detail_keys = {
        "reasoning_tokens",
        "audio_tokens",
        "accepted_prediction_tokens",
        "rejected_prediction_tokens",
    }
    _exact_keys(
        completion_details,
        completion_detail_keys,
        f"{label}.completion_tokens_details",
    )
    for key in completion_detail_keys:
        raw = completion_details.get(key)
        if type(raw) is not int or raw < 0:
            raise GateError(
                f"{label}.completion_tokens_details.{key} must be a non-negative integer"
            )
    for key in ("reasoning_tokens", "audio_tokens", "accepted_prediction_tokens"):
        if completion_details[key] > counts["completion_tokens"]:
            raise GateError(
                f"{label}.completion_tokens_details.{key} exceeds completion_tokens"
            )

    for key in ("time_to_first_token_ms", "response_token/s"):
        raw = usage.get(key)
        if type(raw) not in (int, float) or type(raw) is bool:
            raise GateError(f"{label}.{key} must be numeric")
        if raw < 0 or (type(raw) is float and not math.isfinite(raw)):
            raise GateError(f"{label}.{key} must be finite and non-negative")
    return copy.deepcopy(usage)


def _tool_fragment(
    raw: object,
    event_index: int,
    calls: dict[int, dict[str, Any]],
    fragments: list[dict[str, Any]],
) -> None:
    fragment = _object(raw, f"SSE event {event_index} tool fragment")
    index = fragment.get("index")
    if type(index) is not int or index < 0:
        raise GateError(f"SSE event {event_index} tool index is invalid")
    call = calls.setdefault(
        index,
        {
            "index": index,
            "id": None,
            "type": None,
            "name_parts": [],
            "arguments_parts": [],
        },
    )
    if "id" in fragment and fragment["id"] is not None:
        call_id = _string(fragment["id"], f"SSE event {event_index} tool id")
        if call["id"] not in (None, call_id):
            raise GateError(f"SSE tool {index} changed id")
        call["id"] = call_id
    if "type" in fragment and fragment["type"] is not None:
        call_type = _string(fragment["type"], f"SSE event {event_index} tool type")
        if call_type != "function":
            raise GateError(f"SSE tool {index} type must be function")
        if call["type"] not in (None, call_type):
            raise GateError(f"SSE tool {index} changed type")
        call["type"] = call_type
    function = _object(
        fragment.get("function"), f"SSE event {event_index} tool function"
    )
    name = function.get("name")
    if name is not None:
        call["name_parts"].append(_string(name, f"SSE event {event_index} tool name"))
    arguments = function.get("arguments", "")
    if type(arguments) is not str:
        raise GateError(f"SSE event {event_index} tool arguments must be a string")
    call["arguments_parts"].append(arguments)
    fragments.append(
        {
            "event_index": event_index,
            "index": index,
            "id": fragment.get("id"),
            "type": fragment.get("type"),
            "name": name,
            "arguments": _text_evidence(arguments),
        }
    )


def _finalize_stream_tools(calls: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    final: list[dict[str, Any]] = []
    for index in sorted(calls):
        call = calls[index]
        if call["id"] is None:
            raise GateError(f"SSE tool {index} has no id")
        if call["type"] != "function":
            raise GateError(f"SSE tool {index} has no function type")
        name = "".join(call["name_parts"])
        if not name:
            raise GateError(f"SSE tool {index} has no function name")
        arguments = "".join(call["arguments_parts"])
        parsed_arguments = strict_response_json_loads(
            arguments, f"SSE tool {index} arguments"
        )
        if type(parsed_arguments) is not dict:
            raise RawPromptLeakError(
                f"SSE tool {index} arguments must decode to an object"
            )
        final.append(
            {
                "index": index,
                "id": call["id"],
                "type": call["type"],
                "name": name,
                "arguments": _text_evidence(arguments),
                "parsed_arguments": parsed_arguments,
            }
        )
    return final


def parse_sse(raw: bytes, *, expected_nonce: str) -> dict[str, Any]:
    """Parse strict SSE and preserve whitespace/tool fragments exactly."""

    try:
        raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise RawPromptLeakError(
            "SSE response failed strict privacy-safe UTF-8 decoding"
        ) from error
    if b"\r" in raw.replace(b"\r\n", b""):
        raise RawPromptLeakError("SSE response contains an unsafe framing byte")

    events: list[dict[str, Any]] = []
    data_lines: list[bytes] = []
    done_seen = False

    def dispatch() -> None:
        nonlocal done_seen
        if not data_lines:
            return
        payload = b"\n".join(data_lines)
        data_lines.clear()
        if done_seen:
            raise RawPromptLeakError("SSE contains an unsafe channel after [DONE]")
        if payload == b"[DONE]":
            done_seen = True
            return
        try:
            event_text = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise RawPromptLeakError(
                "SSE data event failed strict privacy-safe UTF-8 decoding"
            ) from error
        event = strict_response_json_loads(event_text, "SSE data event")
        event = _privacy_object(event, "SSE data event")
        _reject_raw_prompt_fields(event, "SSE data event")
        _privacy_stream_event_shape(event, "SSE data event")
        events.append(event)

    if raw and not raw.endswith((b"\n", b"\r\n")):
        raise RawPromptLeakError("SSE response has unsafe truncated framing")
    for wire_line in raw.splitlines(keepends=True):
        if wire_line.endswith(b"\r\n"):
            line = wire_line[:-2]
        elif wire_line.endswith(b"\n"):
            line = wire_line[:-1]
        else:
            raise RawPromptLeakError("SSE response has unsafe truncated framing")
        if not line:
            dispatch()
            continue
        if line.startswith(b":"):
            raise RawPromptLeakError("SSE contains an unapproved non-data channel")
        field, separator, value = line.partition(b":")
        if separator and value.startswith(b" "):
            value = value[1:]
        if field == b"data":
            data_lines.append(value)
        else:
            raise RawPromptLeakError("SSE contains an unapproved non-data channel")
    if data_lines:
        raise RawPromptLeakError("SSE response ends before a safe event delimiter")
    if not done_seen:
        raise RawPromptLeakError("SSE response is missing the strict [DONE] marker")

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    emission_parts: list[str] = []
    finish_reasons: list[str] = []
    usage_value: dict[str, Any] | None = None
    calls: dict[int, dict[str, Any]] = {}
    fragments: list[dict[str, Any]] = []
    token_fields: list[dict[str, Any]] = []
    response_ids: set[str] = set()
    response_models: set[str] = set()
    interventions: dict[str, Any] | None = None
    prompt_evidence: dict[str, Any] | None = None
    finish_event_index: int | None = None
    usage_event_index: int | None = None
    prompt_fields: list[dict[str, Any]] = []

    for event_index, event in enumerate(events):
        token_fields.extend(_token_id_occurrences(event, f"$.events[{event_index}]"))
        prompt_fields.extend(
            _field_occurrences(
                event, "rendered_prompt_evidence", f"$.events[{event_index}]"
            )
        )
        if type(event.get("id")) is str:
            response_ids.add(event["id"])
        if type(event.get("model")) is str:
            response_models.add(event["model"])
        if event.get("usage") is not None:
            if usage_value is not None:
                raise GateError("SSE emitted more than one usage object")
            parsed_usage = _usage(event["usage"], f"SSE event {event_index}.usage")
            usage_value = parsed_usage
            usage_event_index = event_index

        if "interventions" in event:
            if interventions is not None:
                raise GateError("SSE emitted more than one intervention ledger")
            ledger_choices = event.get("choices")
            if (
                type(ledger_choices) is not list
                or len(ledger_choices) != 1
                or type(ledger_choices[0]) is not dict
                or ledger_choices[0].get("finish_reason") is None
            ):
                raise GateError(
                    "SSE intervention ledger must be attached to the terminal "
                    "finish-bearing choice"
                )
            interventions = _empty_intervention_ledger(
                event["interventions"], f"SSE event {event_index}.interventions"
            )

        choices = _array(event.get("choices"), f"SSE event {event_index}.choices")
        if interventions is not None and "interventions" not in event and choices:
            raise GateError(
                "SSE emitted a choice after the terminal intervention ledger"
            )
        if not choices:
            if event.get("usage") is None:
                raise GateError(
                    f"SSE event {event_index} has neither choices nor usage"
                )
            continue
        if len(choices) != 1:
            raise GateError(f"SSE event {event_index} must contain exactly one choice")
        choice = _object(choices[0], f"SSE event {event_index}.choices[0]")
        _exact_int(choice.get("index"), 0, f"SSE event {event_index} choice.index")
        delta = _object(choice.get("delta"), f"SSE event {event_index} choice.delta")

        reasoning_content = delta.get("reasoning_content")
        reasoning_alias = delta.get("reasoning")
        if reasoning_content is not None and type(reasoning_content) is not str:
            raise GateError(
                f"SSE event {event_index} reasoning_content is not a string"
            )
        if reasoning_alias is not None and type(reasoning_alias) is not str:
            raise GateError(f"SSE event {event_index} reasoning is not a string")
        if (
            reasoning_content is not None
            and reasoning_alias is not None
            and reasoning_content != reasoning_alias
        ):
            raise GateError(f"SSE event {event_index} reasoning aliases differ")
        reasoning_piece = (
            reasoning_content if reasoning_content is not None else reasoning_alias
        )
        if reasoning_piece is not None:
            reasoning_parts.append(reasoning_piece)
            emission_parts.append(reasoning_piece)

        content = delta.get("content")
        if content is not None:
            if type(content) is not str:
                raise GateError(f"SSE event {event_index} content is not a string")
            content_parts.append(content)
            emission_parts.append(content)

        tool_calls = delta.get("tool_calls")
        if tool_calls is not None:
            for fragment in _array(
                tool_calls, f"SSE event {event_index} delta.tool_calls"
            ):
                _tool_fragment(fragment, event_index, calls, fragments)

        finish_reason = choice.get("finish_reason")
        if finish_reason is not None:
            finish_event_index = event_index
            finish_reasons.append(
                _string(finish_reason, f"SSE event {event_index} finish_reason")
            )

    if len(finish_reasons) != 1:
        raise GateError(
            f"SSE must emit exactly one finish_reason, got {len(finish_reasons)}"
        )
    if usage_value is None:
        raise GateError("SSE is missing terminal usage")
    if interventions is None:
        raise GateError("SSE is missing the terminal intervention ledger")
    if len(response_ids) > 1:
        raise RawPromptLeakError("SSE response id changed between events")
    if len(response_models) > 1:
        raise RawPromptLeakError("SSE response model changed between events")

    if finish_event_index is None:
        raise GateError("SSE is missing a finish-bearing terminal choice")
    response_id = next(iter(response_ids), None)
    if type(response_id) is not str or not response_id:
        raise GateError("SSE response id must be a stable non-empty string")
    expected_token_path = f"$.events[{finish_event_index}].choices[0].token_ids"
    if len(token_fields) != 1 or token_fields[0]["path"] != expected_token_path:
        raise GateError(
            "token_ids must appear exactly once on the finish-bearing terminal choice"
        )
    terminal_token_ids = token_fields[0]["value"]
    if type(terminal_token_ids) is not list or not all(
        type(token) is int and token >= 0 for token in terminal_token_ids
    ):
        raise GateError("terminal token_ids must be non-negative built-in integers")

    expected_prompt_path = f"$.events[{finish_event_index}].rendered_prompt_evidence"
    if len(prompt_fields) != 1 or prompt_fields[0]["path"] != expected_prompt_path:
        raise GateError(
            "rendered_prompt_evidence must appear exactly once on the terminal SSE event"
        )
    prompt_evidence = _rendered_prompt_evidence(
        prompt_fields[0]["value"],
        "terminal SSE rendered_prompt_evidence",
        expected_nonce=expected_nonce,
        response_id=response_id,
    )
    if usage_event_index is None or usage_event_index <= finish_event_index:
        raise GateError("SSE terminal usage must follow the finish-bearing choice")
    if usage_event_index != len(events) - 1:
        raise GateError("SSE terminal usage must be the final JSON event")

    content_text = "".join(content_parts)
    reasoning_text = "".join(reasoning_parts)
    emitted_text = "".join(emission_parts)
    return {
        "event_count": len(events),
        "done_seen": done_seen,
        "response_id": response_id,
        "model": next(iter(response_models), None),
        "finish_reason": finish_reasons[0],
        "content": _text_evidence(content_text),
        "reasoning_content": _text_evidence(reasoning_text),
        "emitted_text": _text_evidence(emitted_text),
        "max_consecutive_blank_lines": max_consecutive_blank_lines(emitted_text),
        "tool_fragments": fragments,
        "tool_calls": _finalize_stream_tools(calls),
        "usage": usage_value,
        "token_id_fields": token_fields,
        "terminal_token_ids": terminal_token_ids,
        "interventions": interventions,
        "rendered_prompt_evidence": prompt_evidence,
    }


def _blocking_tools(value: object) -> list[dict[str, Any]]:
    if value is None:
        return []
    tools = _array(value, "blocking message.tool_calls")
    final: list[dict[str, Any]] = []
    for index, raw in enumerate(tools):
        tool = _object(raw, f"blocking tool_calls[{index}]")
        tool_id = _string(tool.get("id"), f"blocking tool_calls[{index}].id")
        if tool.get("type") != "function":
            raise GateError(f"blocking tool_calls[{index}].type must be function")
        function = _object(
            tool.get("function"), f"blocking tool_calls[{index}].function"
        )
        name = _string(function.get("name"), f"blocking tool_calls[{index}].name")
        arguments = _string(
            function.get("arguments"),
            f"blocking tool_calls[{index}].arguments",
            nonempty=False,
        )
        parsed_arguments = strict_response_json_loads(
            arguments, f"blocking tool {index} arguments"
        )
        if type(parsed_arguments) is not dict:
            raise RawPromptLeakError(
                f"blocking tool {index} arguments must decode to an object"
            )
        final.append(
            {
                "index": index,
                "id": tool_id,
                "type": "function",
                "name": name,
                "arguments": _text_evidence(arguments),
                "parsed_arguments": parsed_arguments,
            }
        )
    return final


def parse_blocking(raw: bytes, *, expected_nonce: str) -> dict[str, Any]:
    response = _decode_json(raw, "blocking response")
    _reject_raw_prompt_fields(response, "blocking response")
    _privacy_blocking_response_shape(response, "blocking response")
    interventions = _empty_intervention_ledger(
        response.get("interventions"), "blocking response.interventions"
    )
    choices = _array(response.get("choices"), "blocking response.choices")
    if len(choices) != 1:
        raise GateError("blocking response must contain exactly one choice")
    choice = _object(choices[0], "blocking response.choices[0]")
    _exact_int(choice.get("index"), 0, "blocking response choice.index")
    finish_reason = _string(choice.get("finish_reason"), "blocking finish_reason")
    message = _object(choice.get("message"), "blocking response message")
    if message.get("role") != "assistant":
        raise GateError("blocking response role must be assistant")

    content = message.get("content")
    if content is None:
        content = ""
    if type(content) is not str:
        raise GateError("blocking response content must be a string or null")
    reasoning_content = message.get("reasoning_content")
    reasoning_alias = message.get("reasoning")
    if reasoning_alias is not None and type(reasoning_alias) is not str:
        raise GateError("blocking reasoning must be a string or null")
    if reasoning_content is None:
        reasoning_content = reasoning_alias if reasoning_alias is not None else ""
    if type(reasoning_content) is not str:
        raise GateError("blocking reasoning_content must be a string or null")
    if (
        reasoning_alias is not None
        and type(reasoning_alias) is str
        and reasoning_alias != reasoning_content
    ):
        raise GateError("blocking reasoning aliases differ")
    emitted = reasoning_content + content
    response_id = _string(response.get("id"), "blocking response.id")
    token_fields = _token_id_occurrences(response)
    if len(token_fields) != 1 or token_fields[0]["path"] != "$.choices[0].token_ids":
        raise GateError("blocking token_ids must appear exactly once on choices[0]")
    token_ids = token_fields[0]["value"]
    if type(token_ids) is not list or not all(
        type(token) is int and token >= 0 for token in token_ids
    ):
        raise GateError("blocking token_ids must be non-negative built-in integers")
    prompt_fields = _field_occurrences(response, "rendered_prompt_evidence")
    if (
        len(prompt_fields) != 1
        or prompt_fields[0]["path"] != "$.rendered_prompt_evidence"
    ):
        raise GateError(
            "blocking rendered_prompt_evidence must appear exactly once at top level"
        )
    prompt_evidence = _rendered_prompt_evidence(
        prompt_fields[0]["value"],
        "blocking rendered_prompt_evidence",
        expected_nonce=expected_nonce,
        response_id=response_id,
    )
    return {
        "response_id": response_id,
        "model": response.get("model"),
        "finish_reason": finish_reason,
        "content": _text_evidence(content),
        "reasoning_content": _text_evidence(reasoning_content),
        "emitted_text": _text_evidence(emitted),
        "max_consecutive_blank_lines": max_consecutive_blank_lines(emitted),
        "tool_calls": _blocking_tools(message.get("tool_calls")),
        "usage": _usage(response.get("usage"), "blocking response.usage"),
        "token_id_fields": token_fields,
        "terminal_token_ids": token_ids,
        "interventions": interventions,
        "rendered_prompt_evidence": prompt_evidence,
    }


def _validate_tool_arguments(
    calls: list[dict[str, Any]], label: str, request: dict[str, Any]
) -> None:
    schemas: dict[str, dict[str, Any]] = {}
    for index, raw_tool in enumerate(_array(request.get("tools"), "request.tools")):
        tool = _object(raw_tool, f"request.tools[{index}]")
        function = _object(tool.get("function"), f"request.tools[{index}].function")
        name = _string(function.get("name"), f"request.tools[{index}].function.name")
        schemas[name] = _object(
            function.get("parameters"),
            f"request.tools[{index}].function.parameters",
        )

    for index, call in enumerate(calls):
        name = call["name"]
        if name not in schemas:
            raise RawPromptLeakError(
                f"{label} tool[{index}] has no sealed request schema"
            )
        schema = schemas[name]
        if schema != SEALED_TOOL_SCHEMAS.get(name):
            raise RawPromptLeakError(
                f"{label} tool[{index}] schema is not the sealed gate schema"
            )
        arguments = call["parsed_arguments"]
        _reject_raw_prompt_fields(arguments, f"{label} tool[{index}] decoded arguments")
        if type(arguments) is not dict or set(arguments) != {"city"}:
            raise RawPromptLeakError(
                f"{label} tool[{index}] arguments violate required city and "
                "additionalProperties=false"
            )
        if type(arguments["city"]) is not str:
            raise RawPromptLeakError(f"{label} tool[{index}].city must be a string")
        if arguments != SEALED_TOOL_ARGUMENTS[name]:
            raise RawPromptLeakError(
                f"{label} tool[{index}] arguments differ from the sealed gate value"
            )


def _bind_summary_to_manifest(
    summary: dict[str, Any], label: str, manifest: dict[str, Any]
) -> None:
    if summary.get("model") != manifest["request"]["model"]:
        raise RawPromptLeakError(
            f"{label} response model is not bound to the sealed request"
        )
    _validate_tool_arguments(summary["tool_calls"], label, manifest["request"])


def _semantic_tool_view(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": tool["name"],
        "arguments_utf8_hex": tool["arguments"]["utf8_hex"],
        "parsed_arguments": tool["parsed_arguments"],
    }


def _validate_one_summary(
    summary: dict[str, Any],
    label: str,
    manifest: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    finish_reason = summary["finish_reason"]
    if finish_reason == "length":
        errors.append(f"{label}: finish_reason=length")
    elif finish_reason not in {"stop", "tool_calls"}:
        errors.append(f"{label}: unsupported finish_reason={finish_reason!r}")
    if summary["max_consecutive_blank_lines"] >= 4:
        errors.append(
            f"{label}: {summary['max_consecutive_blank_lines']} consecutive blank lines"
        )

    calls = summary["tool_calls"]
    if len(calls) == 0:
        errors.append(f"{label}: expected one tool call, got zero")
    if len(calls) > 1:
        errors.append(
            f"{label}: parallel_tool_calls=false but response emitted {len(calls)} tools"
        )
    if calls and finish_reason != "tool_calls":
        errors.append(f"{label}: tool response must finish with tool_calls")
    expected = set(manifest["expected_tool_names"])
    for call in calls:
        if call["name"] not in expected:
            errors.append(f"{label}: unexpected tool {call['name']!r}")

    completion_tokens = summary["usage"]["completion_tokens"]
    if completion_tokens >= manifest["token_cap"]:
        errors.append(
            f"{label}: completion token count {completion_tokens} reached configured "
            f"cap {manifest['token_cap']}"
        )
    model = summary.get("model")
    if model is not None and model != manifest["request"]["model"]:
        errors.append(
            f"{label}: response model {model!r} differs from request model "
            f"{manifest['request']['model']!r}"
        )
    return errors


def compare_summaries(
    stream: dict[str, Any],
    blocking: dict[str, Any],
    manifest: dict[str, Any],
) -> list[str]:
    errors = _validate_one_summary(stream, "stream", manifest)
    errors.extend(_validate_one_summary(blocking, "blocking", manifest))
    for field in ("content", "reasoning_content", "emitted_text"):
        if stream[field]["utf8_hex"] != blocking[field]["utf8_hex"]:
            errors.append(f"streaming-concat bytes differ from blocking {field}")
    if stream["finish_reason"] != blocking["finish_reason"]:
        errors.append("stream and blocking finish_reason differ")
    if {
        key: stream["usage"][key]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    } != {
        key: blocking["usage"][key]
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }:
        errors.append("stream and blocking token usage differ")
    if [_semantic_tool_view(tool) for tool in stream["tool_calls"]] != [
        _semantic_tool_view(tool) for tool in blocking["tool_calls"]
    ]:
        errors.append("stream and blocking tool calls differ")

    for label, summary in (("stream", stream), ("blocking", blocking)):
        token_ids = summary["terminal_token_ids"]
        if len(token_ids) != summary["usage"]["completion_tokens"]:
            errors.append(
                f"{label} terminal token_ids length differs from usage.completion_tokens"
            )
    if stream["terminal_token_ids"] != blocking["terminal_token_ids"]:
        errors.append("stream and blocking terminal token_ids differ")
    stream_prompt = stream["rendered_prompt_evidence"]
    blocking_prompt = blocking["rendered_prompt_evidence"]
    if stream_prompt["prompt_sha256"] != blocking_prompt["prompt_sha256"]:
        errors.append("stream and blocking server-rendered prompt digests differ")
    return errors


def _capture_meta(
    capture: HttpCapture, raw_name: str, normalized_content_type: str
) -> dict[str, Any]:
    return {
        "status": capture.status,
        "content_type": normalized_content_type,
        "raw_path": raw_name,
        "raw_bytes": len(capture.raw),
        "raw_sha256": sha256_bytes(capture.raw),
        "elapsed_ms": capture.elapsed_ms,
        "first_body_byte_ms": capture.first_body_byte_ms,
        "line_timings_ms": list(capture.line_timings_ms),
    }


def evaluate_captures(
    manifest_value: object,
    stream_capture: HttpCapture,
    blocking_capture: HttpCapture,
    *,
    pair_elapsed_ms: float,
    request_pair: RequestPair,
    capture_mode: str = "offline-fixture",
    endpoint: str | None = None,
    live_provenance: object | None = None,
) -> dict[str, Any]:
    manifest = validate_manifest(manifest_value)
    capture_binding = _capture_binding(
        capture_mode,
        endpoint=endpoint,
        live_provenance=live_provenance,
    )
    _bind_capture_to_manifest(capture_binding, manifest)
    expected_pair = request_bodies(
        manifest,
        stream_nonce=request_pair.stream_nonce,
        blocking_nonce=request_pair.blocking_nonce,
    )
    if request_pair != expected_pair:
        raise GateError("request pair is not the exact nonced manifest request")
    errors: list[str] = []
    stream_content_type = _normalize_content_type(
        stream_capture.content_type, "text/event-stream", "stream"
    )
    blocking_content_type = _normalize_content_type(
        blocking_capture.content_type, "application/json", "blocking"
    )
    pair_elapsed_ms = _nonnegative_number(pair_elapsed_ms, "pair_elapsed_ms")
    for label, capture in (
        ("stream", stream_capture),
        ("blocking", blocking_capture),
    ):
        _nonnegative_number(capture.elapsed_ms, f"{label}.elapsed_ms")
        if capture.first_body_byte_ms != -1.0:
            _nonnegative_number(
                capture.first_body_byte_ms, f"{label}.first_body_byte_ms"
            )
        for index, offset in enumerate(capture.line_timings_ms):
            _nonnegative_number(offset, f"{label}.line_timings_ms[{index}]")
    timeout_ms = manifest["timeout_seconds"] * 1000.0
    if pair_elapsed_ms > timeout_ms:
        errors.append(
            f"paired requests exceeded deadline: {pair_elapsed_ms:.3f} ms > {timeout_ms:.3f} ms"
        )

    if stream_capture.status != 200:
        errors.append(f"stream HTTP status is {stream_capture.status}, expected 200")
    if blocking_capture.status != 200:
        errors.append(
            f"blocking HTTP status is {blocking_capture.status}, expected 200"
        )

    stream_summary: dict[str, Any] | None = None
    blocking_summary: dict[str, Any] | None = None
    try:
        stream_summary = parse_sse(
            stream_capture.raw, expected_nonce=request_pair.stream_nonce
        )
    except RawPromptLeakError:
        # Do not construct a receipt that would later persist the raw capture.
        raise
    except GateError:
        raise RawPromptLeakError(
            "stream response failed complete privacy-safe validation"
        ) from None
    try:
        blocking_summary = parse_blocking(
            blocking_capture.raw, expected_nonce=request_pair.blocking_nonce
        )
    except RawPromptLeakError:
        # Do not construct a receipt that would later persist the raw capture.
        raise
    except GateError:
        raise RawPromptLeakError(
            "blocking response failed complete privacy-safe validation"
        ) from None
    if stream_summary is not None:
        _bind_summary_to_manifest(stream_summary, "stream", manifest)
    if blocking_summary is not None:
        _bind_summary_to_manifest(blocking_summary, "blocking", manifest)
    if stream_summary is not None and blocking_summary is not None:
        errors.extend(compare_summaries(stream_summary, blocking_summary, manifest))
        try:
            expected_prompt = expected_rendered_prompt(manifest["request"])
            expected_audit = validate_rendered_prompt(
                expected_prompt, manifest["request"]
            )
        except GateError as error:
            errors.append(f"expected rendered prompt: {error}")
            expected_prompt = ""
            expected_audit = None
        for label, summary in (
            ("stream", stream_summary),
            ("blocking", blocking_summary),
        ):
            summary["rendered_prompt_audit"] = expected_audit
            evidence = summary["rendered_prompt_evidence"]
            if expected_prompt and evidence["prompt_sha256"] != sha256_bytes(
                expected_prompt.encode("utf-8")
            ):
                errors.append(
                    f"{label} server-rendered prompt digest differs from the exact request envelope"
                )

    return {
        "schema": RECEIPT_SCHEMA,
        "result": "PASS" if not errors else "FAIL",
        "errors": errors,
        "capture": capture_binding,
        "manifest_sha256": sha256_bytes(canonical_bytes(manifest_value)),
        "launch_contract": {
            "profile": manifest["launch_contract"]["profile"],
            "concurrency": manifest["launch_contract"]["concurrency"],
            "max_batch_size": manifest["launch_contract"]["max_batch_size"],
            "max_num_seqs": manifest["launch_contract"]["max_num_seqs"],
            "prefill_chunk_tokens": manifest["launch_contract"]["prefill_chunk_tokens"],
            "kv_cache_dtype": manifest["launch_contract"]["kv_cache_dtype"],
            "speculation": manifest["launch_contract"]["speculation"],
            "tool_call_parser": manifest["launch_contract"]["tool_call_parser"],
            "chat_template": manifest["launch_contract"]["chat_template"],
            "rendered_prompt_audit": {
                "stream": (
                    None
                    if stream_summary is None
                    else stream_summary.get("rendered_prompt_audit")
                ),
                "blocking": (
                    None
                    if blocking_summary is None
                    else blocking_summary.get("rendered_prompt_audit")
                ),
            },
        },
        "request": {
            "model": manifest["request"]["model"],
            "expected_tool_names": manifest["expected_tool_names"],
            "parallel_tool_calls": False,
            "token_cap": manifest["token_cap"],
            "timeout_seconds": manifest["timeout_seconds"],
            "stream_sha256": sha256_bytes(canonical_bytes(request_pair.stream)),
            "blocking_sha256": sha256_bytes(canonical_bytes(request_pair.blocking)),
        },
        "pair_elapsed_ms": pair_elapsed_ms,
        "stream_capture": _capture_meta(
            stream_capture, "response-stream.sse", stream_content_type
        ),
        "blocking_capture": _capture_meta(
            blocking_capture, "response-blocking.json", blocking_content_type
        ),
        "stream": stream_summary,
        "blocking": blocking_summary,
    }


def fixture_captures(
    value: object,
) -> tuple[dict[str, Any], HttpCapture, HttpCapture, float, RequestPair]:
    fixture = _object(value, "fixture")
    _exact_keys(
        fixture,
        {
            "schema",
            "manifest",
            "request_nonces",
            "stream_http",
            "blocking_http",
            "pair_elapsed_ms",
        },
        "fixture",
    )
    if fixture["schema"] != FIXTURE_SCHEMA:
        raise GateError(f"fixture.schema must be {FIXTURE_SCHEMA}")

    def one(raw_value: object, label: str, raw_key: str) -> HttpCapture:
        item = _object(raw_value, label)
        required = {
            "status",
            "content_type",
            raw_key,
            "elapsed_ms",
            "first_body_byte_ms",
        }
        optional = {"line_timings_ms"}
        if not required <= set(item) or set(item) - required - optional:
            raise GateError(
                f"{label} keys differ: missing={sorted(required - set(item))!r} "
                f"extra={sorted(set(item) - required - optional)!r}"
            )
        raw_text = _string(item[raw_key], f"{label}.{raw_key}", nonempty=False)
        line_timings = item.get("line_timings_ms", [])
        _array(line_timings, f"{label}.line_timings_ms")
        return HttpCapture(
            status=_positive_int(item["status"], f"{label}.status"),
            content_type=_string(item["content_type"], f"{label}.content_type"),
            raw=raw_text.encode("utf-8"),
            elapsed_ms=_nonnegative_number(item["elapsed_ms"], f"{label}.elapsed_ms"),
            first_body_byte_ms=_nonnegative_number(
                item["first_body_byte_ms"], f"{label}.first_body_byte_ms"
            ),
            line_timings_ms=tuple(
                _nonnegative_number(offset, f"{label}.line_timings_ms[{index}]")
                for index, offset in enumerate(line_timings)
            ),
        )

    stream = one(fixture["stream_http"], "fixture.stream_http", "raw_sse_utf8")
    blocking = one(fixture["blocking_http"], "fixture.blocking_http", "raw_body_utf8")
    pair_elapsed_ms = _nonnegative_number(
        fixture["pair_elapsed_ms"], "fixture.pair_elapsed_ms"
    )
    nonces = _object(fixture["request_nonces"], "fixture.request_nonces")
    _exact_keys(nonces, {"stream", "blocking"}, "fixture.request_nonces")
    manifest = _object(fixture["manifest"], "fixture.manifest")
    request_pair = request_bodies(
        validate_manifest(manifest),
        stream_nonce=nonces["stream"],
        blocking_nonce=nonces["blocking"],
    )
    return (
        manifest,
        stream,
        blocking,
        pair_elapsed_ms,
        request_pair,
    )


def _exclusive_write(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        offset = 0
        while offset < len(payload):
            written = handle.write(payload[offset:])
            if written is None or written <= 0:
                raise OSError(f"write made no progress: {path}")
            offset += written
        handle.flush()
        os.fsync(handle.fileno())


def write_artifacts(
    out_dir: Path,
    manifest: dict[str, Any],
    stream_capture: HttpCapture,
    blocking_capture: HttpCapture,
    receipt: dict[str, Any],
    request_pair: RequestPair,
) -> None:
    out_dir.mkdir(mode=0o750, parents=False, exist_ok=False)
    _exclusive_write(out_dir / "manifest.json", canonical_bytes(manifest) + b"\n")
    _exclusive_write(
        out_dir / "request-stream.json", canonical_bytes(request_pair.stream) + b"\n"
    )
    _exclusive_write(
        out_dir / "request-blocking.json",
        canonical_bytes(request_pair.blocking) + b"\n",
    )
    _exclusive_write(out_dir / "response-stream.sse", stream_capture.raw)
    _exclusive_write(out_dir / "response-blocking.json", blocking_capture.raw)
    _exclusive_write(out_dir / "receipt.json", canonical_bytes(receipt) + b"\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="live-run manifest JSON")
    source.add_argument(
        "--fixture", type=Path, help="offline fake-capture fixture JSON"
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8899/v1/chat/completions",
        help="full chat-completions endpoint for a live run",
    )
    parser.add_argument(
        "--api-key-env",
        default="FLUX_API_KEY",
        help="environment variable containing the bearer token; never written to receipts",
    )
    parser.add_argument(
        "--live-provenance",
        type=Path,
        help=(
            "operator-declared provenance JSON required for a live capture; "
            "does not grant runtime authority"
        ),
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        live_provenance: dict[str, Any] | None = None
        if args.fixture is not None:
            if args.live_provenance is not None:
                raise GateError("offline fixture mode cannot claim live provenance")
            capture_mode = "offline-fixture"
            fixture = _json_file(args.fixture, "fixture")
            (
                manifest,
                stream_capture,
                blocking_capture,
                pair_elapsed_ms,
                request_pair,
            ) = fixture_captures(fixture)
        else:
            if args.live_provenance is None:
                raise GateError("live mode requires --live-provenance")
            capture_mode = "live-http"
            live_provenance = _json_file(args.live_provenance, "live_provenance")
            # Validate the committed runner and the operator declaration before
            # making either request. Runtime identity remains unobserved.
            preliminary_binding = _capture_binding(
                capture_mode,
                endpoint=args.endpoint,
                live_provenance=live_provenance,
            )
            manifest = _json_file(args.manifest, "manifest")
            validated = validate_manifest(manifest)
            _bind_capture_to_manifest(preliminary_binding, validated)
            request_pair = request_bodies(validated)
            started = time.monotonic()
            deadline = started + validated["timeout_seconds"]
            api_key = os.environ.get(args.api_key_env) or None
            stream_capture = http_post(
                args.endpoint,
                request_pair.stream,
                api_key,
                deadline,
                stream=True,
            )
            blocking_capture = http_post(
                args.endpoint,
                request_pair.blocking,
                api_key,
                deadline,
                stream=False,
            )
            pair_elapsed_ms = (time.monotonic() - started) * 1000.0

        receipt = evaluate_captures(
            manifest,
            stream_capture,
            blocking_capture,
            pair_elapsed_ms=pair_elapsed_ms,
            request_pair=request_pair,
            capture_mode=capture_mode,
            endpoint=args.endpoint if capture_mode == "live-http" else None,
            live_provenance=live_provenance,
        )
        write_artifacts(
            args.out_dir,
            manifest,
            stream_capture,
            blocking_capture,
            receipt,
            request_pair,
        )
    except RawPromptLeakError:
        # A response-controlled field name, value, or structural path may be
        # the attempted exfiltration channel.  Never reflect any part of that
        # exception across the process boundary.
        print(
            "FAIL: untrusted response rejected before artifact creation",
            file=sys.stderr,
        )
        return 2
    except GateError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2
    except OSError:
        print("FAIL: filesystem operation failed", file=sys.stderr)
        return 2

    print(f"{receipt['result']}: {args.out_dir / 'receipt.json'}")
    for error in receipt["errors"]:
        print(f"  - {error}", file=sys.stderr)
    return 0 if receipt["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
