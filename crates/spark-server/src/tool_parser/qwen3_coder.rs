// SPDX-License-Identifier: AGPL-3.0-only
#![allow(unused_imports, dead_code)]

use super::*;

/// Qwen3-Coder format: XML `<function=name><parameter=key>value</parameter></function>`
/// inside `<tool_call>` tags.
///
/// Used by Qwen3.5-27B, Qwen3.5-35B, Qwen3.5-122B, and Nemotron-H-30B.
pub struct Qwen3CoderParser;

/// Compatibility prompt suffix appended to `Bash`/`bash` tool descriptions to
/// prevent blind retries after permanent `command not found` / exit-code-127
/// failures. The `[atlas-f33]` marker and exact suffix text are preserved
/// verbatim as pinned prompt compatibility markers for downstream gate checks.
const F33_BASH_RETRY_SUFFIX: &str = " | After ONE failure with \"command not found\" or exit code 127, do NOT retry the same command — the binary is permanently unavailable in this environment. Choose a different approach or tell the user the dependency is missing.";

/// Apply the Bash retry guard to tool definitions rendered by a native Qwen
/// Jinja template. The parser-owned prompt uses the same helper, keeping the
/// native-template and fallback contracts identical without rendering a second
/// copy of the schema. The `[atlas-f33]` deduplication tag is preserved
/// verbatim as a pinned compatibility prompt marker.
pub(crate) fn qwen3_coder_tools_with_atlas_safety(tools: &[ToolDefinition]) -> Vec<ToolDefinition> {
    tools
        .iter()
        .map(|tool| {
            if !matches!(tool.function.name.as_str(), "Bash" | "bash") {
                return tool.clone();
            }
            let mut guarded = tool.clone();
            guarded.function.description = Some(match guarded.function.description {
                Some(description) if !description.contains("[atlas-f33]") => {
                    format!("{description}\n[atlas-f33]{F33_BASH_RETRY_SUFFIX}")
                }
                Some(description) => description,
                None => format!("[atlas-f33]{F33_BASH_RETRY_SUFFIX}"),
            });
            guarded
        })
        .collect()
}

/// Behavioral delta retained when Qwen4/Flash-Next's checkpoint Jinja owns the
/// tool schema and XML wire-format instructions. This deliberately contains no
/// schema or example call, so the rendered prompt has exactly one protocol
/// definition while preserving anti-narration and tool-choice guards. The
/// `<ATLAS_TOOL_SAFETY>` / `</ATLAS_TOOL_SAFETY>` wrapper tags are preserved
/// verbatim as pinned compatibility prompt markers.
pub(crate) fn qwen3_coder_native_safety_prompt(tool_choice: &ToolChoice) -> String {
    let mut prompt = String::from(
        "<ATLAS_TOOL_SAFETY>\n\
Tools are immediately executable. Once you decide to use a tool, emit the function call directly. Put commands, file content, and every other tool input only in the call arguments; do not pre-render them as prose or markdown fences.\n\
For Bash/bash, do not stage the command in a bash fence or narrate that you are about to run it. For Write/Edit, do not print the file body before calling the tool.\n\
Use separate calls for multiple tools. Never simulate a tool response, file listing, or other result that the tool has not returned.",
    );
    match tool_choice {
        ToolChoice::Mode(mode) if mode == "required" => {
            prompt.push_str(
                "\nYou must call at least one offered function. Do not answer with ordinary text instead.",
            );
        }
        ToolChoice::Specific { function } => {
            prompt.push_str(&format!(
                "\nYou must call the '{}' function. Do not call a different function or answer with ordinary text instead.",
                function.name,
            ));
        }
        _ => {}
    }
    prompt.push_str("\n</ATLAS_TOOL_SAFETY>");
    prompt
}

impl ToolCallParser for Qwen3CoderParser {
    fn name(&self) -> &str {
        "qwen3_coder"
    }

    fn compile_tool_grammar(
        &self,
        engine: &mut GrammarEngine,
        tools: &[ToolDefinition],
        use_triggers: bool,
        stop_after_first: bool,
    ) -> Option<Result<CompiledGrammar, GrammarError>> {
        Some(engine.compile_qwen3_coder_tool_grammar(tools, use_triggers, stop_after_first))
    }

    fn compile_tool_or_response_format_grammar(
        &self,
        engine: &mut GrammarEngine,
        tools: &[ToolDefinition],
        stop_after_first: bool,
        response_format: &crate::api::ResponseFormatConstraint,
    ) -> Option<Result<CompiledGrammar, GrammarError>> {
        Some(engine.compile_qwen3_coder_tool_or_response_format_grammar(
            tools,
            stop_after_first,
            response_format,
        ))
    }

    fn has_tool_grammar(&self) -> bool {
        true
    }

    fn system_prompt(&self, tools: &[ToolDefinition], tool_choice: &ToolChoice) -> String {
        // Match the model's native Jinja chat_template exactly:
        // Tool definitions as JSON (not XML), response format as XML.
        let mut prompt =
            String::from("# Tools\n\nYou have access to the following functions:\n\n<tools>\n");
        // Inject a one-line retry rule into the Bash tool's `description`.
        // Tool-description text is part of the tool schema render and is
        // attended on every Bash call. Applied to a working copy so the rule
        // is present whether the body renders as JSON or TSCG.
        let f33_tools = qwen3_coder_tools_with_atlas_safety(tools);
        if crate::tscg::tscg_enabled() {
            // TSCG: compact function signatures replace the JSON body.
            prompt.push_str(&crate::tscg::compile_tools(&f33_tools));
            prompt.push('\n');
        } else {
            for t in &f33_tools {
                prompt.push_str(&serde_json::to_string(t).unwrap_or_default());
                prompt.push('\n');
            }
        }
        // Keep the instruction block concise: one positive-shot example plus
        // declarative rules avoids long lists of negative examples that can
        // prime pattern-completion models into reproducing bad formats.
        prompt.push_str("\
</tools>\n\n\
If you choose to call a function ONLY reply in the following format with NO suffix:\n\n\
<tool_call>\n\
<function=example_function_name>\n\
<parameter=example_parameter_1>\n\
value_1\n\
</parameter>\n\
<parameter=example_parameter_2>\n\
This is the value for the second parameter\n\
that can span\n\
multiple lines\n\
</parameter>\n\
</function>\n\
</tool_call>\n\n\
<IMMEDIATE_TOOL_USE>\n\
Tools are IMMEDIATELY EXECUTABLE. When you decide to use a tool, emit the <tool_call> directly. The tool_call's parameter values are the ONLY place file content, commands, or other tool inputs should appear — do NOT pre-render that content as prose or in markdown fences before the call.\n\
For 'bash'/'Bash' tools specifically: do NOT stage the command in a ```bash``` fence and do NOT prefix with phrases like \"Let me run this command:\" or \"Let me execute:\". The next emission after deciding to run a shell command must be the <tool_call> — the command goes inside <parameter=command>, not in markdown.\n\
For 'Write'/'Edit' tools specifically: do NOT pre-render the file content as a markdown ```toml/```rust/```python fence before the tool call. Do NOT write phrases like \"Let me create the Cargo.toml:\" or \"Now the source file:\" followed by a code fence containing the file body — the file body goes inside <parameter=content>, not in markdown. Pre-rendering the same content twice (once in markdown, once in the parameter) wastes tokens, can cause the tool call to be dropped if the markdown is long enough, and is the documented \"narrate-then-tool\" loop pattern. The next emission after deciding to write a file must be the <tool_call>.\n\
Example:\n\
    <tool_call>\n\
    <function=Write>\n\
    <parameter=file_path>/path/to/Cargo.toml</parameter>\n\
    <parameter=content>[package]\nname = \"x\"</parameter>\n\
    </function>\n\
    </tool_call>\n\
</IMMEDIATE_TOOL_USE>\n\n\
<IMPORTANT>\n\
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags.\n\
- EVERY required parameter MUST have a non-empty value.\n\
- For 'write'/'Write': ALWAYS provide both file_path AND content. PREFER 'write' over 'edit' — write the complete updated file content instead of find-and-replace. Only use 'edit' for very small single-line changes.\n\
- For 'edit'/'Edit': file_path MUST be non-empty. oldString MUST be copied verbatim from the file.\n\
- To make MULTIPLE tool calls, use SEPARATE <tool_call> blocks — NEVER embed <tool_call> tags inside bash commands or heredocs.\n\
- NEVER simulate a tool response in your content. Do NOT emit <tool_response>, <file>, \"1: ...\\n2: ...\" line-numbered previews, or any other text pretending to show what a tool returned. The real response is provided by the system AFTER you emit the <tool_call>.\n\
- Tool names like 'write', 'read', 'edit', 'bash' are ONLY valid inside the `<function=NAME>` line of a `<tool_call>` block. NEVER emit bare tags like `<write>`, `<filePath>`, or `<command>` at the top level of your content.\n\
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after.\n\
- If there is no function call available, answer the question with your current knowledge and do not tell the user about function calls.\n\
</IMPORTANT>",
        );
        append_tool_choice_instruction(&mut prompt, tool_choice, "a <tool_call> block");
        prompt
    }

    fn format_tool_calls(&self, calls: &[IncomingToolCall]) -> String {
        let mut out = String::new();
        for tc in calls {
            let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                .unwrap_or(serde_json::Value::Object(Default::default()));
            out.push_str("\n<tool_call>\n");
            out.push_str(&format!("<function={}>\n", tc.function.name));
            if let Some(obj) = args.as_object() {
                for (key, val) in obj {
                    let val_str = match val {
                        serde_json::Value::String(s) => s.clone(),
                        other => serde_json::to_string(other).unwrap_or_default(),
                    };
                    out.push_str(&format!("<parameter={key}>\n{val_str}\n</parameter>\n"));
                }
            }
            out.push_str("</function>\n</tool_call>");
        }
        out
    }

    fn leak_markers(&self) -> LeakMarkers {
        // `<tool_response>` is a server-internal wrapper rendered around
        // role=tool messages when they enter the prompt (see the
        // qwen3_5_moe.jinja chat template). The model should never emit it in
        // content — when it does, it is simulating a tool exchange.
        const MARKERS: LeakMarkers = LeakMarkers {
            // Suppress malformed or orphan tool-call/result XML fragments that
            // land in streamed content until a matching close tag arrives. The
            // streaming detector consumes complete `<tool_call>...</tool_call>`
            // blocks before this sanitizer runs, so only orphan/partial
            // fragments reach Content.
            //
            // `<function_results>` (Anthropic-style tool-result wrapper) is
            // listed as an opener and `</function_results>` / `</result>` as
            // close tags; bare `<result>` is intentionally omitted from
            // `orphan_open` so ordinary XML prose containing `<result>` is not
            // suppressed.
            orphan_open: &[
                "<parameter=",
                "<tool_response>",
                "<function_results>",
                "<function=",
                "<tool_call>",
                "<tool_use>",
                "<param=",
            ],
            close: &[
                "</parameter>",
                "</function>",
                "</tool_call>",
                "</tool_response>",
                "</function_results>",
                "</result>",
                "</tool_use>",
            ],
            envelope_open: &[],
            envelope_close: &[],
        };
        MARKERS
    }
}

// ── Gemma-4 parser (native format) ──
