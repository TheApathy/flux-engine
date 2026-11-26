// SPDX-License-Identifier: AGPL-3.0-only
//
// libFuzzer target exercising Flux Engine's pure-Rust `xgrammar` JSON Schema
// and EBNF grammar compiler against untrusted external payloads.

#![no_main]

use libfuzzer_sys::fuzz_target;
use xgrammar::{Grammar, SchemaConverterOptions, json_schema_to_grammar};

fuzz_target!(|data: &[u8]| {
    // Bound input size to match HTTP structured-output guardrails.
    if data.len() > 16 * 1024 {
        return;
    }
    if let Ok(text) = std::str::from_utf8(data) {
        let _ = json_schema_to_grammar(text, &SchemaConverterOptions::default());
        let _ = Grammar::from_ebnf(text, "root");
    }
});
