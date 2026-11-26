// SPDX-License-Identifier: AGPL-3.0-only

// ── Schema enforcement ─────────────────────────────────────────────────

/// CRANE / TAFC schema augmentation.
///
/// Add an optional `_think` field of type string at the head of the
/// tool's `properties`. Per Think-Augmented Function Calling
/// (arXiv:2601.18282) and CRANE (ICML 2025), giving the model a
/// scratchpad slot inside the structured tool-call envelope lets it
/// reason about argument selection without breaking grammar
/// constraints — +12 pp Pass-Rate on Qwen2.5-72B (TAFC).
///
/// `_think` is NOT added to `required` (so the model may skip it);
/// downstream `tool_parser` strips `_think` from parsed args before
/// emitting the ToolCall (the client never sees it).
///
/// Safe no-op when:
///   - the schema isn't a JSON object (`type != "object"`)
///   - `_think` already exists (don't shadow caller's intent)
///   - properties is missing (degenerate schema)
pub fn augment_schema_with_tafc_think(schema: &serde_json::Value) -> serde_json::Value {
    let mut schema = schema.clone();
    let Some(obj) = schema.as_object_mut() else {
        return schema;
    };
    if obj.get("type").and_then(|t| t.as_str()) != Some("object") {
        return schema;
    }
    let Some(props) = obj.get_mut("properties").and_then(|p| p.as_object_mut()) else {
        return schema;
    };
    if props.contains_key("_think") {
        return schema;
    }
    // Reconstruct with `_think` first so the model sees it before the
    // real fields (encourages writing the rationale before answering).
    let mut new_props = serde_json::Map::with_capacity(props.len() + 1);
    new_props.insert(
        "_think".to_string(),
        serde_json::json!({
            "type": "string",
            "description": "Optional scratchpad: brief rationale for selecting this tool and these arguments. Server-side only — not forwarded to the tool implementation.",
        }),
    );
    for (k, v) in props.iter() {
        new_props.insert(k.clone(), v.clone());
    }
    obj.insert(
        "properties".to_string(),
        serde_json::Value::Object(new_props),
    );
    schema
}

/// Add `"minLength": 1` to all required string properties in a tool's
/// JSON schema that don't already specify a `minLength`.
///
/// This causes XGrammar to generate EBNF `{1,}` repetition for these
/// fields, physically preventing the model from emitting `""` as a
/// required parameter value. Solves the "empty parameter" bug where
/// the model generates `"command": ""` etc.
pub(super) fn enforce_min_length_on_required_strings(
    schema: &serde_json::Value,
) -> serde_json::Value {
    let mut schema = schema.clone();
    let obj = match schema.as_object_mut() {
        Some(o) => o,
        None => return schema,
    };

    let required: Vec<String> = obj
        .get("required")
        .and_then(|r| r.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    if required.is_empty() {
        return schema;
    }

    if let Some(props) = obj.get_mut("properties").and_then(|p| p.as_object_mut()) {
        for key in &required {
            if let Some(prop) = props.get_mut(key).and_then(|p| p.as_object_mut()) {
                let is_string = prop.get("type").and_then(|t| t.as_str()) == Some("string");
                if is_string && !prop.contains_key("minLength") {
                    prop.insert("minLength".to_string(), serde_json::Value::Number(1.into()));
                }
            }
        }
    }

    schema
}

// ── Schema sanitization ───────────────────────────────────────────────

/// Sanitize a JSON schema before passing it to `xgrammar`'s EBNF compiler.
///
/// Certain schema patterns (`enum: []`, `anyOf: []`, empty `properties` in XML
/// format, etc.) can yield empty or invalid EBNF rule bodies during JSON-schema
/// conversion and fail compilation.
///
/// This function recursively transforms problematic patterns into safe
/// equivalents. Returns `None` only for `false` boolean schemas (which accept
/// nothing and cannot be meaningfully constrained).
pub(super) fn sanitize_schema_for_grammar(schema: &serde_json::Value) -> Option<serde_json::Value> {
    let mut active_refs = Vec::new();
    sanitize_recursive(schema, schema, 0, &mut active_refs)
}

fn sanitize_recursive(
    schema: &serde_json::Value,
    root: &serde_json::Value,
    depth: usize,
    active_refs: &mut Vec<String>,
) -> Option<serde_json::Value> {
    if depth > 32 {
        return Some(serde_json::json!({}));
    }

    // Boolean schemas: true = any, false = nothing.
    if let Some(b) = schema.as_bool() {
        return if b { Some(serde_json::json!({})) } else { None };
    }

    let obj = schema.as_object()?;
    let mut result = obj.clone();

    // ── $ref resolution (follow acyclic local $refs up to 8 hops; break cycles
    //    so recursive tree schemas terminate in O(N) instead of O(2^32) and let
    //    XGrammar's native `uri_to_rule` compile the recursive EBNF rule) ──
    let saved_refs_len = active_refs.len();
    for _ in 0..8 {
        let Some(ref_str) = result
            .get("$ref")
            .and_then(|v| v.as_str())
            .map(String::from)
        else {
            break;
        };
        if active_refs.iter().any(|r| r == &ref_str) {
            break;
        }
        let Some(resolved) = resolve_local_ref(&ref_str, root) else {
            break;
        };
        active_refs.push(ref_str);
        result.remove("$ref");
        if let Some(resolved_obj) = resolved.as_object() {
            for (k, v) in resolved_obj {
                if !result.contains_key(k) {
                    result.insert(k.clone(), v.clone());
                }
            }
        }
    }

    // ── Empty enum ──
    if let Some(arr) = result.get("enum").and_then(|v| v.as_array())
        && arr.is_empty()
    {
        result.remove("enum");
    }

    // ── Empty / single-element anyOf / oneOf ──
    for key in ["anyOf", "oneOf"] {
        if let Some(arr) = result.get(key).and_then(|v| v.as_array()).cloned() {
            if arr.is_empty() {
                result.remove(key);
            } else if arr.len() == 1 {
                if let Some(inner) = sanitize_recursive(&arr[0], root, depth + 1, active_refs) {
                    result.remove(key);
                    if let Some(inner_obj) = inner.as_object() {
                        for (k, v) in inner_obj {
                            if !result.contains_key(k) {
                                result.insert(k.clone(), v.clone());
                            }
                        }
                    }
                }
            } else {
                let sanitized: Vec<serde_json::Value> = arr
                    .iter()
                    .filter_map(|el| sanitize_recursive(el, root, depth + 1, active_refs))
                    .collect();
                if sanitized.is_empty() {
                    result.remove(key);
                } else {
                    result.insert(key.to_string(), serde_json::Value::Array(sanitized));
                }
            }
        }
    }

    // ── allOf ──
    if let Some(arr) = result.get("allOf").and_then(|v| v.as_array()).cloned() {
        if arr.is_empty() {
            result.remove("allOf");
        } else if arr.len() == 1 {
            if let Some(inner) = sanitize_recursive(&arr[0], root, depth + 1, active_refs) {
                result.remove("allOf");
                if let Some(inner_obj) = inner.as_object() {
                    for (k, v) in inner_obj {
                        if !result.contains_key(k) {
                            result.insert(k.clone(), v.clone());
                        }
                    }
                }
            }
        } else {
            // Naive merge: combine properties and required from all sub-schemas.
            let mut merged_props = serde_json::Map::new();
            let mut merged_required: Vec<serde_json::Value> = Vec::new();
            let mut merged_type: Option<serde_json::Value> = None;
            for sub in &arr {
                if let Some(s) = sanitize_recursive(sub, root, depth + 1, active_refs)
                    && let Some(o) = s.as_object()
                {
                    if let Some(t) = o.get("type") {
                        merged_type.get_or_insert_with(|| t.clone());
                    }
                    if let Some(p) = o.get("properties").and_then(|p| p.as_object()) {
                        for (k, v) in p {
                            merged_props.insert(k.clone(), v.clone());
                        }
                    }
                    if let Some(r) = o.get("required").and_then(|r| r.as_array()) {
                        for item in r {
                            if !merged_required.contains(item) {
                                merged_required.push(item.clone());
                            }
                        }
                    }
                }
            }
            result.remove("allOf");
            if let Some(t) = merged_type {
                result.entry("type").or_insert(t);
            }
            if !merged_props.is_empty() {
                if let Some(existing) = result.get_mut("properties").and_then(|v| v.as_object_mut())
                {
                    for (k, v) in merged_props {
                        existing.entry(k).or_insert(v);
                    }
                } else {
                    result.insert(
                        "properties".to_string(),
                        serde_json::Value::Object(merged_props),
                    );
                }
            }
            if !merged_required.is_empty() {
                if let Some(existing) = result.get_mut("required").and_then(|v| v.as_array_mut()) {
                    for item in merged_required {
                        if !existing.contains(&item) {
                            existing.push(item);
                        }
                    }
                } else {
                    result.insert(
                        "required".to_string(),
                        serde_json::Value::Array(merged_required),
                    );
                }
            }
        }
    }

    // ── Empty object (no properties, no additional) ──
    // In XML format, xgrammar's VisitObject produces an empty EBNF rule body
    // when there are no properties and no additionalProperties/unevaluatedProperties.
    let is_object = result.get("type").and_then(|t| t.as_str()) == Some("object");
    let has_props = result
        .get("properties")
        .and_then(|p| p.as_object())
        .is_some_and(|p| !p.is_empty());
    let has_structural_keys = result.contains_key("patternProperties")
        || result.contains_key("additionalProperties")
        || result.contains_key("unevaluatedProperties")
        || result.contains_key("propertyNames");

    if is_object && !has_props && !has_structural_keys {
        result.insert(
            "additionalProperties".to_string(),
            serde_json::Value::Bool(true),
        );
    }

    // ── Recurse into property schemas ──
    if let Some(props) = result.get("properties").cloned()
        && let Some(props_obj) = props.as_object()
    {
        let mut new_props = serde_json::Map::new();
        for (k, v) in props_obj {
            if let Some(sanitized) = sanitize_recursive(v, root, depth + 1, active_refs) {
                new_props.insert(k.clone(), sanitized);
            } else {
                // Property schema is unsatisfiable — drop it and remove from required.
                if let Some(req) = result.get_mut("required").and_then(|r| r.as_array_mut()) {
                    req.retain(|r| r.as_str() != Some(k.as_str()));
                }
            }
        }
        result.insert(
            "properties".to_string(),
            serde_json::Value::Object(new_props),
        );
    }

    // Recurse into items (array element schema).
    if let Some(items) = result.get("items").cloned()
        && items.is_object()
        && let Some(sanitized) = sanitize_recursive(&items, root, depth + 1, active_refs)
    {
        result.insert("items".to_string(), sanitized);
    }

    // Recurse into additionalProperties when it's a schema object.
    if let Some(addl) = result.get("additionalProperties").cloned()
        && addl.is_object()
        && let Some(sanitized) = sanitize_recursive(&addl, root, depth + 1, active_refs)
    {
        result.insert("additionalProperties".to_string(), sanitized);
    }

    active_refs.truncate(saved_refs_len);
    Some(serde_json::Value::Object(result))
}

/// Resolve a local JSON Pointer `$ref` (e.g. `#/$defs/Foo`).
fn resolve_local_ref(ref_str: &str, root: &serde_json::Value) -> Option<serde_json::Value> {
    let path = ref_str.strip_prefix("#/")?;
    let mut current = root;
    for segment in path.split('/') {
        let decoded = segment.replace("~1", "/").replace("~0", "~");
        current = current.get(&decoded)?;
    }
    Some(current.clone())
}

#[cfg(test)]
mod tests {
    use super::sanitize_schema_for_grammar;

    #[test]
    fn sanitize_follows_chained_local_refs() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "item": {"$ref": "#/$defs/Item"},
                "chained": {"$ref": "#/$defs/Alias"}
            },
            "$defs": {
                "Alias": {"$ref": "#/$defs/Item"},
                "Item": {"type": "string"}
            }
        });
        let result = sanitize_schema_for_grammar(&schema).unwrap();
        assert_eq!(result["properties"]["item"]["type"], "string");
        assert_eq!(result["properties"]["chained"]["type"], "string");
    }

    #[test]
    fn sanitize_all_of_merges_into_existing_outer_properties_and_required() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "id": {"type": "string"}
            },
            "required": ["id"],
            "allOf": [
                {"properties": {"a": {"type": "integer"}}, "required": ["a"]},
                {"properties": {"b": {"type": "boolean"}}, "required": ["b", "id"]}
            ]
        });
        let result = sanitize_schema_for_grammar(&schema).unwrap();
        assert!(result.get("allOf").is_none());
        assert_eq!(result["properties"]["id"]["type"], "string");
        assert_eq!(result["properties"]["a"]["type"], "integer");
        assert_eq!(result["properties"]["b"]["type"], "boolean");
        let req: Vec<&str> = result["required"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        assert_eq!(req, vec!["id", "a", "b"]);
    }

    #[test]
    fn sanitize_breaks_recursive_binary_tree_and_mutual_ref_cycles() {
        // Without active-ref cycle detection, two recursive children explode to
        // O(2^32) calls and hang the request worker.
        let tree = serde_json::json!({
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "val": {"type": "integer"},
                        "left": {"anyOf": [{"$ref": "#/$defs/Node"}, {"type": "null"}]},
                        "right": {"anyOf": [{"$ref": "#/$defs/Node"}, {"type": "null"}]}
                    },
                    "required": ["val"]
                }
            },
            "$ref": "#/$defs/Node"
        });
        let sanitized = sanitize_schema_for_grammar(&tree).unwrap();
        assert_eq!(sanitized["type"], "object");
        assert_eq!(
            sanitized["properties"]["left"]["anyOf"][0]["$ref"],
            "#/$defs/Node"
        );
        assert_eq!(
            sanitized["properties"]["right"]["anyOf"][0]["$ref"],
            "#/$defs/Node"
        );
        assert!(sanitized["$defs"]["Node"].is_object());

        // Direct self-cycle terminates cleanly as well.
        let direct = serde_json::json!({
            "$defs": { "Loop": { "$ref": "#/$defs/Loop" } },
            "$ref": "#/$defs/Loop"
        });
        let direct_out = sanitize_schema_for_grammar(&direct).unwrap();
        assert_eq!(direct_out["$ref"], "#/$defs/Loop");
    }
}
