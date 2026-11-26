// SPDX-License-Identifier: AGPL-3.0-only

//! Fail-closed JSON Schema 2020-12 pre-validator ahead of structured-output
//! grammar compilation (`response_format: { type: "json_schema", strict: true }`).
//!
//! Rejects with HTTP 400 and an exact RFC 6901 JSON Pointer (`#/properties/...`):
//! - Schemas exceeding maximum nesting depth (`> 32`).
//! - Integer bounds or literals outside IEEE-754 exact safe integers (`±(2^53 - 1)`).
//! - Lone UTF-16 surrogates (`\uD800`..`\uDFFF` unpaired) in raw JSON strings.
//! - Contradictory bounds (`minimum > maximum`, `minLength > maxLength`,
//!   `minItems > maxItems`, `minProperties > maxProperties`, or impossible
//!   exclusive bounds).

use std::fmt;

use serde_json::Value;

/// Maximum permitted schema nesting depth before fail-closed rejection.
pub const MAX_SCHEMA_DEPTH: usize = 32;

/// Maximum exact IEEE-754 double-precision integer (`2^53 - 1`).
pub const MAX_SAFE_INTEGER: i64 = 9_007_199_254_740_991;

/// Minimum exact IEEE-754 double-precision integer (`-(2^53 - 1)`).
pub const MIN_SAFE_INTEGER: i64 = -9_007_199_254_740_991;

/// Category of fail-closed JSON Schema validation failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaErrorKind {
    MaxDepthExceeded,
    UnsafeIntegerBound,
    LoneUtf16Surrogate,
    ContradictoryBounds,
    InvalidSchemaType,
}

/// Structured validation failure carrying the exact RFC 6901 JSON Pointer.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub struct SchemaValidationError {
    pub pointer: String,
    pub kind: SchemaErrorKind,
    pub message: String,
}

impl SchemaValidationError {
    fn new(pointer: impl Into<String>, kind: SchemaErrorKind, message: impl Into<String>) -> Self {
        Self {
            pointer: pointer.into(),
            kind,
            message: message.into(),
        }
    }
}

impl fmt::Display for SchemaValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid json_schema at {}: {}",
            self.pointer, self.message
        )
    }
}

impl std::error::Error for SchemaValidationError {}

fn push_pointer_segment(base: &str, segment: &str) -> String {
    let escaped = segment.replace('~', "~0").replace('/', "~1");
    format!("{base}/{escaped}")
}

/// Validate both raw JSON bytes (catching lone UTF-16 surrogates and
/// out-of-range integer tokens before `f64` rounding) and the parsed schema.
///
/// # Errors
/// Returns [`SchemaValidationError`] with an exact `#/...` JSON Pointer on the
/// first violated rule.
pub fn validate_strict_json_schema_str(raw: &str) -> Result<(), SchemaValidationError> {
    check_raw_schema_bytes(raw.as_bytes())?;
    let parsed: Value = serde_json::from_str(raw).map_err(|e| {
        SchemaValidationError::new("#", SchemaErrorKind::InvalidSchemaType, e.to_string())
    })?;
    validate_strict_json_schema(&parsed)
}

/// Inspect a raw `/v1/chat/completions` JSON payload for lone UTF-16 surrogates
/// or >53-bit integer bounds inside `response_format.json_schema.schema` before
/// `serde_json` either strips context on surrogate errors or rounds integers to `f64`.
///
/// Returns `Ok(())` for non-JSON-schema requests, `strict: false` requests, or
/// malformed outer JSON (leaving the standard body parser to report syntax errors).
pub fn precheck_raw_chat_body(body: &[u8]) -> Result<(), SchemaValidationError> {
    let Some(span) = locate_strict_schema_slice(body) else {
        return Ok(());
    };
    check_raw_schema_bytes(span)
}

/// Validate a parsed JSON Schema 2020-12 value under `strict: true`.
///
/// # Errors
/// Returns [`SchemaValidationError`] naming the exact `#/...` JSON Pointer.
pub fn validate_strict_json_schema(schema: &Value) -> Result<(), SchemaValidationError> {
    match schema {
        Value::Object(_) | Value::Bool(true) => validate_schema_node(schema, "#", 0),
        Value::Bool(false) => Err(SchemaValidationError::new(
            "#",
            SchemaErrorKind::InvalidSchemaType,
            "root schema cannot be `false` under strict structured output",
        )),
        _ => Err(SchemaValidationError::new(
            "#",
            SchemaErrorKind::InvalidSchemaType,
            "root schema must be a JSON object",
        )),
    }
}

fn validate_schema_node(
    node: &Value,
    pointer: &str,
    depth: usize,
) -> Result<(), SchemaValidationError> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err(SchemaValidationError::new(
            pointer,
            SchemaErrorKind::MaxDepthExceeded,
            format!("schema nesting depth {depth} exceeds maximum of {MAX_SCHEMA_DEPTH}"),
        ));
    }
    let Some(obj) = node.as_object() else {
        return Ok(());
    };

    let is_integer_type = schema_declares_integer(obj.get("type"));
    validate_type_keyword(obj, pointer)?;
    validate_numeric_bounds(obj, pointer, is_integer_type)?;
    validate_length_and_count_bounds(obj, pointer)?;

    for (key, val) in obj {
        let key_ptr = push_pointer_segment(pointer, key);
        match key.as_str() {
            "properties" | "patternProperties" | "$defs" | "definitions" | "dependentSchemas" => {
                if let Some(map) = val.as_object() {
                    for (prop_name, subschema) in map {
                        let prop_ptr = push_pointer_segment(&key_ptr, prop_name);
                        validate_schema_node(subschema, &prop_ptr, depth + 1)?;
                    }
                }
            }
            "items"
            | "additionalProperties"
            | "unevaluatedItems"
            | "unevaluatedProperties"
            | "not"
            | "if"
            | "then"
            | "else"
            | "contains"
            | "propertyNames" => match val {
                Value::Object(_) | Value::Bool(_) => {
                    validate_schema_node(val, &key_ptr, depth + 1)?;
                }
                Value::Array(arr) if key == "items" => {
                    for (idx, item) in arr.iter().enumerate() {
                        let item_ptr = push_pointer_segment(&key_ptr, &idx.to_string());
                        validate_schema_node(item, &item_ptr, depth + 1)?;
                    }
                }
                _ => {}
            },
            "prefixItems" | "anyOf" | "oneOf" | "allOf" => {
                if let Some(arr) = val.as_array() {
                    for (idx, subschema) in arr.iter().enumerate() {
                        let sub_ptr = push_pointer_segment(&key_ptr, &idx.to_string());
                        validate_schema_node(subschema, &sub_ptr, depth + 1)?;
                    }
                }
            }
            "enum" | "const" | "default" => {
                validate_arbitrary_json_value(val, &key_ptr, depth + 1, is_integer_type)?;
            }
            _ => {}
        }
    }

    Ok(())
}

fn is_valid_json_schema_primitive_type(s: &str) -> bool {
    matches!(
        s,
        "null" | "boolean" | "object" | "array" | "number" | "string" | "integer"
    )
}

fn validate_type_keyword(
    obj: &serde_json::Map<String, Value>,
    pointer: &str,
) -> Result<(), SchemaValidationError> {
    let Some(type_val) = obj.get("type") else {
        return Ok(());
    };
    let type_ptr = push_pointer_segment(pointer, "type");
    match type_val {
        Value::String(s) => {
            if !is_valid_json_schema_primitive_type(s) {
                return Err(SchemaValidationError::new(
                    type_ptr,
                    SchemaErrorKind::InvalidSchemaType,
                    format!(
                        "unknown JSON Schema type {s:?}; expected one of null, boolean, object, array, number, string, integer"
                    ),
                ));
            }
            Ok(())
        }
        Value::Array(arr) => {
            if arr.is_empty() {
                return Err(SchemaValidationError::new(
                    type_ptr,
                    SchemaErrorKind::InvalidSchemaType,
                    "`type` array must not be empty",
                ));
            }
            let mut seen: Vec<&str> = Vec::with_capacity(arr.len());
            for (idx, item) in arr.iter().enumerate() {
                let elem_ptr = push_pointer_segment(&type_ptr, &idx.to_string());
                let Some(s) = item.as_str() else {
                    return Err(SchemaValidationError::new(
                        elem_ptr,
                        SchemaErrorKind::InvalidSchemaType,
                        "`type` array elements must be strings",
                    ));
                };
                if !is_valid_json_schema_primitive_type(s) {
                    return Err(SchemaValidationError::new(
                        elem_ptr,
                        SchemaErrorKind::InvalidSchemaType,
                        format!(
                            "unknown JSON Schema type {s:?}; expected one of null, boolean, object, array, number, string, integer"
                        ),
                    ));
                }
                if seen.contains(&s) {
                    return Err(SchemaValidationError::new(
                        elem_ptr,
                        SchemaErrorKind::InvalidSchemaType,
                        format!("duplicate JSON Schema type {s:?} in `type` union"),
                    ));
                }
                seen.push(s);
            }
            Ok(())
        }
        _ => Err(SchemaValidationError::new(
            type_ptr,
            SchemaErrorKind::InvalidSchemaType,
            "`type` keyword must be a string or non-empty array of strings",
        )),
    }
}

fn schema_declares_integer(type_val: Option<&Value>) -> bool {
    match type_val {
        Some(Value::String(s)) => s == "integer",
        Some(Value::Array(arr)) => arr.iter().any(|v| v.as_str() == Some("integer")),
        _ => false,
    }
}

fn check_safe_integer_number(
    num: &serde_json::Number,
    pointer: &str,
    require_integer: bool,
) -> Result<(), SchemaValidationError> {
    if let Some(i) = num.as_i64() {
        if !(MIN_SAFE_INTEGER..=MAX_SAFE_INTEGER).contains(&i) {
            return Err(SchemaValidationError::new(
                pointer,
                SchemaErrorKind::UnsafeIntegerBound,
                format!(
                    "integer bound {i} exceeds IEEE-754 exact safe range [{MIN_SAFE_INTEGER}, {MAX_SAFE_INTEGER}]"
                ),
            ));
        }
        return Ok(());
    }
    if let Some(u) = num.as_u64() {
        if u > MAX_SAFE_INTEGER as u64 {
            return Err(SchemaValidationError::new(
                pointer,
                SchemaErrorKind::UnsafeIntegerBound,
                format!(
                    "integer bound {u} exceeds IEEE-754 exact safe range [{MIN_SAFE_INTEGER}, {MAX_SAFE_INTEGER}]"
                ),
            ));
        }
        return Ok(());
    }
    if let Some(f) = num.as_f64() {
        if !f.is_finite() {
            return Err(SchemaValidationError::new(
                pointer,
                SchemaErrorKind::UnsafeIntegerBound,
                "numeric schema bound must be finite",
            ));
        }
        if require_integer {
            if f.fract() != 0.0 {
                return Err(SchemaValidationError::new(
                    pointer,
                    SchemaErrorKind::UnsafeIntegerBound,
                    format!("integer schema bound {f} must be an exact integer"),
                ));
            }
            if !(MIN_SAFE_INTEGER as f64..=MAX_SAFE_INTEGER as f64).contains(&f) {
                return Err(SchemaValidationError::new(
                    pointer,
                    SchemaErrorKind::UnsafeIntegerBound,
                    format!(
                        "integer bound {f} exceeds IEEE-754 exact safe range [{MIN_SAFE_INTEGER}, {MAX_SAFE_INTEGER}]"
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn extract_numeric_kw(
    obj: &serde_json::Map<String, Value>,
    key: &str,
    parent_ptr: &str,
    is_integer_type: bool,
) -> Result<Option<(f64, String)>, SchemaValidationError> {
    let Some(val) = obj.get(key) else {
        return Ok(None);
    };
    let kw_ptr = push_pointer_segment(parent_ptr, key);
    let Some(num) = val.as_number() else {
        return Err(SchemaValidationError::new(
            kw_ptr,
            SchemaErrorKind::InvalidSchemaType,
            format!("`{key}` must be a number"),
        ));
    };
    check_safe_integer_number(num, &kw_ptr, is_integer_type)?;
    let Some(as_f64) = num.as_f64().filter(|v| v.is_finite()) else {
        return Err(SchemaValidationError::new(
            kw_ptr,
            SchemaErrorKind::UnsafeIntegerBound,
            format!("`{key}` must be a finite number"),
        ));
    };
    Ok(Some((as_f64, kw_ptr)))
}

fn validate_numeric_bounds(
    obj: &serde_json::Map<String, Value>,
    pointer: &str,
    is_integer_type: bool,
) -> Result<(), SchemaValidationError> {
    let min = extract_numeric_kw(obj, "minimum", pointer, is_integer_type)?;
    let max = extract_numeric_kw(obj, "maximum", pointer, is_integer_type)?;
    let excl_min = extract_numeric_kw(obj, "exclusiveMinimum", pointer, is_integer_type)?;
    let excl_max = extract_numeric_kw(obj, "exclusiveMaximum", pointer, is_integer_type)?;
    if let Some((mult, mult_ptr)) = extract_numeric_kw(obj, "multipleOf", pointer, is_integer_type)?
        && mult <= 0.0
    {
        return Err(SchemaValidationError::new(
            mult_ptr,
            SchemaErrorKind::ContradictoryBounds,
            format!("`multipleOf` must be strictly positive, got {mult}"),
        ));
    }

    if let (Some((min_v, min_ptr)), Some((max_v, _))) = (&min, &max)
        && min_v > max_v
    {
        return Err(SchemaValidationError::new(
            min_ptr.clone(),
            SchemaErrorKind::ContradictoryBounds,
            format!("contradictory bounds: minimum ({min_v}) > maximum ({max_v})"),
        ));
    }
    if let (Some((emin_v, emin_ptr)), Some((max_v, _))) = (&excl_min, &max)
        && emin_v >= max_v
    {
        return Err(SchemaValidationError::new(
            emin_ptr.clone(),
            SchemaErrorKind::ContradictoryBounds,
            format!("contradictory bounds: exclusiveMinimum ({emin_v}) >= maximum ({max_v})"),
        ));
    }
    if let (Some((min_v, min_ptr)), Some((emax_v, _))) = (&min, &excl_max)
        && min_v >= emax_v
    {
        return Err(SchemaValidationError::new(
            min_ptr.clone(),
            SchemaErrorKind::ContradictoryBounds,
            format!("contradictory bounds: minimum ({min_v}) >= exclusiveMaximum ({emax_v})"),
        ));
    }
    if let (Some((emin_v, emin_ptr)), Some((emax_v, _))) = (&excl_min, &excl_max)
        && emin_v >= emax_v
    {
        return Err(SchemaValidationError::new(
            emin_ptr.clone(),
            SchemaErrorKind::ContradictoryBounds,
            format!(
                "contradictory bounds: exclusiveMinimum ({emin_v}) >= exclusiveMaximum ({emax_v})"
            ),
        ));
    }
    Ok(())
}

fn extract_nonneg_int_kw(
    obj: &serde_json::Map<String, Value>,
    key: &str,
    parent_ptr: &str,
) -> Result<Option<(u64, String)>, SchemaValidationError> {
    let Some(val) = obj.get(key) else {
        return Ok(None);
    };
    let kw_ptr = push_pointer_segment(parent_ptr, key);
    let Some(num) = val.as_number() else {
        return Err(SchemaValidationError::new(
            kw_ptr,
            SchemaErrorKind::InvalidSchemaType,
            format!("`{key}` must be a non-negative integer"),
        ));
    };
    check_safe_integer_number(num, &kw_ptr, true)?;
    let Some(u) = num.as_u64() else {
        return Err(SchemaValidationError::new(
            kw_ptr,
            SchemaErrorKind::ContradictoryBounds,
            format!("`{key}` must be >= 0, got {num}"),
        ));
    };
    Ok(Some((u, kw_ptr)))
}

fn validate_length_and_count_bounds(
    obj: &serde_json::Map<String, Value>,
    pointer: &str,
) -> Result<(), SchemaValidationError> {
    for (lo_key, hi_key) in [
        ("minLength", "maxLength"),
        ("minItems", "maxItems"),
        ("minProperties", "maxProperties"),
    ] {
        let lo = extract_nonneg_int_kw(obj, lo_key, pointer)?;
        let hi = extract_nonneg_int_kw(obj, hi_key, pointer)?;
        if let (Some((lo_v, lo_ptr)), Some((hi_v, _))) = (lo, hi)
            && lo_v > hi_v
        {
            return Err(SchemaValidationError::new(
                lo_ptr,
                SchemaErrorKind::ContradictoryBounds,
                format!("contradictory bounds: {lo_key} ({lo_v}) > {hi_key} ({hi_v})"),
            ));
        }
    }
    Ok(())
}

fn validate_arbitrary_json_value(
    val: &Value,
    pointer: &str,
    depth: usize,
    require_safe_int: bool,
) -> Result<(), SchemaValidationError> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err(SchemaValidationError::new(
            pointer,
            SchemaErrorKind::MaxDepthExceeded,
            format!("value nesting depth {depth} exceeds maximum of {MAX_SCHEMA_DEPTH}"),
        ));
    }
    match val {
        Value::Number(n) => check_safe_integer_number(n, pointer, require_safe_int),
        Value::Array(arr) => {
            for (idx, item) in arr.iter().enumerate() {
                let item_ptr = push_pointer_segment(pointer, &idx.to_string());
                validate_arbitrary_json_value(item, &item_ptr, depth + 1, require_safe_int)?;
            }
            Ok(())
        }
        Value::Object(map) => {
            for (k, item) in map {
                let item_ptr = push_pointer_segment(pointer, k);
                validate_arbitrary_json_value(item, &item_ptr, depth + 1, require_safe_int)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

// ── Raw JSON byte scanner for lone UTF-16 surrogates & >53-bit integer literals ──

/// Maximum raw-byte JSON nesting depth accepted before `serde_json` runs.
const MAX_RAW_SCAN_DEPTH: usize = MAX_SCHEMA_DEPTH * 4;

/// Scan raw JSON bytes of a schema document for unpaired UTF-16 surrogates
/// (`\uD800`..`\uDFFF`) and out-of-range integer literals (`> 2^53 - 1` or
/// `< -(2^53 - 1)`), reporting the exact `#/...` JSON Pointer.
pub fn check_raw_schema_bytes(bytes: &[u8]) -> Result<(), SchemaValidationError> {
    let mut scanner = RawJsonScanner { bytes, pos: 0 };
    scanner.skip_ws();
    if scanner.pos < scanner.bytes.len() {
        scanner.scan_value("#", 0)?;
    }
    Ok(())
}

struct RawJsonScanner<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> RawJsonScanner<'a> {
    fn skip_ws(&mut self) {
        while let Some(&b) = self.bytes.get(self.pos) {
            if matches!(b, b' ' | b'\t' | b'\r' | b'\n') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn scan_value(&mut self, pointer: &str, depth: usize) -> Result<(), SchemaValidationError> {
        if depth > MAX_RAW_SCAN_DEPTH {
            return Err(SchemaValidationError::new(
                pointer,
                SchemaErrorKind::MaxDepthExceeded,
                format!("raw JSON nesting depth {depth} exceeds maximum {MAX_RAW_SCAN_DEPTH}"),
            ));
        }
        self.skip_ws();
        let Some(&b) = self.bytes.get(self.pos) else {
            return Ok(());
        };
        match b {
            b'"' => {
                let _ = self.scan_string(pointer)?;
                Ok(())
            }
            b'{' => self.scan_object(pointer, depth + 1),
            b'[' => self.scan_array(pointer, depth + 1),
            b'-' | b'0'..=b'9' => self.scan_number(pointer),
            b't' => {
                self.advance_literal(b"true");
                Ok(())
            }
            b'f' => {
                self.advance_literal(b"false");
                Ok(())
            }
            b'n' => {
                self.advance_literal(b"null");
                Ok(())
            }
            _ => {
                self.pos += 1;
                Ok(())
            }
        }
    }

    fn advance_literal(&mut self, lit: &[u8]) {
        if self.bytes[self.pos..].starts_with(lit) {
            self.pos += lit.len();
        } else {
            self.pos += 1;
        }
    }

    fn scan_object(&mut self, pointer: &str, depth: usize) -> Result<(), SchemaValidationError> {
        self.pos += 1; // '{'
        loop {
            self.skip_ws();
            match self.bytes.get(self.pos) {
                None => return Ok(()),
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(());
                }
                Some(b',') => {
                    self.pos += 1;
                    continue;
                }
                Some(b'"') => {
                    let key = self.scan_string(pointer)?;
                    let child_ptr = push_pointer_segment(pointer, &key);
                    self.skip_ws();
                    if self.bytes.get(self.pos) == Some(&b':') {
                        self.pos += 1;
                    }
                    self.scan_value(&child_ptr, depth)?;
                }
                Some(_) => {
                    self.pos += 1;
                }
            }
        }
    }

    fn scan_array(&mut self, pointer: &str, depth: usize) -> Result<(), SchemaValidationError> {
        self.pos += 1; // '['
        let mut idx = 0usize;
        loop {
            self.skip_ws();
            match self.bytes.get(self.pos) {
                None => return Ok(()),
                Some(b']') => {
                    self.pos += 1;
                    return Ok(());
                }
                Some(b',') => {
                    self.pos += 1;
                    continue;
                }
                Some(_) => {
                    let elem_ptr = push_pointer_segment(pointer, &idx.to_string());
                    self.scan_value(&elem_ptr, depth)?;
                    idx += 1;
                }
            }
        }
    }

    fn scan_string(&mut self, pointer: &str) -> Result<String, SchemaValidationError> {
        self.pos += 1; // opening '"'
        let mut decoded = String::new();
        let mut raw_bytes: Vec<u8> = Vec::new();
        let flush_raw = |raw: &mut Vec<u8>, out: &mut String| {
            if !raw.is_empty() {
                out.push_str(&String::from_utf8_lossy(raw));
                raw.clear();
            }
        };
        while let Some(&b) = self.bytes.get(self.pos) {
            self.pos += 1;
            if b == b'"' {
                flush_raw(&mut raw_bytes, &mut decoded);
                break;
            }
            if b != b'\\' {
                raw_bytes.push(b);
                continue;
            }
            flush_raw(&mut raw_bytes, &mut decoded);
            let Some(&esc) = self.bytes.get(self.pos) else {
                break;
            };
            self.pos += 1;
            match esc {
                b'"' => decoded.push('"'),
                b'\\' => decoded.push('\\'),
                b'/' => decoded.push('/'),
                b'b' => decoded.push('\u{0008}'),
                b'f' => decoded.push('\u{000C}'),
                b'n' => decoded.push('\n'),
                b'r' => decoded.push('\r'),
                b't' => decoded.push('\t'),
                b'u' => {
                    let Some(code) = self.read_hex4() else {
                        break;
                    };
                    if (0xD800..=0xDBFF).contains(&code) {
                        // High surrogate must be immediately followed by `\uDC00`..`\uDFFF`.
                        if self.bytes.get(self.pos..self.pos + 2) != Some(b"\\u") {
                            return Err(SchemaValidationError::new(
                                pointer,
                                SchemaErrorKind::LoneUtf16Surrogate,
                                format!(
                                    "lone high UTF-16 surrogate \\u{code:04X} without trailing low surrogate"
                                ),
                            ));
                        }
                        self.pos += 2;
                        let Some(low) = self.read_hex4() else {
                            return Err(SchemaValidationError::new(
                                pointer,
                                SchemaErrorKind::LoneUtf16Surrogate,
                                format!(
                                    "lone high UTF-16 surrogate \\u{code:04X} followed by malformed escape"
                                ),
                            ));
                        };
                        if !(0xDC00..=0xDFFF).contains(&low) {
                            return Err(SchemaValidationError::new(
                                pointer,
                                SchemaErrorKind::LoneUtf16Surrogate,
                                format!(
                                    "lone high UTF-16 surrogate \\u{code:04X} paired with non-low \\u{low:04X}"
                                ),
                            ));
                        }
                        let scalar =
                            0x10000 + (((code as u32 - 0xD800) << 10) | (low as u32 - 0xDC00));
                        if let Some(ch) = char::from_u32(scalar) {
                            decoded.push(ch);
                        }
                    } else if (0xDC00..=0xDFFF).contains(&code) {
                        return Err(SchemaValidationError::new(
                            pointer,
                            SchemaErrorKind::LoneUtf16Surrogate,
                            format!(
                                "lone low UTF-16 surrogate \\u{code:04X} without leading high surrogate"
                            ),
                        ));
                    } else if let Some(ch) = char::from_u32(code as u32) {
                        decoded.push(ch);
                    }
                }
                other => raw_bytes.push(other),
            }
        }
        flush_raw(&mut raw_bytes, &mut decoded);
        Ok(decoded)
    }

    fn read_hex4(&mut self) -> Option<u16> {
        let slice = self.bytes.get(self.pos..self.pos + 4)?;
        let s = std::str::from_utf8(slice).ok()?;
        let val = u16::from_str_radix(s, 16).ok()?;
        self.pos += 4;
        Some(val)
    }

    fn scan_number(&mut self, pointer: &str) -> Result<(), SchemaValidationError> {
        let start = self.pos;
        if self.bytes.get(self.pos) == Some(&b'-') {
            self.pos += 1;
        }
        while matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
        let mut is_float = false;
        if self.bytes.get(self.pos) == Some(&b'.') {
            is_float = true;
            self.pos += 1;
            while matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if matches!(self.bytes.get(self.pos), Some(b'e' | b'E')) {
            is_float = true;
            self.pos += 1;
            if matches!(self.bytes.get(self.pos), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            while matches!(self.bytes.get(self.pos), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if !is_float
            && is_integer_bound_pointer(pointer)
            && let Ok(token) = std::str::from_utf8(&self.bytes[start..self.pos])
        {
            let valid = token
                .parse::<i128>()
                .is_ok_and(|v| (MIN_SAFE_INTEGER as i128..=MAX_SAFE_INTEGER as i128).contains(&v));
            if !valid {
                return Err(SchemaValidationError::new(
                    pointer,
                    SchemaErrorKind::UnsafeIntegerBound,
                    format!(
                        "integer literal {token} exceeds IEEE-754 exact safe range [{MIN_SAFE_INTEGER}, {MAX_SAFE_INTEGER}]"
                    ),
                ));
            }
        }
        Ok(())
    }
}

fn is_integer_bound_pointer(pointer: &str) -> bool {
    let last = pointer.rsplit('/').next().unwrap_or("");
    matches!(
        last,
        "minimum"
            | "maximum"
            | "exclusiveMinimum"
            | "exclusiveMaximum"
            | "multipleOf"
            | "minLength"
            | "maxLength"
            | "minItems"
            | "maxItems"
            | "minProperties"
            | "maxProperties"
            | "const"
    )
}

/// Locate the raw byte slice of `response_format.json_schema.schema` when
/// `response_format.type == "json_schema"` and `strict` is not `false`.
fn locate_strict_schema_slice(body: &[u8]) -> Option<&[u8]> {
    let mut s = RawSliceFinder {
        bytes: body,
        pos: 0,
    };
    s.skip_ws();
    if s.bytes.get(s.pos) != Some(&b'{') {
        return None;
    }
    let rf_slice = s.find_object_field_slice("response_format")?;
    let mut rf = RawSliceFinder {
        bytes: rf_slice,
        pos: 0,
    };
    let rf_type = rf.find_object_string_field("type")?;
    if rf_type != "json_schema" {
        return None;
    }
    rf.pos = 0;
    let js_slice = rf.find_object_field_slice("json_schema")?;
    let mut js = RawSliceFinder {
        bytes: js_slice,
        pos: 0,
    };
    if js.find_object_raw_field("strict") == Some(b"false") {
        return None;
    }
    js.pos = 0;
    js.find_object_field_slice("schema")
}

struct RawSliceFinder<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> RawSliceFinder<'a> {
    fn skip_ws(&mut self) {
        while matches!(self.bytes.get(self.pos), Some(b' ' | b'\t' | b'\r' | b'\n')) {
            self.pos += 1;
        }
    }

    fn skip_string(&mut self) -> Option<String> {
        if self.bytes.get(self.pos) != Some(&b'"') {
            return None;
        }
        self.pos += 1;
        let mut out: Vec<u8> = Vec::new();
        while let Some(&b) = self.bytes.get(self.pos) {
            self.pos += 1;
            if b == b'"' {
                return Some(String::from_utf8_lossy(&out).into_owned());
            }
            if b == b'\\' {
                if let Some(&esc) = self.bytes.get(self.pos) {
                    self.pos += 1;
                    out.push(esc);
                }
            } else {
                out.push(b);
            }
        }
        None
    }

    fn skip_any_value(&mut self) -> Option<&'a [u8]> {
        self.skip_ws();
        let start = self.pos;
        let &b = self.bytes.get(self.pos)?;
        match b {
            b'"' => {
                self.skip_string()?;
            }
            b'{' | b'[' => {
                let (open, close) = if b == b'{' {
                    (b'{', b'}')
                } else {
                    (b'[', b']')
                };
                let mut depth = 0usize;
                while let Some(&ch) = self.bytes.get(self.pos) {
                    if ch == b'"' {
                        self.skip_string()?;
                        continue;
                    }
                    self.pos += 1;
                    if ch == open {
                        depth += 1;
                    } else if ch == close {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                }
                if depth != 0 {
                    return None;
                }
            }
            _ => {
                while let Some(&ch) = self.bytes.get(self.pos) {
                    if matches!(ch, b',' | b'}' | b']' | b' ' | b'\t' | b'\r' | b'\n') {
                        break;
                    }
                    self.pos += 1;
                }
            }
        }
        Some(&self.bytes[start..self.pos])
    }

    fn find_object_field_slice(&mut self, target_key: &str) -> Option<&'a [u8]> {
        self.skip_ws();
        if self.bytes.get(self.pos) != Some(&b'{') {
            return None;
        }
        self.pos += 1;
        loop {
            self.skip_ws();
            match self.bytes.get(self.pos)? {
                b'}' => return None,
                b',' => {
                    self.pos += 1;
                    continue;
                }
                b'"' => {
                    let key = self.skip_string()?;
                    self.skip_ws();
                    if self.bytes.get(self.pos) != Some(&b':') {
                        return None;
                    }
                    self.pos += 1;
                    let slice = self.skip_any_value()?;
                    if key == target_key {
                        return Some(slice);
                    }
                }
                _ => return None,
            }
        }
    }

    fn find_object_raw_field(&mut self, target_key: &str) -> Option<&'a [u8]> {
        self.find_object_field_slice(target_key)
    }

    fn find_object_string_field(&mut self, target_key: &str) -> Option<String> {
        let raw = self.find_object_field_slice(target_key)?;
        let mut sub = RawSliceFinder { bytes: raw, pos: 0 };
        sub.skip_ws();
        sub.skip_string()
    }
}

#[cfg(test)]
#[path = "schema_check_tests.rs"]
mod tests;
