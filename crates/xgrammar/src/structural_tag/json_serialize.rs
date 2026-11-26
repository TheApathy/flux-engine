// SPDX-License-Identifier: AGPL-3.0-only
//
// Compact, order-preserving JSON serializer for [`JsonValue`].
//
// WHY THIS EXISTS
// ---------------
// The C++ `StructuralTagParser` stores the embedded `json_schema`
// sub-document by calling picojson's `value.serialize(false)` — a
// compact re-serialization. The schema converter then re-parses that
// string. Object key order must survive the round-trip because JSON
// Schema conversion is order-sensitive (`properties` declaration
// order). `serde_json::Map` is not order-preserving in this crate's
// feature set, so we serialize the crate's own ordered `JsonValue`
// directly here. This mirrors picojson `serialize(false)`.

use crate::schema::JsonValue;

/// Compactly serialize a [`JsonValue`] to a JSON string, preserving
/// object key order. Equivalent to picojson `serialize(false)`.
pub fn serialize_compact(value: &JsonValue) -> String {
    crate::schema::serialize_compact(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::JsonValue;

    fn roundtrip(text: &str) -> String {
        serialize_compact(&JsonValue::parse(text).expect("parse"))
    }

    #[test]
    fn preserves_object_key_order() {
        assert_eq!(roundtrip(r#"{"b":1,"a":2}"#), r#"{"b":1,"a":2}"#);
    }

    #[test]
    fn serializes_scalars() {
        assert_eq!(roundtrip("true"), "true");
        assert_eq!(roundtrip("null"), "null");
        assert_eq!(roundtrip("42"), "42");
        assert_eq!(roundtrip(r#""hi""#), r#""hi""#);
    }

    #[test]
    fn serializes_nested() {
        let s = r#"{"type":"object","properties":{"a":{"type":"string"}}}"#;
        assert_eq!(roundtrip(s), s);
    }

    #[test]
    fn escapes_strings() {
        assert_eq!(roundtrip(r#""a\nb""#), r#""a\nb""#);
    }
}
