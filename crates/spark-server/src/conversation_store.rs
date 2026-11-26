// SPDX-License-Identifier: AGPL-3.0-only

//! Conversation store — OpenAI 2026 Conversations API backend.
//!
//! OpenAI's Conversations API replaces the deprecated Assistants
//! `threads` primitive. A Conversation is a durable bag of **items** —
//! most commonly `message` items, plus `function_call` /
//! `function_call_output` items from tool turns. The Responses API
//! gained a `conversation: <conv_id>` field in 2026 that:
//!
//!   1. Prepends the conversation's items onto the new turn's `input`.
//!   2. Appends the new turn's items (user input + assistant output)
//!      back to the conversation after completion.
//!
//! Atlas stores conversations in-memory with an LRU+TTL bound, mirroring
//! the [`crate::response_store`] design but keyed on `conv_<uuid>`.
//! Persistence (filesystem) is a natural follow-up; the public API
//! already hides the backend so that's a drop-in swap.
//!
//! Items are stored as raw `serde_json::Value` so we can round-trip the
//! OpenAI wire format (input_items, function_call, function_call_output,
//! reasoning, …) without coupling to Atlas's internal message schema.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

/// One conversation — a durable, opaque bag of items with per-
/// conversation metadata. `items` preserves insertion order; each item
/// carries an opaque `id` assigned at insert time.
pub struct Conversation {
    pub id: String,
    pub created_at: u64,
    pub metadata: HashMap<String, String>,
    /// Items in wire-format JSON. Each has a top-level `"id"` field.
    pub items: Vec<serde_json::Value>,
    /// Approximate serialized byte footprint of `items` for per-conversation
    /// memory budgeting.
    approx_bytes: usize,
    /// For LRU — bumped on every read + write.
    last_access: Instant,
}

pub struct ConversationStore {
    inner: Mutex<Inner>,
    ttl: Duration,
    max_entries: usize,
    max_items_per_conversation: usize,
    max_bytes_per_conversation: usize,
}

struct Inner {
    map: HashMap<String, Conversation>,
    order: VecDeque<String>,
}

/// Maximum items per conversation insert (mirrors OpenAI's spec).
pub const MAX_ITEMS_PER_INSERT: usize = 20;
/// Maximum total items stored in a single conversation to prevent unbounded
/// per-conversation memory growth.
pub const MAX_ITEMS_PER_CONVERSATION: usize = 2_048;
/// Maximum approximate JSON byte footprint stored in a single conversation
/// (16 MiB default).
pub const MAX_BYTES_PER_CONVERSATION: usize = 16 * 1024 * 1024;

/// Approximate serialized byte footprint of a JSON value without allocating.
fn approx_json_bytes(v: &serde_json::Value) -> usize {
    match v {
        serde_json::Value::Null => 4,
        serde_json::Value::Bool(b) => {
            if *b {
                4
            } else {
                5
            }
        }
        serde_json::Value::Number(_) => 16,
        serde_json::Value::String(s) => s.len().saturating_add(2),
        serde_json::Value::Array(arr) => arr.iter().fold(2usize, |acc, item| {
            acc.saturating_add(approx_json_bytes(item))
                .saturating_add(1)
        }),
        serde_json::Value::Object(map) => map.iter().fold(2usize, |acc, (k, val)| {
            acc.saturating_add(k.len())
                .saturating_add(4)
                .saturating_add(approx_json_bytes(val))
        }),
    }
}

impl ConversationStore {
    pub fn from_env() -> Arc<Self> {
        let max_entries = std::env::var("FLUX_CONVERSATION_MAX_ENTRIES")
            .or_else(|_| std::env::var("ATLAS_CONVERSATION_MAX_ENTRIES"))
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n: &usize| n > 0)
            .unwrap_or(10_000);
        let ttl_secs = std::env::var("FLUX_CONVERSATION_TTL_SECONDS")
            .or_else(|_| std::env::var("ATLAS_CONVERSATION_TTL_SECONDS"))
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(86_400_u64);
        let max_items_per_conversation = std::env::var("FLUX_CONVERSATION_MAX_ITEMS")
            .or_else(|_| std::env::var("ATLAS_CONVERSATION_MAX_ITEMS"))
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n: &usize| n > 0)
            .unwrap_or(MAX_ITEMS_PER_CONVERSATION);
        let max_bytes_per_conversation = std::env::var("FLUX_CONVERSATION_MAX_BYTES")
            .or_else(|_| std::env::var("ATLAS_CONVERSATION_MAX_BYTES"))
            .ok()
            .and_then(|s| s.parse().ok())
            .filter(|&n: &usize| n > 0)
            .unwrap_or(MAX_BYTES_PER_CONVERSATION);
        Arc::new(Self {
            inner: Mutex::new(Inner {
                map: HashMap::new(),
                order: VecDeque::new(),
            }),
            ttl: Duration::from_secs(ttl_secs),
            max_entries,
            max_items_per_conversation,
            max_bytes_per_conversation,
        })
    }

    #[cfg(test)]
    pub fn with_config(max_entries: usize, ttl: Duration) -> Arc<Self> {
        Self::with_limits(
            max_entries,
            ttl,
            MAX_ITEMS_PER_CONVERSATION,
            MAX_BYTES_PER_CONVERSATION,
        )
    }

    #[cfg(test)]
    pub fn with_limits(
        max_entries: usize,
        ttl: Duration,
        max_items_per_conversation: usize,
        max_bytes_per_conversation: usize,
    ) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(Inner {
                map: HashMap::new(),
                order: VecDeque::new(),
            }),
            ttl,
            max_entries,
            max_items_per_conversation,
            max_bytes_per_conversation,
        })
    }

    /// Create a new conversation with optional initial items and
    /// metadata. Returns the generated id.
    pub fn create(
        &self,
        initial_items: Vec<serde_json::Value>,
        metadata: HashMap<String, String>,
    ) -> String {
        self.create_snapshot(initial_items, metadata).id
    }

    /// Create and return the public snapshot from the same insertion
    /// transaction. Callers that must respond with the created object
    /// must not perform a racy second lookup after this lock is released.
    pub fn create_snapshot(
        &self,
        initial_items: Vec<serde_json::Value>,
        metadata: HashMap<String, String>,
    ) -> ConversationSnapshot {
        let id = format!("conv_{}", crate::openai::uuid_v4());
        let now_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let mut approx_bytes = 0usize;
        let items = initial_items
            .into_iter()
            .take(self.max_items_per_conversation)
            .enumerate()
            .map(|(i, v)| stamp_id(v, &id, i))
            .take_while(|v| {
                let sz = approx_json_bytes(v);
                if approx_bytes.saturating_add(sz) <= self.max_bytes_per_conversation {
                    approx_bytes = approx_bytes.saturating_add(sz);
                    true
                } else {
                    false
                }
            })
            .collect::<Vec<_>>();
        let snapshot = ConversationSnapshot {
            id: id.clone(),
            created_at: now_unix,
            metadata: metadata.clone(),
            items: items.clone(),
        };
        let entry = Conversation {
            id: id.clone(),
            created_at: now_unix,
            metadata,
            items,
            approx_bytes,
            last_access: Instant::now(),
        };
        let mut inner = self.inner.lock();
        inner.map.insert(id.clone(), entry);
        inner.order.push_back(id.clone());
        while inner.map.len() > self.max_entries {
            if let Some(oldest) = inner.order.pop_front() {
                inner.map.remove(&oldest);
            } else {
                break;
            }
        }
        snapshot
    }

    /// Fetch a snapshot of the conversation. TTL-expired entries are
    /// evicted and return None.
    pub fn get(&self, id: &str) -> Option<ConversationSnapshot> {
        let mut inner = self.inner.lock();
        let expired = inner
            .map
            .get(id)
            .map(|c| c.last_access.elapsed() > self.ttl)?;
        if expired {
            inner.map.remove(id);
            inner.order.retain(|k| k != id);
            return None;
        }
        bump_order(&mut inner.order, id);
        let c = inner.map.get_mut(id)?;
        c.last_access = Instant::now();
        Some(ConversationSnapshot {
            id: c.id.clone(),
            created_at: c.created_at,
            metadata: c.metadata.clone(),
            items: c.items.clone(),
        })
    }

    /// Merge `patch` into the conversation's metadata. Returns the
    /// updated snapshot; None when the id doesn't exist / is expired.
    pub fn update_metadata(
        &self,
        id: &str,
        patch: HashMap<String, String>,
    ) -> Option<ConversationSnapshot> {
        let mut inner = self.inner.lock();
        let expired = inner
            .map
            .get(id)
            .map(|c| c.last_access.elapsed() > self.ttl)?;
        if expired {
            inner.map.remove(id);
            inner.order.retain(|k| k != id);
            return None;
        }
        bump_order(&mut inner.order, id);
        let c = inner.map.get_mut(id)?;
        c.last_access = Instant::now();
        for (k, v) in patch {
            c.metadata.insert(k, v);
        }
        Some(ConversationSnapshot {
            id: c.id.clone(),
            created_at: c.created_at,
            metadata: c.metadata.clone(),
            items: c.items.clone(),
        })
    }

    /// Append items. Enforces the per-call cap plus per-conversation total
    /// item and byte budgets, returning the newly appended items with assigned ids.
    pub fn add_items(
        &self,
        id: &str,
        new_items: Vec<serde_json::Value>,
    ) -> Result<Vec<serde_json::Value>, AddItemsError> {
        if new_items.len() > MAX_ITEMS_PER_INSERT {
            return Err(AddItemsError::TooMany(new_items.len()));
        }
        let mut inner = self.inner.lock();
        let expired = match inner.map.get(id) {
            Some(c) => c.last_access.elapsed() > self.ttl,
            None => return Err(AddItemsError::NotFound),
        };
        if expired {
            inner.map.remove(id);
            inner.order.retain(|k| k != id);
            return Err(AddItemsError::NotFound);
        }
        let Some(c) = inner.map.get_mut(id) else {
            return Err(AddItemsError::NotFound);
        };
        let start = c.items.len();
        if start.saturating_add(new_items.len()) > self.max_items_per_conversation {
            return Err(AddItemsError::ConversationFull {
                max_items: self.max_items_per_conversation,
                max_bytes: self.max_bytes_per_conversation,
            });
        }
        let stamped: Vec<serde_json::Value> = new_items
            .into_iter()
            .enumerate()
            .map(|(i, v)| stamp_id(v, id, start + i))
            .collect();
        let added_bytes = stamped
            .iter()
            .fold(0usize, |acc, v| acc.saturating_add(approx_json_bytes(v)));
        if c.approx_bytes.saturating_add(added_bytes) > self.max_bytes_per_conversation {
            return Err(AddItemsError::ConversationFull {
                max_items: self.max_items_per_conversation,
                max_bytes: self.max_bytes_per_conversation,
            });
        }
        c.items.extend(stamped.iter().cloned());
        c.approx_bytes = c.approx_bytes.saturating_add(added_bytes);
        c.last_access = Instant::now();
        bump_order(&mut inner.order, id);
        Ok(stamped)
    }

    /// Remove a single item by id. Returns true when an item with that
    /// id was present in a non-expired conversation.
    pub fn remove_item(&self, conv_id: &str, item_id: &str) -> bool {
        let mut inner = self.inner.lock();
        let Some(expired) = inner
            .map
            .get(conv_id)
            .map(|c| c.last_access.elapsed() > self.ttl)
        else {
            return false;
        };
        if expired {
            inner.map.remove(conv_id);
            inner.order.retain(|k| k != conv_id);
            return false;
        }
        let Some(c) = inner.map.get_mut(conv_id) else {
            return false;
        };
        let before = c.items.len();
        c.items
            .retain(|v| v.get("id").and_then(|v| v.as_str()).unwrap_or("") != item_id);
        let removed = c.items.len() < before;
        if removed {
            c.approx_bytes = c
                .items
                .iter()
                .fold(0usize, |acc, v| acc.saturating_add(approx_json_bytes(v)));
            c.last_access = Instant::now();
            bump_order(&mut inner.order, conv_id);
        }
        removed
    }

    pub fn delete(&self, id: &str) -> bool {
        let mut inner = self.inner.lock();
        if inner.map.remove(id).is_some() {
            inner.order.retain(|k| k != id);
            true
        } else {
            false
        }
    }

    pub fn len(&self) -> usize {
        self.inner.lock().map.len()
    }
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }
    pub fn ttl(&self) -> Duration {
        self.ttl
    }
}

pub struct ConversationSnapshot {
    pub id: String,
    pub created_at: u64,
    pub metadata: HashMap<String, String>,
    pub items: Vec<serde_json::Value>,
}

#[derive(Debug)]
pub enum AddItemsError {
    NotFound,
    TooMany(usize),
    ConversationFull { max_items: usize, max_bytes: usize },
}

/// Move `id` to the back of the LRU order queue.
fn bump_order(order: &mut VecDeque<String>, id: &str) {
    order.retain(|k| k != id);
    order.push_back(id.to_string());
}

/// Stamp an `id` onto an item if it doesn't already have one.
/// Preserves any client-supplied `id`; otherwise mints a deterministic
/// `item_<conv>_<idx>`.
fn stamp_id(mut v: serde_json::Value, conv_id: &str, idx: usize) -> serde_json::Value {
    if let Some(obj) = v.as_object_mut()
        && !obj.contains_key("id")
    {
        obj.insert(
            "id".to_string(),
            serde_json::Value::String(format!(
                "item_{}_{}",
                conv_id.trim_start_matches("conv_"),
                idx
            )),
        );
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn create_and_retrieve() {
        let store = ConversationStore::with_config(16, Duration::from_secs(60));
        let id = store.create(
            vec![json!({"type": "message", "role": "user", "content": "hi"})],
            HashMap::new(),
        );
        let snap = store.get(&id).expect("hit");
        assert_eq!(snap.items.len(), 1);
        assert_eq!(snap.items[0]["role"], "user");
        assert!(snap.items[0]["id"].as_str().unwrap().starts_with("item_"));
    }

    #[test]
    fn create_snapshot_does_not_depend_on_a_second_lookup() {
        let store = ConversationStore::with_config(0, Duration::from_secs(60));
        let snapshot = store.create_snapshot(
            vec![json!({"type": "message", "role": "user", "content": "hi"})],
            HashMap::from([("tenant".to_string(), "test".to_string())]),
        );
        assert!(snapshot.id.starts_with("conv_"));
        assert_eq!(snapshot.items[0]["content"], "hi");
        assert_eq!(snapshot.metadata["tenant"], "test");
        assert!(store.get(&snapshot.id).is_none());

        let handler = include_str!("api/conversations.rs");
        assert!(handler.contains(".create_snapshot("));
        assert!(!handler.contains("expect(\"just created\")"));
    }

    #[test]
    fn add_items_respects_cap() {
        let store = ConversationStore::with_config(16, Duration::from_secs(60));
        let id = store.create(Vec::new(), HashMap::new());
        let twenty_one: Vec<_> = (0..21)
            .map(|i| json!({"type": "message", "role": "user", "content": format!("m{i}")}))
            .collect();
        let err = store.add_items(&id, twenty_one).unwrap_err();
        assert!(matches!(err, AddItemsError::TooMany(21)));
    }

    #[test]
    fn delete_item_by_id() {
        let store = ConversationStore::with_config(16, Duration::from_secs(60));
        let id = store.create(
            vec![
                json!({"type": "message", "role": "user", "content": "a"}),
                json!({"type": "message", "role": "user", "content": "b"}),
            ],
            HashMap::new(),
        );
        let snap = store.get(&id).unwrap();
        let item_id = snap.items[0]["id"].as_str().unwrap().to_string();
        assert!(store.remove_item(&id, &item_id));
        let snap2 = store.get(&id).unwrap();
        assert_eq!(snap2.items.len(), 1);
        assert_eq!(snap2.items[0]["content"], "b");
    }

    #[test]
    fn update_metadata_merges() {
        let store = ConversationStore::with_config(16, Duration::from_secs(60));
        let mut md = HashMap::new();
        md.insert("k1".to_string(), "v1".to_string());
        let id = store.create(Vec::new(), md);
        let mut patch = HashMap::new();
        patch.insert("k2".to_string(), "v2".to_string());
        let snap = store.update_metadata(&id, patch).unwrap();
        assert_eq!(snap.metadata["k1"], "v1");
        assert_eq!(snap.metadata["k2"], "v2");
    }

    #[test]
    fn ttl_evicts() {
        let store = ConversationStore::with_config(16, Duration::from_millis(10));
        let id = store.create(Vec::new(), HashMap::new());
        std::thread::sleep(Duration::from_millis(30));
        assert!(store.get(&id).is_none());
    }

    #[test]
    fn add_items_enforces_per_conversation_item_and_byte_caps() {
        let store = ConversationStore::with_limits(16, Duration::from_secs(60), 3, 10_000);
        let id = store.create(Vec::new(), HashMap::new());
        store
            .add_items(&id, vec![json!({"content": "a"}), json!({"content": "b"})])
            .expect("within item cap");
        store
            .add_items(&id, vec![json!({"content": "c"})])
            .expect("reaches exact item cap");
        let err = store
            .add_items(&id, vec![json!({"content": "d"})])
            .unwrap_err();
        assert!(matches!(
            err,
            AddItemsError::ConversationFull { max_items: 3, .. }
        ));

        let tight_bytes = ConversationStore::with_limits(16, Duration::from_secs(60), 100, 128);
        let id2 = tight_bytes.create(Vec::new(), HashMap::new());
        let big = "x".repeat(256);
        let err = tight_bytes
            .add_items(&id2, vec![json!({"content": big})])
            .unwrap_err();
        assert!(matches!(
            err,
            AddItemsError::ConversationFull { max_bytes: 128, .. }
        ));
    }

    #[test]
    fn write_methods_bump_lru_and_remove_item_honors_ttl() {
        let store = ConversationStore::with_config(2, Duration::from_secs(60));
        let a = store.create(Vec::new(), HashMap::new());
        let b = store.create(Vec::new(), HashMap::new());
        // Writing to `a` must refresh its LRU position so `b` is evicted next.
        store
            .add_items(&a, vec![json!({"content": "keep"})])
            .expect("append to a");
        let _c = store.create(Vec::new(), HashMap::new());
        assert!(store.get(&a).is_some(), "recently written `a` survives LRU");
        assert!(store.get(&b).is_none(), "stale `b` was evicted");

        // Expired conversation must not be resurrected by `remove_item`.
        let expiring = ConversationStore::with_config(4, Duration::from_millis(10));
        let d = expiring.create(
            vec![json!({"id": "item_1", "content": "x"})],
            HashMap::new(),
        );
        std::thread::sleep(Duration::from_millis(30));
        assert!(!expiring.remove_item(&d, "item_1"));
        assert!(expiring.get(&d).is_none());
    }

    #[test]
    fn delete_removes_entry() {
        let store = ConversationStore::with_config(16, Duration::from_secs(60));
        let id = store.create(Vec::new(), HashMap::new());
        assert!(store.delete(&id));
        assert!(store.get(&id).is_none());
        assert!(!store.delete(&id));
    }
}
