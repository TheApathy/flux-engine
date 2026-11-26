-- SPDX-License-Identifier: AGPL-3.0-only
-- flux-router: does a POST /v1/responses body reference backend-local state?
--
-- Returns "1" if a TOP-LEVEL `previous_response_id` or `conversation` member has
-- a non-null value, "0" if not, and "bad" if the body is not one complete JSON
-- object (fail closed: a truncated or unparsable body is never forwarded as
-- stateless). Member names are decoded (\uXXXX, surrogate pairs, \n ...) the way
-- serde does, so "previous_response_id" is recognised; nested keys (for
-- example metadata.conversation) are not state and are ignored.

local M = {}
local STATE = { previous_response_id = true, conversation = true }

local function utf8_char(cp)
  if cp < 0x80 then return string.char(cp) end
  if cp < 0x800 then return string.char(0xC0 | (cp >> 6), 0x80 | (cp & 0x3F)) end
  if cp < 0x10000 then
    return string.char(0xE0 | (cp >> 12), 0x80 | ((cp >> 6) & 0x3F), 0x80 | (cp & 0x3F))
  end
  return string.char(0xF0 | (cp >> 18), 0x80 | ((cp >> 12) & 0x3F), 0x80 | ((cp >> 6) & 0x3F), 0x80 | (cp & 0x3F))
end

local SIMPLE = { ['"'] = '"', ['\\'] = '\\', ['/'] = '/', b = '\b', f = '\f', n = '\n', r = '\r', t = '\t' }

local function skip_ws(s, i)
  return s:find("[^ \t\r\n]", i) or (#s + 1)
end

-- Parse a JSON string starting at the opening quote s[i]; return decoded, next index.
local function parse_string(s, i, decode)
  if s:sub(i, i) ~= '"' then return nil end
  i = i + 1
  local out = decode and {} or nil
  while true do
    local j = s:find('["\\\0-\31]', i)
    if not j then return nil end
    local c = s:sub(j, j)
    if out then out[#out + 1] = s:sub(i, j - 1) end
    if c == '"' then return out and table.concat(out) or true, j + 1 end
    if c ~= '\\' then return nil end -- raw control character (< 0x20)
    local e = s:sub(j + 1, j + 1)
    if SIMPLE[e] then
      if out then out[#out + 1] = SIMPLE[e] end
      i = j + 2
    elseif e == 'u' then
      local hex = s:sub(j + 2, j + 5)
      if not hex:match("^%x%x%x%x$") then return nil end
      local cp = tonumber(hex, 16)
      i = j + 6
      if cp >= 0xDC00 and cp <= 0xDFFF then return nil end -- lone low surrogate
      if cp >= 0xD800 and cp <= 0xDBFF then
        local lo = s:sub(i, i + 1) == "\\u" and s:sub(i + 2, i + 5) or ""
        local lcp = lo:match("^%x%x%x%x$") and tonumber(lo, 16)
        if not lcp or lcp < 0xDC00 or lcp > 0xDFFF then return nil end -- lone high surrogate
        cp = 0x10000 + ((cp - 0xD800) << 10) + (lcp - 0xDC00)
        i = i + 6
      end
      if out then out[#out + 1] = utf8_char(cp) end
    else
      return nil
    end
  end
end

-- RFC 8259 number: -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?  -> next index or nil.
local function parse_number(s, i)
  local j = i
  if s:sub(j, j) == "-" then j = j + 1 end
  local c = s:sub(j, j)
  if c == "0" then
    j = j + 1
  elseif c:match("^[1-9]$") then
    j = (s:find("[^0-9]", j + 1) or #s + 1)
  else
    return nil
  end
  if s:sub(j, j) == "." then
    local k = s:find("[^0-9]", j + 1) or #s + 1
    if k == j + 1 then return nil end
    j = k
  end
  if s:sub(j, j):match("^[eE]$") then
    j = j + 1
    if s:sub(j, j):match("^[%+%-]$") then j = j + 1 end
    local k = s:find("[^0-9]", j) or #s + 1
    if k == j then return nil end
    j = k
  end
  return j
end

-- Skip any JSON value at s[i]; return next index and whether it was the literal null.
local function skip_value(s, i, depth)
  if depth > 64 then return nil end
  i = skip_ws(s, i)
  local c = s:sub(i, i)
  if c == '"' then
    local _, n = parse_string(s, i, false)
    return n, false
  elseif c == '{' or c == '[' then
    local close = (c == '{') and '}' or ']'
    i = skip_ws(s, i + 1)
    if s:sub(i, i) == close then return i + 1, false end
    while true do
      if c == '{' then
        local _, n = parse_string(s, i, false)
        if not n then return nil end
        i = skip_ws(s, n)
        if s:sub(i, i) ~= ':' then return nil end
        i = i + 1
      end
      i = skip_value(s, i, depth + 1)
      if not i then return nil end
      i = skip_ws(s, i)
      local d = s:sub(i, i)
      if d == close then return i + 1, false end
      if d ~= ',' then return nil end
      i = skip_ws(s, i + 1)
    end
  else
    for _, lit in ipairs({ "true", "false", "null" }) do
      if s:sub(i, i + #lit - 1) == lit then return i + #lit, lit == "null" end
    end
    return parse_number(s, i), false
  end
end

function M.classify(s)
  if type(s) ~= "string" then return "bad" end
  if not utf8.len(s) or s:find("\237[\160-\191]") then return "bad" end -- invalid UTF-8 / UTF-8 surrogate
  local i = skip_ws(s, 1)
  if s:sub(i, i) ~= '{' then return "bad" end
  i = skip_ws(s, i + 1)
  local ref = false
  if s:sub(i, i) == '}' then
    i = i + 1
  else
    while true do
      local key, n = parse_string(s, i, true)
      if not key then return "bad" end
      i = skip_ws(s, n)
      if s:sub(i, i) ~= ':' then return "bad" end
      local is_null
      i, is_null = skip_value(s, i + 1, 1)
      if not i then return "bad" end
      if STATE[key] and not is_null then ref = true end
      i = skip_ws(s, i)
      local d = s:sub(i, i)
      if d == '}' then i = i + 1; break end
      if d ~= ',' then return "bad" end
      i = skip_ws(s, i + 1)
    end
  end
  if skip_ws(s, i) <= #s then return "bad" end -- trailing data
  return ref and "1" or "0"
end

if core then
  core.register_fetches("flux_state_ref", function(txn)
    return M.classify(txn.sf:req_body())
  end)
end

return M
