-- run: lua5.3 tests/lua/test_flux_state.lua (from repro-kit/) or lua5.3 repro-kit/tests/lua/test_flux_state.lua (from repo root)
local ok, M = pcall(function() return dofile("router/flux_state.lua") end)
if not ok then M = dofile("repro-kit/router/flux_state.lua") end
local cases = {
  { [=[{"model":"m","input":"x","previous_response_id":"resp_1"}]=], "1" },
  { [=[{"conversation":"conv_1"}]=], "1" },
  { [=[{"conversation":{"id":"c"}}]=], "1" },
  { "{\"previous_response_id\" :\n  \"r\"}", "1" },
  { [=[{"conversation":"c","conversation":null}]=], "1" },
  { [=[{"model":"m","input":"hi","metadata":{"conversation":"book club"}}]=], "0" },
  { [=[{"input":"parse {\"previous_response_id\": 1}"}]=], "0" },
  { [=[{"previous_response_id":null,"conversation":null,"store":false}]=], "0" },
  { [=[{"input":[{"role":"user","content":[{"type":"input_text","text":"😀"}]}],"temperature":0.5e-1,"n":-2,"x":true}]=], "0" },
  { "{}", "0" },
  { [=[{"input":"x"]=], "bad" },
  { [=[{"input":"x"} trailing]=], "bad" },
  { [=[["previous_response_id"]]=], "bad" },
  { "", "bad" },
  { [=[{"a":tru}]=], "bad" },
  { "{\"a\":\"raw\ncontrol\"}", "bad" },
  { [=[{"a":"\x"}]=], "bad" },
  -- serde-equivalence adversarial JSON parser cases
  { [=[{"conversation":null,"\u0063onversation":"x"}]=], "1" },
  { [=[{"metadata":{"\u0063onversation":"book club"}}]=], "0" },
  { [=[{"x":01}]=], "bad" },
  { [=[{"x":1.}]=], "bad" },
  { [=[{"x":1e}]=], "bad" },
  { [=[{"x":1+}]=], "bad" },
  { [=[{"x":1-2}]=], "bad" },
  { [=[{"x":-}]=], "bad" },
  { [=[{"x":-0.5E+10,"y":0,"z":12.25e-3}]=], "0" },
  { [=[{"x":"\ud800"}]=], "bad" },
  { [=[{"\udc00":0}]=], "bad" },
  { [=[{"x":"\ud800\u0041"}]=], "bad" },
  { '{"x":"' .. string.char(0xff) .. '"}', "bad" },
  { '{"x":"' .. string.char(0xED, 0xA0, 0x80) .. '"}', "bad" },
  { '{"x":"' .. string.char(127) .. '"}', "0" },
  { '{"x":"' .. string.char(0xE2, 0x82, 0xAC) .. '"}', "0" },
  { [=[{"metadata":{"x":1},"conversation":]=], "bad" },
}
for k, c in ipairs(cases) do
  local got = M.classify(c[1])
  if got ~= c[2] then
    print(("FAIL case %d: got %s want %s: %s"):format(k, got, c[2], c[1]))
    os.exit(1)
  end
end
print(("flux_state.lua: %d cases OK"):format(#cases))
