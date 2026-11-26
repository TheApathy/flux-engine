#!/usr/bin/env python3
"""Stand-in for flux-server in ts_identity tests: `serve ... --port N [--dflash ...]`.

Text is a function of the prompt token IDs. Switches (env):
  MOCK_S_DIVERGE=<toks[-4] token id>  an S server (--dflash) alters the output when prompt[-4] matches
  MOCK_FAIL_START=1                   exit before becoming ready
  MOCK_CONTROL_SAME=1                 every prompt yields the same text (control cannot differ)
"""
import hashlib, http.server, json, os, sys

args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
spec = "--dflash" in args
if os.environ.get("MOCK_FAIL_START"):
    sys.exit(3)
if spec and os.environ.get("ATLAS_GLM53_UNVALIDATED_BRINGUP") != "1" and not os.environ.get("MOCK_ADMITTED"):
    print("install_dflash2 refused: Speculative scope closed", file=sys.stderr)
    sys.exit(4)

if spec:  # what install_dflash2 logs; tests choose the value with MOCK_SPEC_FNV
    print(f"INFO GLM53_SPEC_SOURCES fnv1a64={os.environ.get('MOCK_SPEC_FNV', '0' * 16)} evidence=not-admitting",
          flush=True)
CONTROL = any(os.environ.get(k) == "1" for k in (
    "ATLAS_GLM53_PREFIX_COMMIT_CONTROL_LEGACY_RESTAGE", "ATLAS_GLM53_SPEC_CONTROL_COMMIT_EXTRA_ROW"))
HASH = os.environ.get("ATLAS_GLM53_STATE_HASH")

KEYS = [f"o{o}.{n}" for o in (0, 17, 33) for n in ("conv", "rec")] + [f"d{i}" for i in range(11)]
REQ = [1]


def state_lines(toks):
    """The real state_hash_probe schema, one line per committed decode position (positions continue
    past the prompt). Spec arms commit through prefix/full; a control arm corrupts from the 4th."""
    out = []
    REQ[0] += 1
    plen = len(json.loads(toks))
    # the last 16 committed positions of a 160-token completion: it ends at plen + 160 - 1
    for i, pos in enumerate(range(plen + 160 - 16, plen + 160)):
        fields = []
        for k in KEYS:
            h = hashlib.sha256(f"{toks}:{pos}:{k}".encode()).hexdigest()[:16]
            if CONTROL and i >= 3:
                h = "f" + h[1:] if h[0] != "f" else "e" + h[1:]
            fields.append(f"{k}={h}")
        src = ("prefix" if i % 2 else "full") if spec else "walk"
        ahead = " ".join(f"d{k}ahead={(pos + k) % 2}" for k in range(11))
        out.append(f"req={REQ[0]} pos={pos} src={src} " + " ".join(fields) + " " + ahead + "\n")
    return out


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send({"status": "ready", "model": "glm53-flash"})

    def do_POST(self):
        n = int(self.headers["Content-Length"])
        raw = self.rfile.read(n)
        body = json.loads(raw)
        if HASH:
            with open(HASH, "a") as f:
                f.writelines(state_lines(json.dumps(body["prompt"])))
        seed = "same" if os.environ.get("MOCK_CONTROL_SAME") else json.dumps(body["prompt"])
        text = hashlib.sha256(seed.encode()).hexdigest()
        if spec and os.environ.get("MOCK_S_DIVERGE") == str(body["prompt"][-4]):
            text += "!"
        if CONTROL:
            text += "#corrupted"
        self._send({"choices": [{"text": text, "finish_reason": "length"}],
                    "usage": {"completion_tokens": 160, "time_to_first_token_ms": 1.0}})


http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
