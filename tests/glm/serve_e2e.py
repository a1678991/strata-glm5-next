"""tests/glm/serve_e2e.py - serve/server.py with the GLM (glm5-next) engine, driven over OpenAI chat completions.
Not a ctest: it needs a GPU, a GLM pack and the serve venv.

    python tests/glm/serve_e2e.py --exe build-hip/strata --pack PACK --native SHARD1.gguf --profile PROFILE.strp \
        --work /some/dir [--long-text FILE] [--engine-arg=--expert-cache --engine-arg=64 ...]

Phases (each a fresh server process; the engine log and the request lines the engine read are kept per phase):
  1 "main"  (disk dir D1): streaming, multi-turn prefix reuse, a second conversation that parks the first, the first
            again (restored from RAM), thinking off, the effort levels, a tool-call round trip, a cancellation while
            decoding and one while reading a long prompt; the server is then stopped (the live conversation is written
            to disk at exit)
  2 "disk"  (D1 again): the first conversation's third turn once more: restored from the disk tier
  3 "ref"   (fresh D2): the first conversation's three turns uninterrupted: the reference
Every request is greedy (temperature 0) unless said; the engine log's per-request logits hash (STRATA_GLM_LOGITS_HASH)
and the generated text of the third turn must be identical in 1 (restored from RAM), 2 (from disk) and 3 (never
parked).  The follow-up messages are long enough (> --short-read 64 tokens) to go through the prompt path at pos0 > 0.
The engine runs with --glm-pool off (the auto calibration is timing-based, and the pool rounds differently).
"""
import argparse, http.client, json, os, re, shutil, signal, socket, subprocess, sys, time

TREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--exe", required=True)
ap.add_argument("--pack", required=True)
ap.add_argument("--native", required=True)
ap.add_argument("--profile", required=True)
ap.add_argument("--tokenizer", help="default: PACK/tokenizer")
ap.add_argument("--work", required=True, help="a directory for the logs and the disk tier (large: GLM images)")
ap.add_argument("--long-text", help="a text of 60,000+ characters for the cancel-while-reading case (default: skipped)")
ap.add_argument("--port", type=int, default=8197)
ap.add_argument("--max-context", default="32768")
ap.add_argument("--engine-arg", action="append", default=[], help="more engine arguments (e.g. the tier sizes)")
A = ap.parse_args()
BIN, WORK, PORT = os.path.abspath(A.exe), os.path.abspath(A.work), A.port
TOKDIR = A.tokenizer or os.path.join(A.pack, "tokenizer")
sys.path.insert(0, TREE + "/tools")
import strata_tokenizer as ST  # noqa: E402
TOK = ST.Tokenizer.from_dir(TOKDIR)


def write_config(out, disk, log, cap):
    wrapper = os.path.join(WORK, "strata-capture.sh")
    with open(wrapper, "w") as f:   # the engine, with every request line it reads appended to $GLM_CAPTURE
        f.write('#!/bin/bash\nexec "$GLM_BIN" "$@" < <(tee -a "$GLM_CAPTURE")\n')
    os.chmod(wrapper, 0o755)
    args = ["--pack", A.pack, "--native", A.native, "--expert-profile", A.profile, "--prefill", "4096",
            "--glm-pool", "off", "--max-context", A.max_context, "--kv", "int8", "--conversation-cache-mib", "2048",
            "--conversation-cache-slots", "4", "--prompt-cache-root", "16", "--conversation-cache-disk", disk,
            "--conversation-cache-disk-mib", "8192", "--conversation-cache-disk-keep-superseded", *A.engine_arg]
    cfg = {"exe": wrapper, "args": args, "cwd": WORK, "tokenizer": TOKDIR, "arch": "glm5-next", "model_name": "glm-test",
           "log": log, "host": "127.0.0.1", "port": PORT, "gpu": 0,
           "backend": "hip" if os.path.exists("/opt/rocm") and not os.path.exists("/usr/bin/nvidia-smi") else "cuda",
           "sampling": {"temperature": 1.0, "top_p": 0.95, "min_p": 0.0, "top_k": 40}, "fit_max_tokens": True,
           "env": {"GLM_CAPTURE": cap, "GLM_BIN": BIN, "STRATA_GLM_LOGITS_HASH": "1"}}
    json.dump(cfg, open(out, "w"), indent=1)


results = []


def ok(cond, what):
    results.append((bool(cond), what))
    print(("PASS " if cond else "FAIL ") + what, flush=True)
    return cond


class Server:
    def __init__(self, phase, disk):
        self.phase = phase
        self.dir = f"{WORK}/{phase}"
        os.makedirs(self.dir, exist_ok=True)
        self.log = f"{self.dir}/engine.log"
        self.cap = f"{self.dir}/requests.txt"
        for f in (self.log, self.cap):
            if os.path.exists(f):
                os.remove(f)
        cfg = f"{self.dir}/cfg.json"
        write_config(cfg, disk, self.log, self.cap)
        self.out = open(f"{self.dir}/server.out", "w")
        self.p = subprocess.Popen([sys.executable, "-m", "serve.server", "--engine", "strata", "--config", cfg, "--port", str(PORT)],
                                  cwd=TREE, stdout=self.out, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 300:
            try:
                c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=5)
                c.request("GET", "/health")
                r = c.getresponse()
                if r.status == 200 and json.loads(r.read()).get("loaded"):
                    break
            except OSError:
                pass
            if self.p.poll() is not None:
                raise RuntimeError("server exited: " + open(f"{self.dir}/server.out").read()[-2000:])
            time.sleep(1)
        print(f"[{phase}] server up in {time.time() - t0:.0f} s", flush=True)

    def stop(self):
        self.p.send_signal(signal.SIGTERM)
        try:
            self.p.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self.p.kill()
        self.out.close()

    def requests(self):
        """The GEN lines the engine read, as (max_new, keys, ids)."""
        out = []
        for line in open(self.cap):
            m = re.match(r"GEN (\d+)((?: \S+=\S+)*) ([\d,]+)$", line.strip())
            if m:
                out.append((int(m.group(1)), m.group(2).strip(), [int(x) for x in m.group(3).split(",")]))
        return out

    def engine_lines(self, what):
        return [l.rstrip() for l in open(self.log, errors="replace") if what in l]


def chat(messages, max_tokens=24, tools=None, stream=True, timeout=600, **kw):
    body = {"model": "glm-test", "messages": messages, "max_tokens": max_tokens, "stream": stream,
            "temperature": kw.pop("temperature", 0)}
    if tools:
        body["tools"] = tools
    body.update(kw)
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    res = {"status": r.status, "reasoning": "", "content": "", "tool_calls": [], "finish": None, "usage": None,
           "chunks": 0}
    if not stream:
        j = json.loads(r.read())
        m = j["choices"][0]["message"]
        res.update(content=m.get("content") or "", reasoning=m.get("reasoning_content") or "",
                   tool_calls=m.get("tool_calls") or [], finish=j["choices"][0]["finish_reason"], usage=j.get("usage"))
        return res
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        j = json.loads(line[6:])
        res["chunks"] += 1
        ch = j["choices"][0] if j.get("choices") else {}
        d = ch.get("delta") or {}
        res["reasoning"] += d.get("reasoning_content") or ""
        res["content"] += d.get("content") or ""
        res["tool_calls"] += d.get("tool_calls") or []
        if ch.get("finish_reason"):
            res["finish"] = ch["finish_reason"]
        if j.get("usage"):
            res["usage"] = j["usage"]
    return res


def chat_cancel(messages, after_chunks=None, after_s=None, max_tokens=400):
    """A streamed request the client walks away from: after `after_chunks` data lines, or `after_s` seconds."""
    body = {"model": "glm-test", "messages": messages, "max_tokens": max_tokens, "stream": True, "temperature": 0}
    s = socket.create_connection(("127.0.0.1", PORT))
    payload = json.dumps(body).encode()
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:" + str(PORT).encode() + b"\r\nContent-Type: application/json\r\nContent-Length: " +
              str(len(payload)).encode() + b"\r\n\r\n" + payload)
    t0, got, buf = time.time(), 0, b""
    s.settimeout(0.5)
    while True:
        if after_s is not None and time.time() - t0 >= after_s:
            break
        try:
            data = s.recv(65536)
        except socket.timeout:
            continue
        if not data:
            break
        buf += data
        got = buf.count(b"data: {")
        if after_chunks is not None and got >= after_chunks:
            break
    s.close()
    return got


def turn_ids(srv, i):
    return srv.requests()[i][2]


def logits_hash(srv, i):
    lines = srv.engine_lines("strata serve: prompt ")
    m = re.search(r"logits ([0-9a-f]{16})", lines[i])
    return m.group(1) if m else None


def reused_of(srv, i):
    m = re.search(r"prompt (\d+) tokens = (\d+) reused", srv.engine_lines("strata serve: prompt ")[i])
    return int(m.group(1)), int(m.group(2))


SYS_A = "You are a careful assistant for a small hardware store. Answer briefly and cite the aisle number. " * 3
SYS_B = "You translate English into French and nothing else. " * 3
LONG2 = "I need them for oak, pine and some steel brackets, in sizes from 3 to 10 millimetres, ideally a set with a case. " * 4
LONG3 = "We open a workshop next week and need to plan two visits, one on a weekday morning and one on Saturday afternoon. " * 4
A1 = [{"role": "system", "content": SYS_A}, {"role": "user", "content": "Where do I find wood screws?"}]


def a_turns(srv):
    """A1, A2, A3 of conversation A, each answered greedily; returns the three results."""
    r1 = chat(A1)
    m2 = A1 + [{"role": "assistant", "content": r1["content"]}, {"role": "user", "content": "And drill bits? " + LONG2}]
    r2 = chat(m2)
    return r1, r2, m2


def a3_messages(m2, r2):
    return m2 + [{"role": "assistant", "content": r2["content"]}, {"role": "user", "content": "Thanks. Opening hours? " + LONG3}]


def phase_main():
    shutil.rmtree(f"{WORK}/disk1", ignore_errors=True)
    srv = Server("main", f"{WORK}/disk1")
    n = 0
    try:
        r1, r2, m2 = a_turns(srv)
        n += 2
        ok(r1["status"] == 200 and r1["chunks"] > 2 and r1["finish"] in ("length", "stop"),
           f"A1 streamed: {r1['chunks']} chunks, finish {r1['finish']}, usage {r1['usage']}")
        ok(r1["usage"]["completion_tokens"] == 24 or r1["finish"] == "stop", "A1 completion_tokens")
        p2, re2 = reused_of(srv, 1)
        ok(re2 > 0 and r2["usage"]["prompt_tokens_details"]["cached_tokens"] == re2,
           f"A2 reuses the conversation: {re2} of {p2} prompt tokens (usage cached_tokens "
           f"{r2['usage']['prompt_tokens_details']['cached_tokens']})")
        # conversation B parks A
        rb = chat([{"role": "system", "content": SYS_B}, {"role": "user", "content": "Good morning"}])
        n += 1
        ok(rb["status"] == 200, "B1 answered")
        ok(any("conversation cache: parked" in l for l in srv.engine_lines("conversation cache: parked")),
           "A was parked when B started")
        # A3: restored from RAM
        m3 = a3_messages(m2, r2)
        r3 = chat(m3)
        n += 1
        p3, re3 = reused_of(srv, n - 1)
        restored = srv.engine_lines("conversation cache: restored")
        ok(re3 > 0 and restored, f"A3 resumed from the parked conversation: {re3} of {p3} reused; {restored[-1:]}")
        ok(p3 - re3 > 64 and p2 - re2 > 64, f"A2 and A3 read their new parts ({p2 - re2}, {p3 - re3} tokens) through the "
           "prompt path at pos0 > 0 (more than --short-read 64)")
        h3, ids3 = logits_hash(srv, n - 1), r3["reasoning"] + r3["content"]
        # thinking off and the effort levels: the prompt the engine got
        roff = chat(A1, reasoning_effort="none")
        n += 1
        off_txt = TOK.decode(turn_ids(srv, n - 1))
        ok(off_txt.endswith("<|assistant|><think></think>") and roff["status"] == 200,
           "thinking off: the prompt ends with <|assistant|><think></think>")
        ok(roff["reasoning"] == "", f"thinking off: no reasoning_content (content {len(roff['content'])} chars)")
        for eff, want in (("low", "Low"), ("medium", "High"), ("high", "High"), ("max", "Max")):
            r = chat(A1, reasoning_effort=eff, max_tokens=4)
            n += 1
            t = TOK.decode(turn_ids(srv, n - 1))
            ok(r["status"] == 200 and f"<|system|>Reasoning Effort: {want}" in t and t.endswith("<|assistant|><think>"),
               f"effort {eff}: 'Reasoning Effort: {want}', thinking on")
        r = chat(A1, max_tokens=4)
        n += 1
        ok("Reasoning Effort: Max" in TOK.decode(turn_ids(srv, n - 1)), "effort unset: Max (the template's default)")
        # a tool-call round trip: the call and its result rendered into the next prompt, which reuses the first
        tools = [{"type": "function", "function": {"name": "get_stock", "description": "Units in stock for an item",
                                                   "parameters": {"type": "object", "properties": {
                                                       "item": {"type": "string"}, "aisle": {"type": "integer"}},
                                                       "required": ["item"]}}}]
        tm = [{"role": "user", "content": "How many hammers are in stock?"}]
        rt1 = chat(tm, tools=tools)
        n += 1
        t1 = TOK.decode(turn_ids(srv, n - 1))
        ok(rt1["status"] == 200 and "# Tools" in t1 and '"name": "get_stock"' in t1, "tools rendered into the prompt")
        tm2 = tm + [{"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "get_stock",
                                                              "arguments": json.dumps({"item": "hammer", "aisle": 7})}}]},
                    {"role": "tool", "tool_call_id": "call_1", "content": "{\"units\": 42}"}]
        rt2 = chat(tm2, tools=tools)
        n += 1
        t2 = TOK.decode(turn_ids(srv, n - 1))
        ok("<tool_call>get_stock<arg_key>item</arg_key><arg_value>hammer</arg_value><arg_key>aisle</arg_key>"
           "<arg_value>7</arg_value></tool_call>" in t2, "the tool call rendered in GLM's grammar")
        ok("<|observation|><tool_response>{\"units\": 42}</tool_response>" in t2, "the tool result as an observation")
        pt2, ret2 = reused_of(srv, n - 1)
        ok(rt2["status"] == 200 and ret2 > 0, f"the tool turn reuses the conversation: {ret2} of {pt2}")
        # cancellation while decoding
        got = chat_cancel(A1, after_chunks=4)
        t0 = time.time()
        while len(srv.engine_lines("strata serve: prompt ")) < n + 1 and time.time() - t0 < 600:
            time.sleep(1)
        n += 1
        last = srv.engine_lines("strata serve: prompt ")
        ok(len(last) == n and "(cancelled)" not in last[-1] and re.search(r"(\d+) generated", last[-1]) and
           int(re.search(r"(\d+) generated", last[-1]).group(1)) < 400,
           f"cancel while decoding: the client left after {got} chunks, the engine stopped at "
           f"{re.search(r'(\d+) generated', last[-1]).group(1)} of 400 tokens")
        r = chat(A1, max_tokens=4)
        n += 1
        ok(r["status"] == 200 and r["finish"] == "length", "the next request after the cancel runs")
        # cancellation while reading a long prompt (16K tokens of text: ~4 s on the prompt path here)
        if A.long_text:
            long_text = open(A.long_text, errors="replace").read()[:60000]
            lm = [{"role": "user", "content": "Summarize: " + long_text}]
            chat_cancel(lm, after_s=1.5)
            t0 = time.time()
            while len(srv.engine_lines("strata serve: prompt ")) < n + 1 and time.time() - t0 < 600:
                time.sleep(1)
            n += 1
            last = srv.engine_lines("strata serve: prompt ")
            ok(len(last) == n and "(cancelled)" in last[-1], f"cancel while reading the prompt: {last[-1][13:120]}")
            r = chat(A1, max_tokens=4)
            n += 1
            ok(r["status"] == 200, "the next request after a cancelled prompt runs")
        # A3 once more, last: the live conversation the engine writes to disk at exit
        r3b = chat(m3)
        n += 1
        h3b = logits_hash(srv, n - 1)
        ok(h3b == h3 and r3b["reasoning"] + r3b["content"] == ids3,
           f"A3 again (restored again): logits {h3b} = {h3}, the same text")
    finally:
        srv.stop()
    ok(srv.engine_lines("wrote the live conversation"), "the live conversation was written to disk at exit")
    return {"h3": h3, "text3": ids3, "m3": m3, "ids3": turn_ids(srv, 3)}


def phase_disk(main):
    srv = Server("disk", f"{WORK}/disk1")
    try:
        ok(srv.engine_lines("conversation cache disk:") and
           re.search(r"(\d+) entries", srv.engine_lines("conversation cache disk: /")[0]).group(1) != "0",
           "the disk tier reopened with entries: " + srv.engine_lines("conversation cache disk: /")[0][13:])
        r3 = chat(main["m3"])
        p, re3 = reused_of(srv, 0)
        ok(srv.engine_lines("conversation cache disk: read"), "A3 read from the disk tier: " +
           (srv.engine_lines("conversation cache disk: read") or ["-"])[0][13:])
        ok(re3 > 0, f"A3 after a restart reuses {re3} of {p}")
        h = logits_hash(srv, 0)
        ok(h == main["h3"] and r3["reasoning"] + r3["content"] == main["text3"],
           f"A3 from disk: logits {h} = {main['h3']}, the same text")
        ok(turn_ids(srv, 0) == main["ids3"], "A3's prompt ids are the same as in the first run")
    finally:
        srv.stop()


def phase_ref(main):
    shutil.rmtree(f"{WORK}/disk2", ignore_errors=True)
    srv = Server("ref", f"{WORK}/disk2")
    try:
        r1, r2, m2 = a_turns(srv)
        r3 = chat(a3_messages(m2, r2))
        ok(a3_messages(m2, r2) == main["m3"], "the reference renders the same three turns")
        h = logits_hash(srv, 2)
        ok(h == main["h3"] and r3["reasoning"] + r3["content"] == main["text3"],
           f"A3 never parked: logits {h} = {main['h3']} (restored from RAM / disk), the same text")
        ok(not srv.engine_lines("conversation cache: restored"), "the reference restored nothing")
    finally:
        srv.stop()


if __name__ == "__main__":
    os.makedirs(WORK, exist_ok=True)
    print(f"engine {BIN}; work {WORK}", flush=True)
    main = phase_main()
    phase_disk(main)
    phase_ref(main)
    bad = [w for good, w in results if not good]
    print(f"\n{len(results) - len(bad)} of {len(results)} checks passed" + ("" if not bad else "; FAILED: " + "; ".join(bad)))
    sys.exit(1 if bad else 0)
