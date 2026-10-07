"""serve/test_glm.py - GLM-5.3-Flash (glm5-next) in the server: its template, efforts, thinking off, tool calls and
end-of-turn tokens, against the mock engine (no GPU, no pack).

    python -m unittest serve.test_glm -v

The prompts in serve/chat_golden_glm.json are serve/glm_chat_template.jinja (the pack's chat_template.jinja) rendered
by the server for API requests.  With a GLM pack's tokenizer (STRATA_GLM_TOKENIZER=<pack>/tokenizer) the real ids are
checked too; with a Qwen one (STRATA_TOKENIZER) that Qwen's ids did not change.
"""
from __future__ import annotations

import contextlib
import http.client
import io
import json
import os
import random
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from serve.frontend import (EFFORT, GLM_EFFORTS, QWEN_EFFORTS, ChatTemplate, GlmOutputParser,  # noqa: E402
                            anthropic_to_messages, budget_effort, effort_kwargs, glm_call_end, openai_to_messages,
                            parse_glm_tool_call)
from serve.models import GLM, QWEN, profile_for  # noqa: E402
from serve.server import (ByteTokenizer, GlmByteTokenizer, MockEngine, Service, clean_shared_defaults,  # noqa: E402
                          effort_end_args, serve)

TEMPLATE = ChatTemplate(ROOT / "serve/glm_chat_template.jinja")
GOLDEN = json.loads((ROOT / "serve/chat_golden_glm.json").read_text(encoding="utf-8"))

WEATHER_PROPS = {"city": {"type": "string"}, "days": {"type": "integer"}, "opts": {"type": "object"}}
OPENAI_TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather.",
                                                  "parameters": {"type": "object", "properties": WEATHER_PROPS}}}]
ANTHROPIC_TOOLS = [{"name": "get_weather", "description": "Weather.",
                    "input_schema": {"type": "object", "properties": WEATHER_PROPS}}]
RESPONSES_TOOLS = [{"type": "function", "name": "get_weather", "description": "Weather.",
                    "parameters": {"type": "object", "properties": WEATHER_PROPS}}]
# two calls: unicode, a nested JSON object with escaped quotes and non-ASCII, an integer; the model ends the turn
# with <|observation|> as GLM does after tool calls
CALLS = ("Two cities.</think>Checking both.<tool_call>get_weather<arg_key>city</arg_key><arg_value>京都"
         "</arg_value><arg_key>opts</arg_key><arg_value>{\"units\": \"C\", \"hours\": [6, 12], \"note\": "
         "\"雨 \\\"heavy\\\" \U0001F327\"}</arg_value></tool_call><tool_call>get_weather<arg_key>city</arg_key>"
         "<arg_value>Paris</arg_value><arg_key>days</arg_key><arg_value>3</arg_value></tool_call><|observation|>")
CALL_ARGS = [{"city": "京都", "opts": {"units": "C", "hours": [6, 12], "note": "雨 \"heavy\" \U0001F327"}},
             {"city": "Paris", "days": 3}]


def glm_service(script="Thinking.</think>Hello.", tok=None, **kw):
    tok = tok or GlmByteTokenizer()
    eng = MockEngine(tok, script, max_context=1 << 16)
    return Service(eng, tok, TEMPLATE, **kw), eng, tok


def events_of(text, chunks, tools=None, thinking=True):
    p = GlmOutputParser(thinking=thinking, tools=tools, stream_tools=True)
    evs = []
    for c in chunks:
        evs += p.feed(c)
    return evs + p.finish()


def summary(evs):
    """(reasoning, content, [(name, arguments)], [json.loads(streamed arguments)] per call)."""
    reasoning = "".join(e.text for e in evs if e.kind == "reasoning")
    content = "".join(e.text for e in evs if e.kind == "content")
    calls = [e.call for e in evs if e.kind == "tool_call"]
    streamed = {}
    for e in evs:
        if e.kind == "tool_args":
            streamed[e.call.id] = streamed.get(e.call.id, "") + e.text
    return reasoning, content, [(c.name, c.arguments) for c in calls], [json.loads(streamed[c.id]) for c in calls]


# ------------------------------------------------------------------------------------------------ prompts
class Golden(unittest.TestCase):
    def test_the_golden_prompts(self):
        svc, _, _ = glm_service()
        self.assertIs(svc.profile, GLM)
        for case in GOLDEN:
            with self.subTest(case["name"]):
                to = openai_to_messages if case["api"] == "openai" else \
                    lambda r, e: anthropic_to_messages(r, True, e)
                messages, tools, kw = to(case["request"], svc.profile.efforts)
                self.assertEqual(kw, case["kwargs"])
                self.assertEqual(svc.render_prompt(messages, tools, kw), case["rendered"])
                ids, thinking, _ = svc.prepare(messages, tools, kw, 16)
                self.assertEqual(thinking, case["thinking"])
                self.assertEqual(svc.tok.decode(ids), case["rendered"])

    def test_the_main_forms_by_hand(self):
        """The golden set's key prompts, written out here rather than generated."""
        by = {c["name"]: c["rendered"] for c in GOLDEN}
        head = "[gMASK]<sop><|system|>Reasoning Effort: "
        self.assertEqual(by["single user"], head + "Max<|user|>Hello<|assistant|><think>")
        self.assertEqual(by["effort low"], head + "Low<|user|>Hello<|assistant|><think>")
        self.assertEqual(by["effort medium"], head + "High<|user|>Hello<|assistant|><think>")
        self.assertEqual(by["effort high"], head + "High<|user|>Hello<|assistant|><think>")
        self.assertEqual(by["effort xhigh"], head + "Max<|user|>Hello<|assistant|><think>")
        self.assertEqual(by["effort none (thinking off)"], head + "Max<|user|>Hello<|assistant|><think></think>")
        self.assertEqual(by["system + user"], head + "Max<|system|>You are terse.<|user|>Hello<|assistant|><think>")
        self.assertEqual(by["multi-turn with reasoning"], head + "Max<|user|>Hello<|assistant|><think>Greet back."
                         "</think>Hi there.<|user|>And now?<|assistant|><think>")
        self.assertTrue(by["two tool calls and their results"].endswith(
            "<|user|>Weather in Paris and Kyoto?<|assistant|><think>Two cities.</think><tool_call>get_weather"
            "<arg_key>city</arg_key><arg_value>Paris</arg_value><arg_key>days</arg_key><arg_value>3</arg_value>"
            "</tool_call><tool_call>get_weather<arg_key>city</arg_key><arg_value>京都</arg_value>"
            "<arg_key>opts</arg_key><arg_value>{\"units\": \"C\", \"hours\": [6, 12]}</arg_value></tool_call>"
            "<|observation|><tool_response>sunny, 21C</tool_response><tool_response>雨 (rain)</tool_response>"
            "<|assistant|><think>"))

    def test_a_quoted_think_tag_stays_text(self):
        """#537 on GLM: a literal </think> in a message is plain text, the template's own tags are the markers."""
        svc, _, tok = glm_service()
        msgs = [{"role": "user", "content": "say </think> now"}]
        text = tok.decode(svc.encode_prompt(msgs, None, {}))
        self.assertTrue(text.endswith("<|user|>say </think> now<|assistant|><think>"))


class Efforts(unittest.TestCase):
    def test_glm_mapping(self):
        want = {"none": {"enable_thinking": False}, "off": {"enable_thinking": False},
                "minimal": {"enable_thinking": False}, "low": {"reasoning_effort": "low"},
                "medium": {"reasoning_effort": "high"}, "high": {"reasoning_effort": "high"},
                "xhigh": {"reasoning_effort": "max"}, "max": {"reasoning_effort": "max"}, "": {}, None: {}}
        for value, kw in want.items():
            self.assertEqual(effort_kwargs(value, GLM_EFFORTS), kw, value)
        self.assertEqual(effort_kwargs(False, GLM_EFFORTS), {"enable_thinking": False})
        with self.assertRaisesRegex(ValueError, "none, low, medium, high or max"):
            effort_kwargs("huge", GLM_EFFORTS)
        self.assertEqual([budget_effort(n, GLM_EFFORTS) for n in (1000, 4000, 9000)],
                         [{"reasoning_effort": x} for x in ("low", "high", "max")])

    def test_qwen_mapping_is_unchanged(self):
        self.assertEqual(EFFORT, {"none": None, "off": None, "minimal": None, "disabled": None, "false": None,
                                  "low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh", "max": "xhigh",
                                  "maximum": "xhigh"})
        self.assertIs(effort_kwargs.__defaults__[0], None)
        for value in EFFORT:
            self.assertEqual(effort_kwargs(value), effort_kwargs(value, QWEN_EFFORTS))
        self.assertEqual([budget_effort(n) for n in (1000, 4000, 9000)],
                         [{"reasoning_effort": x} for x in ("low", "medium", "xhigh")])
        with self.assertRaisesRegex(ValueError, "use none, low, medium or high$"):
            effort_kwargs("huge")

    def test_shared_settings(self):
        self.assertEqual(clean_shared_defaults({"reasoning_effort": "max"}, GLM.shared_efforts),
                         {"reasoning_effort": "max"})
        with self.assertRaisesRegex(ValueError, "^reasoning_effort: none, low, medium or high$"):
            clean_shared_defaults({"reasoning_effort": "max"})
        svc, eng, tok = glm_service()
        svc.set_shared({"reasoning_effort": "max"})
        req = svc.with_shared({"messages": [{"role": "user", "content": "hi"}]}, "openai")
        self.assertEqual(openai_to_messages(req, svc.profile.efforts)[2], {"reasoning_effort": "max"})

    def test_effort_at_the_end(self):
        """#458's prompt form for GLM: the default (Max) up to the answer, then the effort's turn.  The engine's
        --tail-role-token rule is ChatML's, so the config's "effort_position": "end" is refused for GLM for now."""
        svc, _, tok = glm_service()
        chat = [{"role": "system", "content": "Be careful."}, {"role": "user", "content": "Explain."}]
        default = svc.render_prompt(chat, None, {})
        svc.effort_end = True
        low = svc.render_prompt(chat, None, {"reasoning_effort": "low"})
        off = svc.render_prompt(chat, None, {"enable_thinking": False})
        self.assertEqual(svc.render_prompt(chat, None, {"reasoning_effort": "max"}), default)
        cut = len(default) - len("<|assistant|><think>")
        self.assertEqual(low, default[:cut] + "<|system|>Reasoning Effort: Low<|assistant|><think>")
        self.assertEqual(off, default + "</think>")
        self.assertEqual(svc.render_prompt(chat, None, {"reasoning_effort": "high"}),
                         default[:cut] + "<|system|>Reasoning Effort: High<|assistant|><think>")
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "strata"
            exe.write_bytes(b"... --tail-role-token ID --serve ...")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertIsNone(effort_end_args({"effort_position": "end"}, str(exe), tok, GLM))
            self.assertIn("not available for glm5-next", out.getvalue())


# ------------------------------------------------------------------------------------------------ the parser
class Parser(unittest.TestCase):
    TOOLS = [{"name": "get_weather", "parameters": {"properties": WEATHER_PROPS}},
             {"name": "magic_int", "parameters": {"properties": {"ref": {"type": "integer"}, "name": {"type": "string"}}}},
             {"name": "write", "parameters": {"properties": {"path": {"type": "string"}, "content": {"type": "string"}}}},
             {"name": "set_nullable_str", "parameters": {"properties": {"name": {"type": ["string", "null"]}}}}]

    # llama.cpp f1cee99 tests/test-chat.cpp L4555-4596 (GLM-4.7-Flash, the same grammar): six calls
    LLAMA = ("Preparing updates.<tool_call>magic_int<arg_key>ref</arg_key><arg_value>42</arg_value><arg_key>name"
             "</arg_key><arg_value>上海</arg_value></tool_call><tool_call>amount<arg_key>orig</arg_key>"
             "<arg_value>2.5</arg_value></tool_call><tool_call>toggle<arg_key>enabled</arg_key><arg_value>true"
             "</arg_value></tool_call><tool_call>set_config<arg_key>config</arg_key><arg_value>{\"source\": \"spark\", "
             "\"options\": {\"strict\": true}}</arg_value></tool_call><tool_call>nested_args<arg_key>tags</arg_key>"
             "<arg_value>[\"alpha\", \"测试\"]</arg_value><arg_key>entries</arg_key><arg_value>[{\"id\": 1, "
             "\"label\": \"first\"}, {\"id\": 2, \"label\": \"第二\"}]</arg_value></tool_call><tool_call>"
             "empty_args</tool_call>")
    LLAMA_CALLS = [("magic_int", {"ref": 42, "name": "上海"}), ("amount", {"orig": 2.5}),
                   ("toggle", {"enabled": True}),
                   ("set_config", {"config": {"source": "spark", "options": {"strict": True}}}),
                   ("nested_args", {"tags": ["alpha", "测试"],
                                    "entries": [{"id": 1, "label": "first"}, {"id": 2, "label": "第二"}]}),
                   ("empty_args", {})]

    def check_all_splits(self, text, want, tools=None, thinking=True, randoms=300):
        """Whole, every split in two, every 1-character step and random 1-5 character steps: the same events."""
        splits = [[text], list(text)] + [[text[:k], text[k:]] for k in range(1, len(text))]
        rnd = random.Random(0)
        for _ in range(randoms):
            chunks, i = [], 0
            while i < len(text):
                n = rnd.randint(1, 5)
                chunks.append(text[i:i + n])
                i += n
            splits.append(chunks)
        for chunks in splits:
            r, c, calls, streamed = summary(events_of(text, chunks, tools or self.TOOLS, thinking))
            self.assertEqual((r, c, calls), want, chunks[:3])
            self.assertEqual(streamed, [a for _, a in calls], chunks[:3])     # the streamed JSON is the call's

    def test_llama_cpp_cases(self):
        self.check_all_splits(self.LLAMA, ("", "Preparing updates.", self.LLAMA_CALLS), thinking=False)

    def test_calls_after_thinking_nested_json_and_unicode(self):
        text = CALLS[:-len("<|observation|>")]
        self.check_all_splits(text, ("Two cities.", "Checking both.",
                                     [("get_weather", CALL_ARGS[0]), ("get_weather", CALL_ARGS[1])]))

    def test_marker_text_inside_values(self):
        """A value may contain the call grammar's own tags (llama.cpp L4703-4713; #210 for Qwen): a </arg_value> that
        is not followed by <arg_key> or </tool_call> belongs to the value."""
        text = ("check this\n</think>\n<tool_call>write\n<arg_key>path</arg_key>\n<arg_value>doc.md</arg_value>\n"
                "<arg_key>content</arg_key>\n<arg_value>Write <tool_call>f<arg_key>k</arg_key><arg_value>v"
                "</arg_value>, and </think> stays.</arg_value>\n</tool_call>")
        self.check_all_splits(text, ("check this\n", "", [("write", {
            "path": "doc.md", "content": "Write <tool_call>f<arg_key>k</arg_key><arg_value>v</arg_value>, and "
                                         "</think> stays."})]))

    def test_types(self):
        cases = [("magic_int", "<arg_key>ref</arg_key><arg_value>42</arg_value>", {"ref": 42}),
                 ("magic_int", "<arg_key>name</arg_key><arg_value>42</arg_value>", {"name": "42"}),   # a string
                 ("set_nullable_str", "<arg_key>name</arg_key><arg_value>null</arg_value>", {"name": None}),
                 ("set_nullable_str", "<arg_key>name</arg_key><arg_value>123 Main St</arg_value>",
                  {"name": "123 Main St"}),
                 ("unknown", "<arg_key>x</arg_key><arg_value>[1, 2]</arg_value><arg_key>y</arg_key><arg_value>plain"
                  "</arg_value>", {"x": [1, 2], "y": "plain"}),
                 ("write", "<arg_key>content</arg_key><arg_value>\n  indented\n</arg_value>",
                  {"content": "\n  indented\n"})]                                # GLM's values are not trimmed
        for name, body, args in cases:
            with self.subTest(body):
                self.check_all_splits("</think><tool_call>" + name + body + "</tool_call>",
                                      ("", "", [(name, args)]), randoms=20)

    def test_glm46_newlines(self):
        self.check_all_splits("</think><tool_call>magic_int\n<arg_key>ref</arg_key>\n<arg_value>1</arg_value>\n"
                              "</tool_call>", ("", "", [("magic_int", {"ref": 1})]))

    def test_cut_and_unterminated(self):
        """#211's rules: a call the output ends inside is not a whole call (its JSON stays open); one that only lacks
        its </tool_call> is whole; a <tool_call> that never got a name comes back as text."""
        head = "</think><tool_call>write<arg_key>path</arg_key><arg_value>a.txt</arg_value><arg_key>content</arg_key>"
        for step in (1, 3, 10_000):
            evs = events_of("", [x for x in [(head + "<arg_value>half of it")[i:i + step]
                                             for i in range(0, len(head) + 30, step)] if x], self.TOOLS)
            self.assertEqual([e.kind for e in evs if e.kind == "tool_call"], [])
            self.assertEqual("".join(e.text for e in evs if e.kind == "tool_args"),
                             '{"path":"a.txt","content":"half of it')
            whole = head + "<arg_value>all</arg_value>"
            _, _, calls, streamed = summary(events_of("", [whole[i:i + step] for i in range(0, len(whole), step)],
                                                      self.TOOLS))
            self.assertEqual(calls, [("write", {"path": "a.txt", "content": "all"})])
            self.assertEqual(streamed, [{"path": "a.txt", "content": "all"}])
        evs = events_of("", ["</think>Text <tool_call>get_wea"], self.TOOLS)
        self.assertEqual("".join(e.text for e in evs if e.kind == "content"), "Text <tool_call>get_wea")

    def test_a_body_not_in_the_grammar(self):
        with self.assertRaisesRegex(ValueError, "malformed tool call"):
            parse_glm_tool_call('{"name": "f", "arguments": {}}')
        self.assertEqual(glm_call_end('{"name": "f"}</tool_call>'), len('{"name": "f"}'))
        self.assertEqual(glm_call_end("f<arg_key>a</arg_key><arg_value>1</arg_value>"), -1)
        self.assertEqual(glm_call_end("f<arg_key>a</arg_key><arg_value>1</arg_value></tool_call>x"),
                         len("f<arg_key>a</arg_key><arg_value>1</arg_value>"))


# ------------------------------------------------------------------------------------------------ over HTTP
class OverHttp(unittest.TestCase):
    """The three APIs, whole and streamed, with GLM output from the mock engine: one token per byte, so every tag and
    every multi-byte character arrives split."""

    def setUp(self):
        self.svc, self.engine, self.tok = glm_service([CALLS])
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def script(self, *texts):
        self.engine.scripts = [self.tok.encode(t, parse_special=True) for t in texts]
        self.engine.script, self.engine.turns = self.engine.scripts[0], 0

    def post(self, path, body):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                c.request("POST", path, body=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
                r = c.getresponse()
                raw = r.read().decode()
            self.assertEqual(r.status, 200, raw[:500])
            if r.getheader("Content-Type") == "text/event-stream":
                return [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
            return json.loads(raw)
        finally:
            c.close()

    def openai(self, stream, **extra):
        body = {"model": "glm-5.3-flash", "stream": stream, "tools": OPENAI_TOOLS,
                "messages": [{"role": "user", "content": "Weather in Kyoto and Paris?"}], **extra}
        r = self.post("/v1/chat/completions", body)
        if not stream:
            c = r["choices"][0]
            return (c["finish_reason"], c["message"].get("reasoning_content"), c["message"].get("content"),
                    [(t["function"]["name"], json.loads(t["function"]["arguments"]))
                     for t in c["message"].get("tool_calls") or []], r["model"])
        calls, reasoning, content = {}, "", ""
        for e in r:
            if not e.get("choices"):
                continue
            d = e["choices"][0]["delta"]
            reasoning += d.get("reasoning_content") or ""
            content += d.get("content") or ""
            for t in d.get("tool_calls") or []:
                slot = calls.setdefault(t["index"], ["", ""])
                slot[0] += (t.get("function") or {}).get("name") or ""
                slot[1] += (t.get("function") or {}).get("arguments") or ""
        last = [e for e in r if e.get("choices")][-1]
        return (last["choices"][0]["finish_reason"], reasoning or None, content or None,
                [(n, json.loads(a)) for n, a in (calls[i] for i in sorted(calls))], last["model"])

    def anthropic(self, stream, **extra):
        body = {"model": "glm-5.3-flash", "max_tokens": 4000, "stream": stream, "tools": ANTHROPIC_TOOLS,
                "messages": [{"role": "user", "content": "Weather in Kyoto and Paris?"}], **extra}
        r = self.post("/v1/messages", body)
        if not stream:
            blocks = r["content"]
            return (r["stop_reason"], "".join(b["thinking"] for b in blocks if b["type"] == "thinking") or None,
                    "".join(b["text"] for b in blocks if b["type"] == "text") or None,
                    [(b["name"], b["input"]) for b in blocks if b["type"] == "tool_use"])
        blocks = {}
        stop = None
        for e in r:
            if e["type"] == "content_block_start":
                blocks[e["index"]] = dict(e["content_block"], partial="")
            elif e["type"] == "content_block_delta":
                b, d = blocks[e["index"]], e["delta"]
                b["partial"] += d.get("partial_json") or d.get("thinking") or d.get("text") or ""
            elif e["type"] == "message_delta":
                stop = e["delta"]["stop_reason"]
        bl = [blocks[i] for i in sorted(blocks)]
        return (stop, "".join(b["partial"] for b in bl if b["type"] == "thinking") or None,
                "".join(b["partial"] for b in bl if b["type"] == "text") or None,
                [(b["name"], json.loads(b["partial"])) for b in bl if b["type"] == "tool_use"])

    def responses(self, stream, **extra):
        body = {"model": "glm-5.3-flash", "stream": stream, "tools": RESPONSES_TOOLS,
                "input": "Weather in Kyoto and Paris?", **extra}
        r = self.post("/v1/responses", body)
        resp = r[-1]["response"] if stream else r
        if stream:
            deltas = {}
            for e in r:
                if e["type"] == "response.function_call_arguments.delta":
                    deltas[e["item_id"]] = deltas.get(e["item_id"], "") + e["delta"]
            for o in resp["output"]:
                if o["type"] == "function_call":
                    self.assertEqual(json.loads(deltas[o["id"]]), json.loads(o["arguments"]))
        out = resp["output"]
        return (resp["status"],
                "".join(c["text"] for o in out if o["type"] == "reasoning" for c in o["content"]) or None,
                "".join(c["text"] for o in out if o["type"] == "message" for c in o["content"]) or None,
                [(o["name"], json.loads(o["arguments"])) for o in out if o["type"] == "function_call"])

    def test_tool_calls_on_all_three_apis(self):
        calls = [("get_weather", CALL_ARGS[0]), ("get_weather", CALL_ARGS[1])]
        for stream in (False, True):
            with self.subTest(api="openai", stream=stream):
                self.assertEqual(self.openai(stream), ("tool_calls", "Two cities.", "Checking both.", calls,
                                                       "glm-5.3-flash"))
            with self.subTest(api="anthropic", stream=stream):
                self.assertEqual(self.anthropic(stream), ("tool_use", "Two cities.", "Checking both.", calls))
            with self.subTest(api="responses", stream=stream):
                self.assertEqual(self.responses(stream), ("completed", "Two cities.", "Checking both.", calls))
        self.assertTrue(self.tok.decode(self.engine.last_prompt).endswith(
            "<|user|>Weather in Kyoto and Paris?<|assistant|><think>"))

    def test_each_end_token_ends_the_turn(self):
        """<|endoftext|>, <|user|> and <|observation|> each end the answer; nothing after them is sent, and ChatML's
        <|im_end|> (plain text in GLM's vocabulary) is not an end."""
        self.assertEqual(self.svc.stop_ids, {self.tok.encode(t, parse_special=True)[0]
                                             for t in ("<|endoftext|>", "<|user|>", "<|observation|>")})
        for end in ("<|endoftext|>", "<|user|>", "<|observation|>"):
            self.script("Brief.</think>The answer<|im_end|> is 4." + end + "junk after the end")
            for stream in (False, True):
                with self.subTest(end=end, stream=stream):
                    self.assertEqual(self.openai(stream, tools=None)[:3], ("stop", "Brief.", "The answer<|im_end|> is 4."))
                    self.assertEqual(self.anthropic(stream, tools=None)[:3], ("end_turn", "Brief.", "The answer<|im_end|> is 4."))
        self.assertEqual(self.svc.history[-1]["finish"], "stop")

    def test_thinking_off(self):
        """effort none: the prompt closes the thinking, and the answer is all content (none of it is reasoning)."""
        self.script("Direct answer.<|user|>")
        for kw in ({"reasoning_effort": "none"}, {"chat_template_kwargs": {"enable_thinking": False}}):
            self.assertEqual(self.openai(False, tools=None, **kw)[:3], ("stop", None, "Direct answer."))
            self.assertTrue(self.tok.decode(self.engine.last_prompt).endswith("<|assistant|><think></think>"))
        self.assertEqual(self.anthropic(True, tools=None, thinking={"type": "disabled"})[:3],
                         ("end_turn", None, "Direct answer."))
        self.assertEqual(self.responses(False, tools=None, reasoning={"effort": "none"})[1:3], (None, "Direct answer."))
        self.script("Thought.</think>Low.<|user|>")
        self.openai(False, tools=None, reasoning_effort="medium")
        self.assertIn("<|system|>Reasoning Effort: High<|user|>", self.tok.decode(self.engine.last_prompt))

    def test_the_prompt_continues_where_the_model_stopped(self):
        """The next request (the call, its result) renders the model's own output back exactly, up to its
        <|observation|>: the engine's conversation cache reuses the whole first prompt and answer."""
        call = ("Look.</think>Checking.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>"
                "<arg_key>days</arg_key><arg_value>3</arg_value></tool_call>")
        self.script(call + "<|observation|>", "Read it.</think>Sunny.")
        user = {"role": "user", "content": "Weather in Paris?"}
        r = self.post("/v1/chat/completions", {"model": "m", "tools": OPENAI_TOOLS, "messages": [user]})
        first = self.tok.decode(self.engine.last_prompt)
        msg = r["choices"][0]["message"]
        tc = msg["tool_calls"][0]
        self.post("/v1/chat/completions", {"model": "m", "tools": OPENAI_TOOLS, "messages": [
            user, msg, {"role": "tool", "tool_call_id": tc["id"], "content": "sunny"}]})
        second = self.tok.decode(self.engine.last_prompt)
        self.assertTrue(second.startswith(first + call + "<|observation|>"), second[-400:])
        self.assertTrue(second.endswith("<|observation|><tool_response>sunny</tool_response><|assistant|><think>"))


# ------------------------------------------------------------------------------------------------ the start
def tiny_glm_tokenizer(d: Path) -> Path:
    """A GLM-shaped tokenizer directory (tokenizer.json pre glm4, GLM's specials, byte tokens, no merges) as
    tools/strata_tokenizer.py's extract writes it, without a chat_template.jinja (the server's fallback is used)."""
    from strata_tokenizer import bytes_to_unicode
    tokens = list(bytes_to_unicode().values())
    control = ["<|endoftext|>", "[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"]
    user = ["<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>",
            "</arg_value>"]
    types = [1] * len(tokens) + [3] * len(control) + [4] * len(user)
    tokens += control + user
    d.mkdir(parents=True, exist_ok=True)
    (d / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(tokens)}), encoding="utf-8")
    (d / "merges.txt").write_text("", encoding="utf-8")
    (d / "token_type.json").write_text(json.dumps(types))
    ids = {t: i for i, t in enumerate(tokens)}
    (d / "tokenizer.json").write_text(json.dumps({"model": "gpt2", "pre": "glm4", "special_ids": {
        "tokenizer.ggml.eos_token_id": ids["<|endoftext|>"], "tokenizer.ggml.eot_token_id": ids["<|user|>"],
        "tokenizer.ggml.eom_token_id": ids["<|observation|>"]}}))
    return d


class TheStart(unittest.TestCase):
    """python -m serve.server --engine mock with a GLM pack's tokenizer: the family, model name, template and end
    token come from the pack."""

    def start(self, cfg: dict, script: str):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        cfgp = Path(self.tmp.name) / "strata-glm.json"
        cfgp.write_text(json.dumps(cfg))
        proc = subprocess.Popen([sys.executable, "-m", "serve.server", "--engine", "mock", "--config", str(cfgp),
                                 "--port", str(port), "--script", script], cwd=ROOT,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.addCleanup(lambda: (proc.kill(), proc.wait(), proc.stdout.close()))
        base = f"http://127.0.0.1:{port}"
        for _ in range(300):
            try:
                with urllib.request.urlopen(base + "/v1/models", timeout=2) as r:
                    return base, json.loads(r.read()), proc
            except OSError:
                if proc.poll() is not None:
                    self.fail(proc.stdout.read())
                time.sleep(0.1)
        self.fail("the server did not start")

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tokdir = tiny_glm_tokenizer(Path(self.tmp.name) / "tokenizer")

    def chat(self, base, body):
        req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def test_a_glm_config_end_to_end(self):
        base, models, proc = self.start({"tokenizer": str(self.tokdir), "model_name": "glm-5.3-flash"},
                                        "Hm.</think>Hello from GLM.")
        self.assertEqual([m["id"] for m in models["data"]][:1], ["glm-5.3-flash"])
        r = self.chat(base, {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual((r["model"], r["choices"][0]["message"]["content"], r["choices"][0]["finish_reason"],
                          r["choices"][0]["message"]["reasoning_content"]),
                         ("glm-5.3-flash", "Hello from GLM.", "stop", "Hm."))
        with urllib.request.urlopen(base + "/props", timeout=5) as p:
            self.assertIn("Reasoning Effort", json.loads(p.read())["chat_template"])   # the GLM fallback template

    def test_the_default_name_and_a_tool_call(self):
        call = "</think><tool_call>get_weather<arg_key>days</arg_key><arg_value>2</arg_value></tool_call><|observation|>"
        base, models, proc = self.start({"tokenizer": str(self.tokdir)}, call)
        self.assertEqual(models["data"][0]["id"], "glm-5.3-flash")
        r = self.chat(base, {"model": "x", "tools": OPENAI_TOOLS, "messages": [{"role": "user", "content": "hi"}]})
        tc = r["choices"][0]["message"]["tool_calls"]
        self.assertEqual([(t["function"]["name"], json.loads(t["function"]["arguments"])) for t in tc],
                         [("get_weather", {"days": 2})])


# ------------------------------------------------------------------------------------------------ real packs
def _dir(env, *globs):
    for c in [os.environ.get(env, "")] + [str(p) for g in globs for p in ROOT.glob(g)]:
        if c and (Path(c) / "vocab.json").exists():
            return Path(c)
    return None


GLM_TOK = _dir("STRATA_GLM_TOKENIZER")
QWEN_TOK = _dir("STRATA_TOKENIZER", "packs/*/tokenizer", "pack/*/tokenizer")


@unittest.skipIf(GLM_TOK is None, "no GLM pack tokenizer here (STRATA_GLM_TOKENIZER)")
class RealGlmTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import strata_tokenizer as ST
        cls.tok = ST.Tokenizer.from_dir(GLM_TOK)

    def test_ids(self):
        tok = self.tok
        self.assertIs(profile_for(tok), GLM)
        self.assertEqual(GLM.stop_ids(tok), {154820, 154827, 154829})
        self.assertEqual(GLM.engine_args(tok, []), ["--eos-ids", "154820,154827,154829", "--turn-token", "154828"])
        self.assertEqual(GLM.engine_args(tok, ["--eos-ids", "1", "--turn-token", "2"]), [])
        self.assertGreater(len(tok.encode("<|im_end|>", parse_special=True)), 1)   # why <|im_end|> could not stay
        svc, _, _ = glm_service(tok=tok)
        ids = svc.encode_prompt([{"role": "user", "content": "hi"}], None, {"enable_thinking": False})
        self.assertEqual(ids[:3], [154822, 154824, 154826])                        # [gMASK] <sop> <|system|>
        self.assertEqual(ids[-3:], [154828, 154841, 154842])                       # <|assistant|> <think> </think>
        for case in GOLDEN:
            self.assertEqual(tok.decode(tok.encode(case["rendered"], parse_special=True)), case["rendered"])

    def test_a_call_through_the_server(self):
        svc, eng, tok = glm_service([CALLS], tok=self.tok)
        httpd = serve(svc, port=0)
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions",
                                         data=json.dumps({"model": "m", "tools": OPENAI_TOOLS, "stream": True,
                                                          "messages": [{"role": "user", "content": "w"}]}).encode(),
                                         headers={"Content-Type": "application/json"})
            with contextlib.redirect_stdout(io.StringIO()), urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode()
        finally:
            httpd.shutdown()
            httpd.server_close()
        chunks = [json.loads(x[6:]) for x in raw.splitlines() if x.startswith("data: {")]
        args = {}
        for c in chunks:
            for t in (c.get("choices") or [{}])[0].get("delta", {}).get("tool_calls") or []:
                args[t["index"]] = args.get(t["index"], "") + t["function"].get("arguments", "")
        self.assertEqual([json.loads(args[i]) for i in sorted(args)], CALL_ARGS)
        self.assertEqual([c for c in chunks if c.get("choices")][-1]["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(svc.history[-1]["output_tokens"], len(self.tok.encode(CALLS, parse_special=True)))


@unittest.skipIf(QWEN_TOK is None, "no Qwen pack tokenizer here (STRATA_TOKENIZER)")
class RealQwenTokenizer(unittest.TestCase):
    def test_qwen_is_unchanged(self):
        import strata_tokenizer as ST
        tok = ST.Tokenizer.from_dir(QWEN_TOK)
        self.assertIs(profile_for(tok), QWEN)
        self.assertEqual(QWEN.stop_ids(tok), {248044, 248046})
        self.assertEqual(QWEN.engine_args(tok, []), [])
        svc = Service(MockEngine(tok, "ok"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.assertEqual((svc.profile, svc.model, svc.stop_ids), (QWEN, "qwen3.8-flash-next", {248044, 248046}))


class ByteTokenizers(unittest.TestCase):
    def test_families(self):
        self.assertIs(profile_for(ByteTokenizer()), QWEN)
        self.assertIs(profile_for(GlmByteTokenizer()), GLM)
        self.assertIs(profile_for(ByteTokenizer(), "glm5-next"), GLM)
        with self.assertRaisesRegex(ValueError, "arch"):
            profile_for(ByteTokenizer(), "llama")
        self.assertEqual(Service(MockEngine(ByteTokenizer(), "x"), ByteTokenizer(), TEMPLATE).stop_ids, {257, 258})
        with self.assertRaisesRegex(ValueError, r"no single token for <\|user\|>, <\|observation\|>: not a glm5"):
            GLM.stop_ids(ByteTokenizer())          # a ChatML tokenizer is not taken for a GLM pack


if __name__ == "__main__":
    unittest.main()
