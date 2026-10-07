"""serve/models.py - what the server does differently per model family, chosen from the pack.

Two families: Qwen3.8-Flash-Next (GGUF architecture `qwen4exp`, ChatML) and GLM-5.3-Flash (`glm5-next`).  The engine
boundary (token ids in, text out) is the same; what differs is text: the chat template's specials, the tokens that end
a turn, the thinking levels, how thinking is turned off, and the tool-call grammar.  A profile holds those, and the
server reads them instead of Qwen's constants.  QWEN keeps every value the server had before GLM, byte for byte.

The profile is chosen from the pack's tokenizer (tokenizer.json `pre`: `glm4` is GLM, anything else Qwen, as
tools/strata_tokenizer.py's PRE_TYPES), or from the config's optional "arch".
"""
from __future__ import annotations

from dataclasses import dataclass

from serve.frontend import (CALL_START, GLM_EFFORTS, QWEN_CALL_OPENING, QWEN_EFFORTS, Efforts, GlmOutputParser,
                            OutputParser)

# Qwen (ChatML)
IM_END = "<|im_end|>"
IMAGE_PAD = "<|image_pad|>"
# #458: the trailing effort turn ("effort_position": "end"); low is the template's own sentence, medium (which the
# template says nothing for: no xhigh sentence is medium there) says it, since the xhigh one stays at the top
EFFORT_TURN = "<|im_start|>system\n{}<|im_end|>\n"
EFFORT_TEXT = {"low": "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
                      "conclusion without unnecessary elaboration.",
               "medium": "Reasoning effort is set to medium. Think as much as the task needs, without unnecessary "
                         "elaboration.",
               # the template's own xhigh sentence (chat_template.jinja), which #728's recovery swaps for the low one
               "xhigh": "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
                        "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
                        "clarity in the final answer."}
VISION_START = "<|vision_start|>"

# llama.cpp f1cee99 src/llama-vocab.cpp L2890-2915 (end-of-generation by token text) and L2944-2957 (the GGUF's
# eos / eot / eom ids are end-of-generation too): for GLM-5.3-Flash <|endoftext|> 154820 (eos), <|user|> 154827 (eot),
# <|observation|> 154829 (eom)
END_ID_KEYS = ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id")


@dataclass(frozen=True)
class ModelProfile:
    arch: str                       # the GGUF's general.architecture
    model_name: str                 # the API's model name when the config gives none
    template: str                   # serve/<file>: the chat template when the pack has none
    end_tokens: tuple               # the texts of the tokens that end the model's turn
    mock_end: str                   # what MockEngine appends to a scripted reply
    image_pad: str                  # the template's image placeholder (one per image token after encoding)
    vision_start: str               # the token right before it
    efforts: Efforts                # the thinking levels (serve.frontend)
    shared_efforts: tuple           # the values POST /settings accepts as everyone's reasoning_effort
    effort_turn: str                # #458's trailing effort turn, "{}" = effort_text[level]
    effort_text: dict
    tail_role: str | None           # the token --tail-role-token names (None: effort_position "end" is not offered)
    close_think: bool               # thinking off: the server closes the generation prompt's <think> (no kwarg)
    parser: type                    # the output parser (reasoning, content, tool calls)
    end_ids_from_vocab: bool        # the end ids are each one token (checked) plus the GGUF's eos/eot/eom ids
    turn_token: str | None          # passed as --turn-token (None: the engine's default, <|im_start|>)
    pass_end_ids: bool              # passed as --eos-ids (the engine's default is Qwen's 248044,248046)
    call_opening: tuple             # a forced call's opening text (tool_choice): any tool, a named one ("{}")

    def stop_ids(self, tok) -> set[int]:
        """The ids the server stops at."""
        encoded = [tok.encode(s, parse_special=True) for s in self.end_tokens]
        if not self.end_ids_from_vocab:             # Qwen: as it always was
            return set(sum(encoded, []))
        bad = [s for s, e in zip(self.end_tokens, encoded) if len(e) != 1]
        if bad:
            raise ValueError(f"the tokenizer has no single token for {', '.join(bad)}: not a {self.arch} pack")
        special = getattr(tok, "special_ids", None) or {}
        return {e[0] for e in encoded} | {int(special[k]) for k in END_ID_KEYS if k in special}

    def engine_args(self, tok, args: list[str]) -> list[str]:
        """What the engine needs beyond the config's args for this family (none for Qwen): the end ids and the turn
        token, unless the config's args already give them."""
        out = []
        if self.pass_end_ids and "--eos-ids" not in args:
            out += ["--eos-ids", ",".join(str(i) for i in sorted(self.stop_ids(tok)))]
        if self.turn_token and "--turn-token" not in args:
            ids = tok.encode(self.turn_token, parse_special=True)
            if len(ids) == 1:
                out += ["--turn-token", str(ids[0])]
        return out


QWEN = ModelProfile(
    arch="qwen4exp", model_name="qwen3.8-flash-next", template="chat_template.jinja",
    end_tokens=(IM_END, "<|endoftext|>"), mock_end=IM_END, image_pad=IMAGE_PAD, vision_start=VISION_START,
    efforts=QWEN_EFFORTS, shared_efforts=("none", "low", "medium", "high"),
    effort_turn=EFFORT_TURN, effort_text=EFFORT_TEXT, tail_role="system", close_think=False, parser=OutputParser,
    end_ids_from_vocab=False, turn_token=None, pass_end_ids=False, call_opening=QWEN_CALL_OPENING)

# GLM-5.3-Flash (glm_chat_template.jinja, the pack's chat_template.jinja): [gMASK]<sop>, then
# <|system|>Reasoning Effort: Max|High|Low, the tools as a <|system|> turn, <|user|> / <|assistant|> / <|observation|>
# turns without newlines, an assistant turn as <think>...</think>answer, and the generation prompt <|assistant|><think>.
#   * thinking off: the template has no enable_thinking; the server ends the prompt with <|assistant|><think></think>,
#     the form the template itself writes for an assistant turn without reasoning.
#   * the end of a turn: <|endoftext|>, <|user|> (the model hands the turn back) and <|observation|> (after tool calls).
#   * --turn-token <|assistant|>: the engine checkpoints the conversation in front of the new answer's turn.
#   * a forced call (tool_choice) opens as `<tool_call>` (any tool) or `<tool_call>NAME` (a named one).
#   * effort_position "end" is not offered (tail_role None): the engine's --tail-role-token rule looks for the role
#     token right after --turn-token (ChatML's "<|im_start|>system"), and GLM's turns open with the role token itself.
#     render_prompt still renders the tail turn (<|system|>Reasoning Effort: Low before <|assistant|>) for when it is.
GLM = ModelProfile(
    arch="glm5-next", model_name="glm-5.3-flash", template="glm_chat_template.jinja",
    end_tokens=("<|endoftext|>", "<|user|>", "<|observation|>"), mock_end="<|user|>",
    image_pad="<|image|>", vision_start="<|begin_of_image|>",
    efforts=GLM_EFFORTS, shared_efforts=("none", "low", "medium", "high", "max"),
    effort_turn="<|system|>Reasoning Effort: {}", effort_text={"low": "Low", "high": "High"},
    tail_role=None, close_think=True, parser=GlmOutputParser,
    end_ids_from_vocab=True, turn_token="<|assistant|>", pass_end_ids=True,
    call_opening=(CALL_START, CALL_START + "{}"))

PROFILES = {p.arch: p for p in (QWEN, GLM)}


def profile_for(tok, arch: str | None = None) -> ModelProfile:
    """The family of the pack whose tokenizer is `tok` (its `pre`: glm4 -> GLM, else Qwen), or the one `arch` (the
    config's "arch") names.  ValueError for an arch not listed here."""
    if arch:
        if arch not in PROFILES:
            raise ValueError(f'"arch" must be one of {", ".join(PROFILES)}, not {arch!r}')
        return PROFILES[arch]
    return GLM if getattr(tok, "pre", None) == "glm4" else QWEN
