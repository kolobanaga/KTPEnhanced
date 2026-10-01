# KTPEnhanced

A SillyTavern extension for Kimi/GLM models that enables client-side handling of `reasoning_content` without modifying the SillyTavern server.

This is a fork of [Rurijian/KimiThinkingPrefill](https://github.com/Rurijian/KimiThinkingPrefill), with additional fixes for preserved reasoning and context-budget handling.

## Features

* Client-side equivalent of the Kimi/GLM thinking prefill server patch.
* Re-attaches stored `extra.reasoning` as `reasoning_content` for previous assistant messages, with selectable strategies to trade reasoning tokens against context budget.
* Correctly aligns preserved reasoning when SillyTavern trims the beginning of the chat history.
* Optional post-attachment context trimming to keep the request within the configured SillyTavern context budget.
* Optional reasoning prefill support.
* No modifications to the SillyTavern server.

## Additional fixes in this fork

### Preserved reasoning and context trimming

The original extension matched chat messages from the beginning of the conversation. When SillyTavern removed older messages because of context limits, reasoning from the beginning of the chat could be attached to newer messages incorrectly.

This fork matches assistant messages from the end of the conversation, keeping the remaining history aligned when older messages are trimmed.

### Context budget

SillyTavern normally calculates its context budget before this extension re-attaches `reasoning_content`. Since preserved reasoning adds tokens to the outgoing request, it could otherwise push the final request beyond the configured context size.

The optional **Trim to budget** setting performs an additional budget check after reasoning has been attached and removes the oldest chat-history messages when necessary.

This uses SillyTavern's configured context size and response-token limit as the target budget.

## Settings

### Prior reasoning to the API

Re-attaches stored `extra.reasoning` from previous assistant messages as `reasoning_content`.

This is useful for models that expect previous reasoning to be preserved between turns, but reasoning is billed as input tokens and can crowd out real chat history, so the dropdown **Prior reasoning sent to the API** selects how much of it is sent:

| Strategy | Behaviour |
| --- | --- |
| `None` | Nothing is re-attached. |
| `All` | Every stored reasoning is re-attached. Most input tokens and the most pressure on the context budget. |
| `Last K` | Only the newest `K` reasonings are re-attached; older assistant turns are sent without `reasoning_content` at all. |
| `Last K + Omitted` | The newest `K` are sent verbatim; every older reasoning is replaced with a placeholder (default `[earlier thinking elided to fit the context window]`, editable), so the model still sees that thinking happened there. |

`K` is configured in the **How many latest reasonings to send** field (default `3`), shown only for the two windowed strategies. The placeholder is configured in the **Placeholder for omitted reasoning** field, shown only for `Last K + Omitted`; clearing it sends nothing for those turns, which makes that strategy behave like `Last K`.

Notes:

* `K` counts *stored reasonings*, not assistant turns, so a stretch of turns without reasoning (for example from a non-reasoning model) does not silently consume the window.
* The default placeholder is worded as a note about the context budget, not as a redaction. It is written into the model's own `reasoning_content`, so wording like "omitted" or "withheld" reads as *reasoning hidden for policy reasons* — the convention providers use for encrypted reasoning — instead of "this turn's thinking was compressed away". It also never claims that no reasoning happened, which could make the model re-derive a turn it actually reasoned through.
* Only the outgoing request payload is modified. The stored `extra.reasoning` in the chat is never rewritten, so switching strategies back and forth is lossless and re-sends the full reasoning.
* Nothing is re-attached for turns whose reasoning was never stored. The model and the request source have to return reasoning in the first place — the **Request model reasoning** setting controls whether the API is asked for it at all (and whether it is displayed), and a non-reasoning model stores none regardless of that setting.
* Upgrading from the old "Send all prior assistant reasoning" checkbox: checked maps to `All`, unchecked maps to `None`.

#### Model contracts behind the strategies

The windowed strategies are a deliberate trade-off, not a free win. Providers document conflicting requirements for historical reasoning, and the three models in the default **Apply to models matching** filter (`kimi,moonshot,GLM`) are the strictest of them.

| Model | Thinking control | Cross-turn requirement |
| --- | --- | --- |
| `GLM-5.3`, `GLM-5.3-Flash` | always on — `thinking.type: "disabled"` fails the request; depth via `reasoning_effort` (`low`/`high`/`max`) | Preserved Thinking (`clear_thinking: false`) requires the historical `reasoning_content` **complete, unmodified and in its original order**; editing, trimming or reordering degrades quality and cache hit rate. On by default on the Coding Plan endpoint. |
| `kimi-k3` | always on; the `thinking` parameter should not be sent; depth via top-level `reasoning_effort` | Preserved Thinking is **always on**: every historical assistant message must be returned as-is, and the docs state this is required for multi-turn *and* tool-call loops. |
| `kimi-k2.6` / `kimi-k2.7-code` | `thinking.type` + `keep` (`null` by default / `"all"`); `k2.7-code` accepts only `{"type":"enabled","keep":"all"}` | Cross-turn preservation follows `keep`. `kimi-k2.5` has no Preserved Thinking at all. |
| `deepseek-v4-pro` / `deepseek-v4-flash` | on by default | The gate is the `tools` parameter on the **request**, not whether a tool was actually called. With `tools` present, `reasoning_content` of **all previous turns** must be passed back, or the API returns `400 The reasoning_content in the thinking mode must be passed back to the API`. Without `tools` the field is ignored entirely and never concatenated into the context. |
| `deepseek-reasoner` / R1 (legacy) | — | The opposite rule: `reasoning_content` must **not** be passed back, or the request fails. |
| `qwen3.8` (DashScope) | `enable_thinking` | `preserve_thinking: true` (default `false`). The model is trained to keep its reasoning blocks in the conversation; when they are stripped it re-derives conclusions it had already reached — reportedly performing like a lower effort tier while spending *more* tokens. |
| `MiMo V2.5` | — | Same 400 as DeepSeek V4 when the request carries tools and a previous turn lost its `reasoning_content`. |
| OpenRouter, user-defined tools | `reasoning.effort` / `max_tokens` / `exclude` | Documented: echo reasoning back for tool use via `message.reasoning` (or the `reasoning_content` alias), or `message.reasoning_details` for encrypted/summarized blocks. The whole sequence of consecutive reasoning blocks must match what the model produced in the original request and cannot be rearranged or modified. |
| OpenRouter, server tools | server-side execution, `openrouter:*` entries in the `tools` array | **Undocumented.** The [server tools guide](https://openrouter.ai/docs/guides/features/server-tools) never mentions reasoning. Observed in request logs: with server tools, `reasoning_content` from the earlier history is discarded and only the reasoning produced during that request's server-side tool loop survives — the opposite of the documented advice for user-defined tools. Treat as empirical behaviour, not a contract. |
| `minimax-M2` (legacy) | always on | Thinking is inline `<think>` tags inside `content`, not a separate field. |
| Anthropic / Gemini | separate request shapes | Reasoning is not a `reasoning_content` field: Anthropic uses signed thinking blocks and `clear_thinking_20251015` (keep the last N thinking turns, clearing older ones, which invalidates the cache from the clearing point); Gemini uses `thinkingConfig`. Out of scope for this extension. |

Practical consequences for the strategies above:

* `All` is the only contract-compliant option for `kimi-k3` and for GLM's Preserved Thinking. `Last K` is literally a truncation and `Last K + Omitted` is literally an edit, which is what those docs warn against.
* DeepSeek V4 (and MiMo) can return 400 when a request carries tools and a previous turn lost its `reasoning_content`. **This extension cannot trigger that**: the handler bails out before attaching anything whenever tools are in play. The check predates the strategies below and also covers tool messages and `tool_calls` already present in the history, not just a `tools` array in the request — so it is stricter than the documented DeepSeek rule. The price is that the strategies and the budget trimming are inert on tool requests too. Requests without tools are unaffected, and there DeepSeek ignores the field rather than concatenating it, so the window is free.
* OpenRouter documents preserving reasoning for user-defined tools, but its server tools appear to work the other way round (previous history dropped). There is an explicit control for that — `reasoning.context: "current_turn"` versus `"all_turns"` — though OpenRouter lists it as supported only by OpenAI GPT-5.6 and newer, so it is not an available lever for the `reasoning_content` lineage this extension targets.
* The economy argument inverts for models trained on preserved thinking — dropping reasoning can cost more tokens than it saves, because the model re-derives what it already concluded. A window is most defensible where the context is dominated by the chat itself rather than by reasoning continuity.
* Field naming differs per provider (`reasoning_content`, `reasoning`, `reasoning_details[]`, inline `<think>`), and sending the wrong one can also be a hard 400 — e.g. GLM-4.7 on Cerebras rejects `reasoning_content` and expects `reasoning`.

> **Caveat — not validated for roleplay.** All of the above is transcribed from vendor documentation that was written for coding and agentic workloads. This extension was designed around roleplay, where the priorities differ: a long narrative history, little or no tool use, and coherence of the *story* rather than of the reasoning chain. The practical effect of `Last K` / `Last K + Omitted` on those models in an RP chat is therefore **unverified** and needs empirical checking against your own chats before treating any of it as settled — measure output quality, not just token counts.

### Trim to budget

After attaching preserved reasoning, checks the estimated token usage again and removes the oldest user/assistant history messages until the request fits the configured context budget.

Enabled by default.

Note that this is skipped entirely on requests that involve tools: the handler returns before reasoning is attached, so neither the strategies nor this trimming run there. That guard is inherited from the original thinking-prefill patch and has never been relaxed.

### Thinking prefill

Optionally injects a `reasoning_content` prefill for the current generation. This is separate from preserved reasoning history.

## Important

Preserved reasoning increases prompt size. A long reasoning history can consume a significant part of the available context. The windowed strategies exist to counter this — see [Model contracts behind the strategies](#model-contracts-behind-the-strategies) for what each provider documents, and for what is still unverified in a roleplay context.

Token counts calculated by the extension are estimates and may differ from the exact token accounting performed by the model provider.

## Credits

Original extension by [Rurijian](https://github.com/Rurijian).