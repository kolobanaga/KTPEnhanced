# KTPEnhanced

A SillyTavern extension for Kimi/GLM models that enables client-side handling of `reasoning_content` without modifying the SillyTavern server.

This is a fork of [Rurijian/KimiThinkingPrefill](https://github.com/Rurijian/KimiThinkingPrefill), with additional fixes for preserved reasoning and context-budget handling.

## What is Preserved Thinking / Multi-turn Reasoning?

Different labs and providers use different names for closely related behaviour, including **Preserved Thinking**, **Multi-turn Reasoning**, **Interleaved Thinking**, **Reasoning Persistence**, and **CoT Persistence**. The exact API contract varies by provider, but the shared idea is that a model's reasoning is carried forward from one turn to the next instead of being discarded after each response.

In a typical chat-completion request, the conversation contains alternating user messages and assistant messages. An assistant message may have both a user-facing answer (`content`) and a separate reasoning field (`reasoning_content`). For multi-turn reasoning, the client sends the previous assistant reasoning back as part of the conversation history, alongside the corresponding answer:

```js
{
  messages: [
    { role: 'user', content: 'Turn 1.' },
    { role: 'assistant', content: 'C1', reasoning_content: 'R1' },
    { role: 'user', content: 'Turn 2.' },
    { role: 'assistant', content: 'C2', reasoning_content: 'R2' },
    { role: 'user', content: 'Turn 3.' },
    { role: 'assistant', content: 'C3', reasoning_content: 'R3' },
    { role: 'user', content: 'Turn 4.' },
    { role: 'assistant', content: 'C4', reasoning_content: 'R4' },
    { role: 'user', content: 'Turn 5.' },
    { role: 'assistant', content: 'C5', reasoning_content: 'R5' },
    { role: 'user', content: 'Turn 6.' },
    { role: 'assistant', content: 'C6', reasoning_content: 'R6' },
    { role: 'user', content: 'Turn 7.' },
    { role: 'assistant', content: 'C7', reasoning_content: 'R7' },
    { role: 'user', content: 'Turn 8.' },
    { role: 'assistant', content: 'C8', reasoning_content: 'R8' },
    { role: 'user', content: 'Turn 9.' }
  ],
  // other request parameters ...
}
```

The expected next response is:

```js
{ role: 'assistant', content: 'C9', reasoning_content: 'R9' }
```

A failure mode is that the model starts reasoning from scratch instead of continuing the existing reasoning thread, for example:

```js
{
  role: 'assistant',
  content: 'C9',
  reasoning_content: 'We need answer? The user is giving "Turn 1." then assistant "C1" etc. ...'
}
```

The point is not merely to preserve the visible conversation. The model should be able to continue both the **conversation thread** and the **reasoning thread**. Replaying the prior reasoning can matter for models trained or configured to rely on it across turns; dropping or modifying it may cause the model to re-derive earlier steps, lose continuity, or spend extra reasoning tokens.

This is not a universal behaviour of every model or API. Some providers require previous reasoning to be replayed exactly and in order; others ignore it unless specific request parameters are present; some reject it. The provider-specific notes below describe those differences. The extension can only re-attach reasoning that SillyTavern actually stored, and the model/provider must support the relevant contract.

## Features

* Client-side equivalent of the Kimi/GLM thinking prefill server patch.
* Re-attaches stored `extra.reasoning` as `reasoning_content` for previous assistant messages, with selectable strategies to trade reasoning tokens against context budget.
* Aligns preserved reasoning to the correct outgoing assistant messages when SillyTavern trims, inserts, drops, or edits messages, using a longest common subsequence (LCS) match rather than a greedy end-based match.
* Optional post-attachment context trimming to keep the request within the configured SillyTavern context budget.
* Optional multi-turn reasoning for DeepSeek, which otherwise discards historical `reasoning_content` — see [Multi-turn reasoning on DeepSeek](#multi-turn-reasoning-on-deepseek).
* Optional reasoning prefill support.
* No modifications to the SillyTavern server.

## Additional fixes in this fork

### Preserved reasoning and message alignment

SillyTavern stores reasoning alongside chat turns, but the outgoing API payload does not carry stable identifiers that connect each message to its original chat turn. Before the request is sent, SillyTavern can also transform the history: for example, older turns can be trimmed, messages can be inserted or removed, and message text can be changed.

The extension therefore reconstructs the correspondence by matching message text. The current implementation uses a **longest common subsequence (LCS)**: it finds the largest order-preserving set of matching chat turns and outgoing assistant messages. Both sequences are compared from oldest to newest, and the result preserves their original order.

A simple greedy match is unsafe. A payload message that was injected by a preset or another prompt feature — and is not an actual stored chat turn — may have text identical to a candidate turn. Greedily assigning that message can consume the candidate and prevent later messages from being paired correctly, causing reasoning to be attached to the wrong turn or omitted from the turns that need it most.

The LCS pairing prioritizes:

1. **The maximum number of valid pairs.** A longer order-preserving match always wins.
2. **Newer payload positions as a tie-breaker.** Among matches with the same number of pairs, the algorithm prefers the larger total of matched payload positions, biasing ambiguous matches towards the newest end. This is a sum-based tie-breaker, not a strict rule that every individual pair must be as far right as possible. If identical text appears multiple times, the text alone cannot always identify the true origin.

The reasoning window (`Last K` or `Last K + Omitted`) is applied **after** pairing. This is intentional: selecting a smaller window must not alter which messages form the best overall correspondence. It also helps prevent an injected old message at the end of the payload from stealing a candidate needed to match newer turns.

Before matching, messages that already have `reasoning_content` are left alone, and messages with no usable matching key are excluded. Matching keys account for relevant SillyTavern transformations and, in group chats, mirror the speaker-name prefix added to outgoing message content. If the text has been transformed in a way the extension cannot reliably reconstruct, it is safer for the message to remain unmatched than to attach reasoning to the wrong turn.

The LCS calculation uses O(n × m) time for `n` candidate chat turns and `m` outgoing assistant messages. Text keys are interned so the inner comparison uses integer IDs rather than repeatedly comparing long message bodies. If matching fails, the error is contained and the outgoing payload is left untouched.

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
* DeepSeek V4 (and MiMo) can return 400 when a request carries tools and a previous turn lost its `reasoning_content`. **The extension never triggers that by accident**: the handler bails out before attaching anything whenever tools are in play. The check predates the strategies below and also covers tool messages and `tool_calls` already present in the history, not just a `tools` array in the request — so it is stricter than the documented DeepSeek rule. The price is that the strategies and the budget trimming are inert on tool requests too. Requests without tools are unaffected, and there DeepSeek ignores the field rather than concatenating it, so the window is free. The one deliberate exception is [Multi-turn reasoning on DeepSeek](#multi-turn-reasoning-on-deepseek), which opts into sending `tools` on purpose.
* OpenRouter documents preserving reasoning for user-defined tools, but its server tools appear to work the other way round (previous history dropped). There is an explicit control for that — `reasoning.context: "current_turn"` versus `"all_turns"` — though OpenRouter lists it as supported only by OpenAI GPT-5.6 and newer, so it is not an available lever for the `reasoning_content` lineage this extension targets.
* The economy argument inverts for models trained on preserved thinking — dropping reasoning can cost more tokens than it saves, because the model re-derives what it already concluded. A window is most defensible where the context is dominated by the chat itself rather than by reasoning continuity.
* Field naming differs per provider (`reasoning_content`, `reasoning`, `reasoning_details[]`, inline `<think>`), and sending the wrong one can also be a hard 400 — e.g. GLM-4.7 on Cerebras rejects `reasoning_content` and expects `reasoning`.

> **Caveat — not validated for roleplay.** All of the above is transcribed from vendor documentation that was written for coding and agentic workloads. This extension was designed around roleplay, where the priorities differ: a long narrative history, little or no tool use, and coherence of the *story* rather than of the reasoning chain. The practical effect of `Last K` / `Last K + Omitted` on those models in an RP chat is therefore **unverified** and needs empirical checking against your own chats before treating any of it as settled — measure output quality, not just token counts.

### Multi-turn reasoning on DeepSeek

DeepSeek is the awkward entry in the table above. Without `tools` on the request it drops the `reasoning_content` of every turn before the last user message, so re-attaching it achieves nothing — the field is accepted and then silently ignored. With `tools` present it concatenates all of it instead. There is no documented way to ask for the second behaviour directly: `chat_template_kwargs.drop_thinking: false` is a vLLM chat-template passthrough, and on an NVIDIA NIM endpoint it reached the request without changing the reply.

The optional **Send a placeholder tool** setting exploits the side effect instead. It adds a single function definition to `generateData.tools` — `noop_placeholder`, no parameters, described to the model as never-callable — which is enough to flip the gate.

> **This enforces behaviour DeepSeek does not document as a supported mode.** It is an undocumented side effect of a documented rule, not a contracted feature: the reasoning gate is an implementation detail of DeepSeek's chat encoder, and the setting stops working if DeepSeek changes how that decision is made. There is no promise from the vendor that it keeps working.

The setting is hidden while **Prior reasoning sent to the API** is `None`, because the tool exists to make DeepSeek keep the reasoning this extension sends — with nothing sent there is nothing to keep. It is injected deliberately *after* the guard that switches the extension off on tool requests, so a request that already uses real tools is untouched.

Verified on `deepseek-ai/deepseek-v4.1-flash` through NVIDIA NIM: with the placeholder the model answered `R9 C9` across nine turns, without it a single-turn reasoning and `C9`. Whether the documented 400 for an incomplete reasoning replay is enforced on that endpoint is **not verified** — every send strategy, including the windowed ones, was accepted there, but DeepSeek's own endpoint is documented to require the full replay.

Two costs worth weighing:

* Reasoning that used to be discarded is now billed as input tokens for the whole history, so [Trim to budget](#trim-to-budget) starts dropping real chat turns sooner. The feature and its price arrive together: on a small context the trimmed turns take the preserved reasoning with them, which can leave the request no better off than before.
* If the model calls the placeholder despite the description, SillyTavern turns the invocation into a tool message, and the extension then disables itself for that chat until those messages are gone.

One gap in the budget path: it estimates `generateData.messages` and never adds the tool schema, so a trimmed request carries a few dozen tokens more than the trim concluded it could afford. That sits inside the estimator's stated accuracy, and it only starts to matter when the budget is small enough for a few dozen tokens to be a visible share of it.

### Group chats

In a group chat several characters take turns, and only the one being generated is "the model". SillyTavern prefixes every message with its speaker name in the outgoing request (`names_behavior`), so the extension mirrors that prefix when it matches chat turns to request messages — without it, nothing matches in a group chat and no prior reasoning is sent at all, silently.

The dropdown **Group chats: whose reasoning is sent** narrows that further:

| Mode | Behaviour |
| --- | --- |
| `All characters` | Every character's stored reasoning is re-attached (default). |
| `Only the current character` | Only the turns of the character being generated right now. Keeps the other characters' thinking out of the request, so the model is not told it reasoned about turns it never wrote. |
| `None` | No prior reasoning is attached at all in a group chat. |

The setting only applies to group chats; everywhere else every character's reasoning is re-attached as usual. It is shown only while **Prior reasoning sent to the API** is not `None`, since it narrows a feature that is off. Note that `Only the current character` intentionally leaves the other characters' request messages without `reasoning_content`, so they appear as unmatched in the debug log — that is the mode working, not a fault.

### Trim to budget

After attaching preserved reasoning, checks the estimated token usage again and removes the oldest user/assistant history messages until the request fits the configured context budget.

Enabled by default.

On requests that involve real tools the handler returns before reasoning is attached, so neither the strategies nor this trimming run there. That guard is inherited from the original thinking-prefill patch and has never been relaxed.

The one exception is the opt-in placeholder tool of [Multi-turn reasoning on DeepSeek](#multi-turn-reasoning-on-deepseek). It is injected *after* that guard on purpose, so with it enabled the strategies and this trimming run exactly as they do on a request without tools — there is nothing tool-specific about the budget path.

### Thinking prefill

Optionally injects a `reasoning_content` prefill for the current generation. This is separate from preserved reasoning history.

## Important

Preserved reasoning increases prompt size. A long reasoning history can consume a significant part of the available context. The windowed strategies exist to counter this — see [Model contracts behind the strategies](#model-contracts-behind-the-strategies) for what each provider documents, and for what is still unverified in a roleplay context.

Token counts calculated by the extension are estimates and may differ from the exact token accounting performed by the model provider.

## Credits

Original extension by [Rurijian](https://github.com/Rurijian).
