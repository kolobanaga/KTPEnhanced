# KTPEnhanced

A SillyTavern extension for Kimi/GLM models that enables client-side handling of `reasoning_content` without modifying the SillyTavern server.

This is a fork of [Rurijian/KimiThinkingPrefill](https://github.com/Rurijian/KimiThinkingPrefill), with additional fixes for preserved reasoning and context-budget handling.

## Contents

- [What is Preserved Thinking and Multi-turn Reasoning?](#what-is-preserved-thinking-and-multi-turn-reasoning)
- [Features](#features)
- [Preserved reasoning and message alignment (LCS)](#preserved-reasoning-and-message-alignment)
- [Context budget](#context-budget)
- [Settings](#settings)
  - [Prior reasoning to the API](#prior-reasoning-to-the-api)
  - [Provider-specific model contracts](#model-contracts-behind-the-strategies)
  - [Personal RP observation: GLM 5.1 vs GLM 5.3](#personal-rp-observation-glm-51-vs-glm-53)
  - [Multi-turn reasoning on DeepSeek](#multi-turn-reasoning-on-deepseek)
  - [Group chats](#group-chats)
  - [Trim to budget](#trim-to-budget)
  - [Thinking prefill](#thinking-prefill)
- [Limitations and token budget](#limitations-and-token-budget)
- [Credits](#credits)

## What Is Preserved Thinking and Multi-turn Reasoning?

Different labs and providers use overlapping terms for related behaviour: **Preserved Thinking**, **Multi-turn Reasoning**, **Interleaved Thinking**, **Reasoning Persistence**, and **CoT Persistence**. These labels are not always exact synonyms, and the API requirements vary, but this README uses them as a broad family of techniques for carrying previous assistant reasoning into later requests instead of discarding it after every response.

A chat-completion message may contain both the user-facing answer (`content`) and a separate reasoning field (`reasoning_content`). In the example below, `C` stands for the visible answer and `R` for that turn's reasoning:

```js
{
  messages: [
    { role: 'user', content: 'Turn 1.' },
    { role: 'assistant', content: 'C1', reasoning_content: 'R1' },
    // Turns 2–8 follow the same user/assistant pattern.
    { role: 'user', content: 'Turn 8.' },
    { role: 'assistant', content: 'C8', reasoning_content: 'R8' },
    { role: 'user', content: 'Turn 9.' }
  ],
  // other request parameters ...
}
```

The expected next response is `{ role: 'assistant', content: 'C9', reasoning_content: 'R9' }`. A failure mode is that the model starts reasoning from scratch instead of continuing the existing reasoning thread, for example: `reasoning_content: "Let me work through this from the beginning..."` despite the previous turns already containing reasoning.

The point is not merely to preserve the visible conversation: the model should be able to continue both the **conversation thread** and the **reasoning thread**. Dropping or changing historical reasoning can make some models re-derive earlier steps, lose continuity, or spend extra reasoning tokens.

This is not a universal behaviour of every model or API. Some providers require previous reasoning to be replayed exactly and in order; others ignore it unless specific request parameters are present; some reject it. The provider-specific notes below describe those differences. KTPEnhanced can only re-attach reasoning that SillyTavern actually stored, and the model/provider must support the relevant request format.

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

The important user-facing guarantee is that matching stays **in order** and prioritizes the largest valid set of pairs. The `Last K` window is applied **after** pairing, so a smaller window does not distort the underlying correspondence. This also helps prevent an injected old message at the end of the payload from stealing a candidate needed to match newer turns.

Before matching, messages that already have `reasoning_content` are left alone, and messages with no usable matching key are excluded. Matching keys account for relevant SillyTavern transformations and, in group chats, mirror the speaker-name prefix added to outgoing message content. If the text has been transformed in a way the extension cannot reliably reconstruct, it is safer for the message to remain unmatched than to attach reasoning to the wrong turn.

<details>
<summary>Implementation details: tie-breaking and complexity</summary>

The LCS pairing prioritizes:

1. **The maximum number of valid pairs.** A longer order-preserving match always wins.
2. **Newer payload positions as a tie-breaker.** Among matches with the same number of pairs, the algorithm prefers the larger total of matched payload positions, biasing ambiguous matches towards the newest end. This is a sum-based tie-breaker, not a strict rule that every individual pair must be as far right as possible. If identical text appears multiple times, the text alone cannot always identify the true origin.

The calculation uses O(n × m) time for `n` candidate chat turns and `m` outgoing assistant messages. Text keys are interned so the inner comparison uses integer IDs rather than repeatedly comparing long message bodies. If matching fails, the error is contained and the outgoing payload is left untouched.

</details>

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

**A practical starting point:** use `All` when the provider explicitly requires the complete, unchanged reasoning history. Consider `Last K` when context pressure matters and the exact endpoint tolerates a shortened history; try `Last K + Omitted` when older turns should still be marked as having reasoning. These windowed modes save prompt space but can conflict with provider contracts, so evaluate them on the model and endpoint you actually use.

Notes:

* `K` counts *stored reasonings*, not assistant turns, so a stretch of turns without reasoning (for example from a non-reasoning model) does not silently consume the window.
* The default placeholder is worded as a note about the context budget, not as a redaction. It is written into the model's own `reasoning_content`, so wording like "omitted" or "withheld" reads as *reasoning hidden for policy reasons* — the convention providers use for encrypted reasoning — instead of "this turn's thinking was compressed away". It also never claims that no reasoning happened, which could make the model re-derive a turn it actually reasoned through.
* Only the outgoing request payload is modified. The stored `extra.reasoning` in the chat is never rewritten, so switching strategies back and forth is lossless and re-sends the full reasoning.
* Nothing is re-attached for turns whose reasoning was never stored. The model and the request source have to return reasoning in the first place — the **Request model reasoning** setting controls whether the API is asked for it at all (and whether it is displayed), and a non-reasoning model stores none regardless of that setting.
* Upgrading from the old "Send all prior assistant reasoning" checkbox: checked maps to `All`, unchecked maps to `None`.

#### Model contracts behind the strategies

The strategies are a trade-off, not a universal optimization. Providers differ in what they require for historical reasoning: some accept only a complete, unchanged sequence, while others ignore the field unless a request parameter activates it. These contracts are written primarily for API/agent workflows; roleplay behaviour can differ in practice.

<details>
<summary>Detailed provider matrix and compatibility notes</summary>

| Model | Thinking control | Cross-turn requirement |
| --- | --- | --- |
| `GLM-5.3`, `GLM-5.3-Flash` | always on — `thinking.type: "disabled"` fails the request; depth via `reasoning_effort` (`low`/`high`/`max`) | Preserved Thinking (`clear_thinking: false`) requires historical `reasoning_content` **complete, unmodified and in its original order**; editing, trimming or reordering may degrade quality and cache hit rate. On by default on the Coding Plan endpoint. |
| `kimi-k3` | always on; do not send `thinking`; depth via top-level `reasoning_effort` | Preserved Thinking is **always on**: every historical assistant message must be returned as-is; documented as required for multi-turn and tool-call loops. |
| `kimi-k2.6` / `kimi-k2.7-code` | `thinking.type` + `keep` (`null` by default / `"all"`); `k2.7-code` accepts only `{"type":"enabled","keep":"all"}` | Cross-turn preservation follows `keep`. `kimi-k2.5` has no Preserved Thinking. |
| `deepseek-v4-pro` / `deepseek-v4-flash` | on by default | The gate is whether `tools` is present on the **request**, not whether a tool is called. With `tools`, all previous `reasoning_content` must be passed back or the API may return `400 The reasoning_content in the thinking mode must be passed back to the API`. Without `tools`, the field is ignored rather than added to context. |
| `deepseek-reasoner` / R1 (legacy) | — | Opposite rule: `reasoning_content` must **not** be passed back, or the request fails. |
| `qwen3.8` (DashScope) | `enable_thinking` | `preserve_thinking: true` (default `false`). The model is trained to keep reasoning blocks in the conversation; stripping them can make it re-derive earlier conclusions. |
| `MiMo V2.5` | — | Same documented 400 as DeepSeek V4 when a request carries tools and a previous turn lost its `reasoning_content`. |
| OpenRouter, user-defined tools | `reasoning.effort` / `max_tokens` / `exclude` | Documented: echo reasoning via `message.reasoning` (or `reasoning_content` alias), or `message.reasoning_details` for encrypted/summarized blocks. Consecutive reasoning blocks must remain in their original sequence and unmodified. |
| OpenRouter, server tools | server-side execution; `openrouter:*` entries in `tools` | **Undocumented.** Request logs observed prior-history reasoning being discarded while reasoning from the server-side tool loop survived. Treat this as empirical behaviour, not a contract. |
| `minimax-M2` (legacy) | always on | Thinking is inline `<think>` tags inside `content`, not a separate field. |
| Anthropic / Gemini | separate request shapes | Reasoning is not a `reasoning_content` field: Anthropic uses signed thinking blocks and `clear_thinking_20251015`; Gemini uses `thinkingConfig`. Out of scope for this extension. |

Practical consequences:

- `All` is the only option consistent with the documented full-history contract for `kimi-k3` and GLM Preserved Thinking. `Last K` truncates the sequence; `Last K + Omitted` edits older reasoning.
- DeepSeek V4 (and MiMo) can return 400 when a request carries tools and a previous turn lost its `reasoning_content`. The extension normally exits before attaching reasoning whenever real tools are in play. The deliberate exception is [Multi-turn reasoning on DeepSeek](#multi-turn-reasoning-on-deepseek), which opts into sending a placeholder tool.
- OpenRouter documents preserving reasoning for user-defined tools, but server tools have shown different behaviour in request logs. Its `reasoning.context: "current_turn"` / `"all_turns"` control is documented for newer OpenAI models and is not a general lever for the `reasoning_content` flow targeted here.
- Dropping reasoning does not always save total tokens: some models may re-derive earlier conclusions and spend more reasoning tokens. The best trade-off depends on the model, endpoint, and workload.
- Field names differ (`reasoning_content`, `reasoning`, `reasoning_details[]`, or inline `<think>`). Sending the wrong field can also cause a hard 400; for example, GLM-4.7 on Cerebras expects `reasoning` rather than `reasoning_content`.

</details>

#### Personal RP observation: GLM 5.1 vs GLM 5.3

I often use GLM-5-series models for roleplay because I like their prose and find their reactions the most human-like. I noticed a quality drop in GLM-5.3 compared with GLM-5.1: in ordinary use, GLM-5.3 more often misses small but important details and sometimes skips reasoning entirely, going straight to the response.

To compare, I regenerated several swipes in an older chat whose replies and stored reasoning came from GLM-5.1, testing GLM-5.3 both with KTPEnhanced enabled (`Last K`, `K = 10`) and without it. The difference was noticeable: with preserved reasoning, both the reasoning and the resulting replies were substantially better and closer to GLM-5.1.

This is an informal observation from one workflow, not a controlled benchmark. It does, however, show why a windowed strategy can be useful in RP even when it conflicts with some providers' documented full-history contracts. In practice, the tested model appeared able to use the retained reasoning coherently; this does **not** establish that every model/endpoint accepts or benefits from `Last K` or `Last K + Omitted`. Treat API requirements as constraints, and test output quality on the exact endpoint you use.

### Multi-turn reasoning on DeepSeek

DeepSeek is an exception to the usual behaviour. On the tested V4-style path, without `tools` in the request, historical `reasoning_content` is accepted but ignored; with `tools` present, the endpoint can preserve it. There is no documented parameter for enabling the latter behaviour directly.

The optional **Send a placeholder tool** setting uses this side effect: it adds a no-parameter `noop_placeholder` function to `generateData.tools`, described to the model as never-callable. This is an undocumented workaround, not a provider-supported feature.

<details>
<summary>DeepSeek implementation notes, verification, and caveats</summary>

`chat_template_kwargs.drop_thinking: false` is a vLLM chat-template passthrough; on an NVIDIA NIM endpoint it reached the request without changing the reply. The placeholder tool is deliberately injected after the guard that disables the extension for requests already using real tools, so existing tool requests are left untouched.

> **This enforces behaviour DeepSeek does not document as a supported mode.** It is an undocumented side effect of a documented rule, not a contracted feature: the reasoning gate is an implementation detail of DeepSeek's chat encoder, and the setting stops working if DeepSeek changes how that decision is made. There is no promise from the vendor that it keeps working.

The setting is hidden while **Prior reasoning sent to the API** is `None`, because the tool exists to make DeepSeek keep the reasoning this extension sends — with nothing sent there is nothing to keep. It is injected deliberately *after* the guard that switches the extension off on tool requests, so a request that already uses real tools is untouched.

Verified on `deepseek-ai/deepseek-v4.1-flash` through NVIDIA NIM: with the placeholder the model answered `R9 C9` across nine turns, without it a single-turn reasoning and `C9`. Whether the documented 400 for an incomplete reasoning replay is enforced on that endpoint is **not verified** — every send strategy, including the windowed ones, was accepted there, but DeepSeek's own endpoint is documented to require the full replay.

Two costs worth weighing:

* Reasoning that used to be discarded is now billed as input tokens for the whole history, so [Trim to budget](#trim-to-budget) starts dropping real chat turns sooner. The feature and its price arrive together: on a small context the trimmed turns take the preserved reasoning with them, which can leave the request no better off than before.
* If the model calls the placeholder despite the description, SillyTavern turns the invocation into a tool message, and the extension then disables itself for that chat until those messages are gone.

One gap in the budget path: it estimates `generateData.messages` and never adds the tool schema, so a trimmed request carries a few dozen tokens more than the trim concluded it could afford. That sits inside the estimator's stated accuracy, and it only starts to matter when the budget is small enough for a few dozen tokens to be a visible share of it.

</details>

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

### Partial mode filter

The Kimi API marks an unfinished assistant turn with a `partial: true` field on the message. The extension sets it on the message that carries the unfinished turn — the transformed Continue target, or the injected prefill message.

**Partial mode is officially supported by the Moonshot provider only.** Everywhere else it is best-effort: expect a `400` from a strict endpoint, a flag that is silently ignored, or behaviour nobody documented. Use it at your own risk, and check whether it actually did something.

The **Partial mode filter** field (comma-separated substrings, default `kimi`) lists the models whose requests may carry the flag. It is matched against the model id, so the same rule works on the official API, on OpenRouter, on a proxy and on a self-hosted server alike — name the model the way your endpoint names it.

The flag is sent when the model id contains one of these entries. If the filter is not empty and the model does not match, a `partial` left in the payload by another extension is removed as well, because a strict endpoint rejects the whole request on a field it does not know. An empty filter switches the mode off completely and leaves other extensions' payloads alone.

**On the official Moonshot AI Chat Completion Source the core sets the flag by itself — but only for a request that ends with an assistant turn.** SillyTavern's server marks the last message of a Moonshot-source request as `partial` whenever that message is an assistant turn. On a Continue generation the request ends with the system nudge instead, so the core marks nothing there and this filter becomes the only decider. Every other source (OpenRouter, a custom endpoint, a proxy) gets nothing from the core, and there the filter decides on its own.

<details>
<summary>Observed behaviour on live requests — measured, not documented by the providers</summary>

| Endpoint | Result |
| --- | --- |
| Moonshot, official provider | works |
| OpenRouter, `Moonshot AI` provider | works |
| OpenRouter, another provider | flag ignored, no error, ordinary generation |
| Cloudflare Workers AI, `kimi-k2.7-code` | accepted, effect undocumented |
| Strict server (NIM), wrong model | `400 Invalid request` on the unknown field |

</details>

Without the flag the reasoning still travels in its own `reasoning_content` channel. What is lost is the marker that tells the Kimi API "this turn is unfinished, continue it" — the prefill is no longer continued as a partial turn.

### Continue nudge and the partial mode

The partial mode expects the request to **end** with the assistant turn that is to be continued. With the default core settings a Continue generation appends a system nudge after that turn (`Continue nudge prompt`, `openai.js:898`), so the request ends with the nudge instead:

```json
{ "role": "assistant", "content": "", "reasoning_content": "R9" },
{ "role": "system", "content": "[Continue your last message without repeating its original content.]" }
```

Two consequences, both measured on live requests.

**The flag is no longer on the last message.** The Kimi API documents `partial: true` on the last message of the request. With the nudge in place the flag — wherever it is set — sits on the turn before it. On the official Moonshot source the core marks the last message, which is now the nudge, so the core marks nothing at all and the extension's filter becomes the only decider.

**The nudge triggers reasoning.** A model reacts to the instruction by reasoning about it, and the answer comes back with a visible trace of the model thinking about "continue without repeating" instead of continuing the turn silently. Measured on OpenRouter, the same request answered correctly in both cases — but with the nudge the reasoning was on and visible, without it the response carried no reasoning at all:

```json
// with the nudge
{ "role": "assistant", "content": "C9", "reasoning": "The system says 'Continue your last message without repeating its original content.' …" }
// without it
{ "role": "assistant", "content": "C9", "reasoning": null }
```

<details>
<summary>How to get a request that ends with the assistant turn</summary>

The nudge is only added when the core's **Continue Prefill** option is off (`openai.js:898`), and an empty nudge is not sent at all — so either enable Continue Prefill, or empty the **Continue nudge prompt** field. The extension's debug log prints the final payload, so it shows which shape you got.

| Setup | Continue nudge | Partial mode filter | Who marks the assistant turn |
| --- | --- | --- | --- |
| Official Moonshot source | off | off | the core — the request ends with that turn |
| OpenRouter, or a custom endpoint at `api.moonshot.ai` | off | on | this extension — the request ends with that turn |
| any source, nudge left on | on | any | this extension, but on the turn *before* the nudge — not the position the API documents |

On the official source, leaving the filter off is what leaves the flag to the core. On every other source the extension has to write it, because the core writes nothing there.

</details>


## Limitations and token budget

Preserved reasoning increases prompt size. A long reasoning history can consume a significant part of the available context. The windowed strategies exist to counter this — see [Model contracts behind the strategies](#model-contracts-behind-the-strategies) for documented provider requirements and the distinction between those requirements and informal RP observations.

Token counts calculated by the extension are estimates and may differ from the exact token accounting performed by the model provider.

## Credits

Original extension by [Rurijian](https://github.com/Rurijian).
