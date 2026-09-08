# KTPEnhanced

A SillyTavern extension for Kimi/GLM models that enables client-side handling of `reasoning_content` without modifying the SillyTavern server.

This is a fork of [Rurijian/KimiThinkingPrefill](https://github.com/Rurijian/KimiThinkingPrefill), with additional fixes for preserved reasoning and context-budget handling.

## Features

* Client-side equivalent of the Kimi/GLM thinking prefill server patch.
* Re-attaches stored `extra.reasoning` as `reasoning_content` for previous assistant messages.
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

### Send all prior reasoning to the API

Re-attaches stored `extra.reasoning` from previous assistant messages as `reasoning_content`.

This is useful for models that expect previous reasoning to be preserved between turns.

### Trim to budget

After attaching preserved reasoning, checks the estimated token usage again and removes the oldest user/assistant history messages until the request fits the configured context budget.

Enabled by default.

### Thinking prefill

Optionally injects a `reasoning_content` prefill for the current generation. This is separate from preserved reasoning history.

## Important

Preserved reasoning increases prompt size. A long reasoning history can consume a significant part of the available context.

Token counts calculated by the extension are estimates and may differ from the exact token accounting performed by the model provider.

## Credits

Original extension by [Rurijian](https://github.com/Rurijian).