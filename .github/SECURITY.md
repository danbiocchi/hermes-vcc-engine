# Security policy

## Reporting a vulnerability

Please report security problems privately, not in a public issue. Use GitHub's **Report a vulnerability** button on the [Security tab](https://github.com/danbiocchi/hermes-vcc-engine/security) of this repository. Include the Hermes version, the plugin version, and the steps or input that trigger the problem.

## Supported versions

Fixes go into the latest release. The Hermes plugin catalog pins an exact commit, so a fix reaches catalog users when the pin is updated and they run `hermes plugins update hermes-vcc-engine`.

## Scope

The plugin's runtime code is `__init__.py` and the `vcc/` folder. By design it makes no network calls of its own, runs no shell commands or subprocesses, reads no credentials or environment variables, and writes no files. Anything that makes it do one of these, or lets tool output change the summary's structure or inject instructions, is in scope. The full list of what the plugin does and doesn't do is in the README's [Security and privacy](../README.md#security-and-privacy) section.

The `bench/` scripts are run by hand and are not loaded by Hermes. They read `~/.hermes/state.db` read-only and write results to `~/.cache/hermes-vcc-engine/`. Problems there are in scope too.
