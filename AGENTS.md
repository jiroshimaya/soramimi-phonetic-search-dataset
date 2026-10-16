# Repository agent rules

## Worktrees and delivery

- Read [README.md](README.md) and the relevant project documentation before editing.
- Preserve the primary checkout's branch, local changes, and running services.
  Make implementation changes in a session-specific linked worktree and task
  branch based on `origin/main`.
- Deliver implementation through a pull request merged into `main` after all
  mandatory checks pass. Do not bypass branch protections.
- Keep changes within the requested scope. Preserve existing data, results,
  interfaces, license notices, and attribution unless their change is requested.
- Do not commit credentials, local environment files, private inputs, generated
  media, or model weights. Do not invoke publishing or deployment commands unless
  the user requests them.
- Use the documented development and verification commands. For documentation-only
  changes, check referenced paths and commands and run `git diff --check`;
  all mandatory CI checks still apply.
