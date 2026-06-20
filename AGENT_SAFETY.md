# Agent safety rules (read first — applies to every delegated session)

These sessions run with `--permission-mode bypassPermissions`: you are NOT prompted before
destructive actions, so the burden is on you. Violating these is a hard failure.

1. **Never delete or modify files you did not create.** No exceptions. If a file or directory
   was present when your session started, leave it alone — even if it looks like junk.
2. **Never `rm -rf` a directory you didn't create**, and never run recursive/glob deletes over
   parent or shared paths. Specifically off-limits: anything untracked you didn't make,
   `.worktrees/`, other `PROMPT_*.md`, `setup-*.sh`, checkpoints/outputs you didn't generate.
3. **Do all scratch/test work inside ONE directory you create yourself** — e.g.
   `scratch/<your-task-id>/` or `/tmp/<your-task-id>/` — and clean up by removing **only that
   exact path**. Never clean up with broad patterns.
4. **Stay in your lane.** Only edit files your task names. Keep v1 (`nca.py`, `evolve.py`,
   `env.py`, existing checkpoints/outputs) intact unless the task explicitly says to change it.
5. **When unsure, don't.** Leave the file, finish your task, and note it in your report rather
   than deleting or "fixing" something ambiguous.
6. Prefer `git status` to confirm what you added before removing anything; only remove paths
   you can see you created this session.
7. **Commit your work on your branch before you finish** (do not merge to main unless told).
   Never leave deliverables uncommitted in the working tree for the "next" session — the next
   branch will be built on a phantom base and work gets tangled. "Don't merge" ≠ "don't commit."
