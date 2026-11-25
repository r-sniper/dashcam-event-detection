---
name: linear-task
description: Implement a Linear issue and open a pull request. Use when the user asks to take, execute, or work on a Linear task by issue key or issue URL, or invokes /linear-task.
---

# Linear task

## Identify the issue

Accept a team key and issue number such as `ENG-42`, or extract the key from a Linear issue URL such as `https://linear.app/acme/issue/ENG-42/fix-export`. Read the argument from the user's request. If it contains no issue key, ask for the key or issue URL and wait before any lookup or change. If it names multiple distinct issues, ask which one to take.

## Execute

1. Read the issue title, description, current state, linked PRs and **all** comments, including replies and inline threads. Follow pagination to the end. Read linked material needed for the task. Identify the intended outcome and what would demonstrate completion.
2. Check the repository's applicable `AGENTS.md` files and contribution instructions. Identify the base branch, existing issue branch or PR, and local edits. Start the branch name with the full issue key, for example `ENG-42-fix-export`. Reuse work for this issue. For new work, branch from the current target branch and isolate unrelated edits.
3. Resolve the issue team's workflow states by name. Move the ticket to **In Progress** when starting implementation. On a retry, resume the last successful step. For an issue already in review, inspect its PR before changing state. If the issue is closed, report its disposition before reopening it.
4. Implement the requested change. Check the full PR diff against the issue, its comments and the applicable repository instructions. Run the relevant checks, fix findings and review the resulting diff again.
5. Commit and push the issue branch with the local Git CLI. Open a PR against the intended base, or update the existing PR. Include the full issue key in the PR title, for example `ENG-42 Fix export`. In the PR body, link the issue and describe the change and check results. Follow the repository's PR template when present.
6. Verify the PR is open and contains the reviewed change. Link it to the Linear issue if needed, then move the issue to the team's **In Review** state. Report the PR link, checks and final ticket state. If publishing fails, keep the ticket in progress and report the blocker. If the PR exists but the Linear update fails, report the PR and retry the outstanding update.
