- Keep responses concise.
- Avoid defensive coding.
- Use class-based patterns as the standard and keep code readable.
- Annotate every function argument. Python infers the implicit `self` and `cls` receivers.
- Annotate every function that returns a value or yields items. Include `None` in the type when it is an intentional result. Omit the return annotation on procedures that only perform an action.

## Git

- Use the local `git` CLI for regular Git operations, including creating branches, committing, pulling, and pushing. Do not use the GitHub API for these operations.
