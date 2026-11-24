- Keep responses concise.
- Avoid defensive coding.
- Keep the first implementation simple. Document outstanding concerns in a code comment or mention them in your response.
- Use class-based patterns as the standard and keep code readable.
- Follow the single responsibility principle. Give each function one job and separate work that changes for different reasons.
- Keep implementation details behind a stable module interface (information hiding). Apply the open/closed principle by extending behavior in the owning module without editing shared callers.
- Annotate every function argument. Python infers the implicit `self` and `cls` receivers.
- Annotate every function that returns a value or yields items. Include `None` in the type when it is an intentional result. Omit the return annotation on procedures that only perform an action.
- Manage project dependencies with `uv add` and `uv remove`. After editing dependencies in `pyproject.toml` directly, run `uv sync` to regenerate `uv.lock` and update the environment. Dont edit `uv.lock` by hand unless required for specific operations

## Git

- Use the local `git` CLI for regular Git operations, including creating branches, committing, pulling, and pushing. Do not use the GitHub API for these operations.
