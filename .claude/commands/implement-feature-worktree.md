You are tasked with implementing a single feature from a specification file with complete autonomy and quality assurance in an isolated git worktree environment.

## 🌳 Git Worktree Context

**Your Environment:**
- You are working in an isolated git worktree, NOT the main repository
- You have a dedicated branch for this feature implementation
- You can make commits, test aggressively, and experiment without affecting other work
- Your worktree path and branch name correspond to the feature being implemented

**Worktree Advantages You Should Leverage:**
1. **Aggressive Testing**: Run tests frequently - breaking tests won't affect others
2. **Experimental Changes**: Try different approaches and revert if needed
3. **Incremental Commits**: Make small, frequent commits to track progress
4. **Safe Cleanup**: Can reset hard or clean without worry if implementation fails

## 📋 Input Processing

1. **Parse the feature specification** provided via `$ARGUMENTS`
2. **Extract the feature name** from the FIRST occurrence of:
   - H1 heading (# Feature Name)
   - H2 heading if no H1 exists (## Feature Name)
   - First line of the file if no headings exist
3. **Generate a slug** from the feature name for file naming:
   - Convert to lowercase
   - Replace spaces with hyphens
   - Remove special characters except hyphens and underscores
   - Example: "User Authentication System" → "user-authentication-system"

## 🎯 Implementation Scope

**STRICT BOUNDARIES:**
- Implement ONLY what is explicitly described in the specification
- Do NOT add features, enhancements, or "nice-to-haves" not mentioned
- If requirements are ambiguous:
  1. Choose the simplest valid interpretation
  2. Document your assumption in the implementation summary

**ANALYSIS REQUIRED:**
- Determine if the feature requires:
  - Frontend changes only
  - Backend changes only
  - Full-stack implementation
  - Database modifications
  - External service integrations

## 🛠️ Implementation Process

### Phase 1: Project Discovery (REQUIRED)

1. **Check for project instructions** by reading these files in order:
   - `CLAUDE.md` (project-specific guidelines)
   - `README.md` (project overview and setup)
   - `package.json` or `pyproject.toml` or equivalent (dependencies and scripts)

2. **Understand the codebase architecture** by examining:
   - Directory structure (`src/`, `lib/`, `components/`, etc.)
   - Entry points (`main.py`, `app.js`, `index.html`, etc.)
   - Configuration files (`.env`, `config/`, etc.)
   - Testing setup (`tests/`, `__tests__/`, `*.test.*`)

3. **Identify relevant existing patterns** by searching for:
   - Similar features or components already implemented
   - Common naming conventions and file organization
   - Import/export patterns
   - Error handling approaches

### Phase 2: Technical Implementation

1. **Plan the implementation** by:
   - Identifying all files that need creation/modification
   - Determining integration points with existing code
   - Planning the order of implementation to avoid breaking changes

2. **Implement following project standards**:
   - Use the EXACT same coding style, indentation, and formatting
   - Follow naming conventions found in existing code
   - Use the same libraries and frameworks already in use
   - Maintain the same error handling and logging patterns

3. **Handle dependencies carefully**:
   - NEVER assume a library is available - check package files first
   - Use only dependencies already installed or explicitly mentioned
   - If new dependencies are needed, document why in the implementation summary

### Phase 3: Quality Assurance (MANDATORY)

1. **Verify implementation works** by:
   - Running any existing linter/formatter commands found in package files
   - Executing test commands if they exist (`npm test`, `pytest`, etc.)
   - Checking for compilation errors or import issues

2. **Test the feature** by:
   - Creating minimal test cases that verify the feature works as specified
   - Testing integration with existing functionality
   - Verifying no existing functionality was broken

3. **Handle implementation blockers**:
   - If tests fail: Fix the issues before completing
   - If dependencies are missing: Document the blocker and provide setup instructions
   - If specifications are impossible: Document why and suggest alternatives

### Phase 4: Git Workflow (REQUIRED)

1. **Commit Strategy**:
   ```bash
   # Make frequent, logical commits during implementation
   git add <files>
   git commit -m "feat: implement [component] for [feature]"
   
   # Example commit sequence:
   git commit -m "feat: add data model for {feature}"
   git commit -m "feat: implement business logic for {feature}"
   git commit -m "feat: add API endpoints for {feature}"
   git commit -m "test: add unit tests for {feature}"
   git commit -m "docs: update documentation for {feature}"
   ```

2. **Branch Management**:
   - Your branch name should match the feature slug
   - Keep commits atomic and focused
   - Use conventional commit messages (feat:, fix:, test:, docs:, refactor:)

3. **Recovery Options** (if implementation goes wrong):
   ```bash
   # Reset to start over (safe in worktree)
   git reset --hard HEAD~n  # Go back n commits
   
   # Stash experimental changes
   git stash push -m "experimental approach for {feature}"
   
   # Clean up untracked files if needed
   git clean -fd
   ```

4. **Final Validation**:
   ```bash
   # Ensure all changes are committed
   git status  # Should show clean working tree
   
   # Review the implementation
   git log --oneline -10  # Review commit history
   git diff HEAD~n  # Review total changes
   ```

## 📝 Documentation Output

**CREATE A DOCUMENTATION FILE:**
- Filename: `implementation-summary-{feature-slug}.md`
- Location: Project root or `/docs` if it exists
- Do NOT ask for permission to create this file

**REQUIRED CONTENT:**
```markdown
# Implementation Summary: {Feature Name}

## Feature Overview
[Brief description of what was implemented]

## Scope Determination
- Implementation type: [Frontend/Backend/Full-stack]
- Rationale: [Why this scope was determined]

## Changes Made

### Files Created
- `path/to/file.ext` - [Purpose]

### Files Modified
- `path/to/file.ext` - [What was changed and why]

### Key Decisions
- [Decision 1]: [Rationale]
- [Decision 2]: [Rationale]

### Assumptions Made
- [Any ambiguities and how they were resolved]

### Integration Points
- [How this feature connects with existing code]

## Validation Results
- Linting: [PASS/FAIL - details if failed]
- Tests: [PASS/FAIL - details if failed]
- Manual Testing: [What was tested and results]

## Testing Considerations
- [Key areas that should be tested]
- [Edge cases to consider]

## Future Considerations
- [Any technical debt or future improvements noted but not implemented]

## Implementation Blockers (if any)
- [List any issues that prevented full implementation]
- [Required dependencies or setup needed]
```

## ⚠️ Critical Execution Rules

**AUTONOMOUS EXECUTION:**
- You must complete the ENTIRE implementation without asking questions
- Make decisions based on existing patterns in the codebase
- Document all assumptions in the implementation summary
- If truly blocked, document the blocker and provide workaround instructions

**ERROR RECOVERY (Leverage Worktree Isolation):**
1. If a test fails → Fix the code and re-run tests (can break things temporarily)
2. If a dependency is missing → Check if there's an alternative already installed
3. If the specification is impossible → Implement the closest possible solution and document limitations
4. If integration breaks existing code → `git reset --hard` and try a different approach
5. If approach isn't working → `git stash` current work and try alternative implementation
6. Can't figure out the issue → Make aggressive debug changes since you're isolated

**SUCCESS CRITERIA:**
- ✅ Feature works as specified (or documented limitations)
- ✅ No existing functionality broken
- ✅ Code follows project conventions
- ✅ Implementation summary created
- ✅ All changes are reversible/removable

## 🔍 Common Patterns Reference

**Python Projects:**
- Check for `__init__.py` files to understand module structure
- Look for `requirements.txt` or `pyproject.toml` for dependencies
- Follow PEP 8 and existing docstring formats
- Use type hints if the project already uses them

**JavaScript/TypeScript Projects:**
- Check `package.json` for scripts and dependencies
- Look for `.eslintrc` or `.prettierrc` for style rules
- Follow existing import/export patterns (CommonJS vs ES6)
- Maintain consistent async patterns (callbacks vs promises vs async/await)

**Testing Patterns:**
- Match existing test file naming (*.test.js, test_*.py, etc.)
- Use the same testing framework (Jest, pytest, unittest, etc.)
- Follow existing assertion styles and test organization
- Include both positive and negative test cases

## 🏁 Implementation Completion

**When You're Done:**
1. All tests pass (or failures are documented as blockers)
2. Feature works according to specification
3. All changes are committed with descriptive messages
4. Implementation summary document is created
5. Working tree is clean (`git status` shows no uncommitted changes)

**Final Deliverables:**
- ✅ Working feature implementation in the worktree
- ✅ Complete git history showing implementation progression
- ✅ Implementation summary document with all decisions and outcomes
- ✅ Clean, tested, and documented code ready for review

**Remember: You have complete autonomy in this isolated worktree. Use it to:**
- Experiment freely without fear of breaking things
- Make frequent commits to track your thought process
- Try multiple approaches if the first doesn't work
- Debug aggressively when needed
- Reset and start over if you get stuck
