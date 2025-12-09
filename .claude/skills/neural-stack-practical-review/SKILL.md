---
name: neural-stack-practical-review
description: Review and grade Neural Stack Program weekly practical submissions from GitHub issues and pull requests. Use when evaluating student code submissions, providing feedback, or assessing practical work.
---

# Neural Stack Practical Review Skill

Specialized skill for evaluating Neural Stack Program weekly practicals against learning objectives, validating understanding, and providing rigorous feedback.

## When to Use

Activate when the conversation involves:
- Reviewing practical submissions (Module X Week Y)
- Grading student code from GitHub PRs
- Assessing completion of weekly assignments
- Providing feedback on portfolio projects
- Evaluating checkpoint practicals

## Required Inputs

- **Module**: M[1-8] identifier
- **Week**: Week number (1-10)
- **GitHub Issue #**: Contains requirements
- **GitHub PR #**: Code submission
- **Portfolio Flag**: If portfolio-quality practical (optional)

## Review Workflow

### 1. Load Context (2-3 min)
- Fetch GitHub issue for requirements/success criteria
- Read module documentation: `/Users/a-f/claude-container/neural-stack/course_docs/module_[N]_*.md`
- Review PR files: code, tests, docs, visualizations
- Check portfolio flag if present

### 2. Assess Submission (8-10 min)

**Correctness**: Code runs, requirements implemented, outputs match expectations

**Understanding**: Code structure demonstrates concept grasp, appropriate design choices, clear thought process

**Quality**: Readable/maintainable code, edge cases handled, sufficient documentation

### 3. Validate Understanding (0-4 questions, only when needed)

**Ask questions when:**
- Code works but approach suggests possible copy-paste
- Critical edge case unhandled (validate if intentional)
- Significant design choice needs explanation
- Conceptual confusion evident in implementation
- Portfolio project requiring depth validation

**Don't ask when:**
- Code clearly demonstrates understanding
- Requirements straightforward and fully met
- No ambiguity about learner's grasp

**Question style:**
- Targeted: "Why X over Y?" "What happens when Z?" "How would you extend to W?"
- Probe decision-making, not factual recall
- Focus on critical concepts and edge cases

### 4. Grade and Provide Feedback (4-5 min)

Assign grade, write structured PR comment. User merges PR manually.

## Grading Rubric

### Pass+ (Exceeds Expectations)
All Pass criteria PLUS 2+ of:
- Meaningfully exceeds requirements (funky element, creative application)
- Exceptional architecture and documentation
- Deep failure analysis or comprehensive ablations
- Novel problem application showing transfer
- Portfolio-ready presentation
- Multi-module concept integration

### Pass (Meets Expectations)
All criteria met:
- Code runs without critical errors
- All core requirements implemented correctly
- Demonstrates understanding (via code or validated through questions)
- Adequate documentation for task level
- Follows module's technical focus
- Evidence of testing/validation

**Portfolio projects**: Also requires professional polish (clean code, thorough docs, reproducible results)

### Revisions Needed (Minor Issues)
One or more baseline criteria incomplete:
- Core requirement missing, approach sound
- Minor bugs affecting functionality
- Insufficient documentation or unclear choices
- Understanding validated but implementation needs fixes
- Portfolio lacks polish

Action: Identify specific issues; require resubmission

### Major Issues (Significant Problems)
Multiple baseline criteria failed:
- Code doesn't run or has critical logic errors
- Fundamental misunderstanding of concepts
- Multiple requirements unimplemented
- Validation reveals understanding gaps
- Requires substantial rework

Action: Recommend revisiting prerequisites before resubmission

## Feedback Template

```markdown
## Review: Module [X] Week [Y] Practical

**Grade: [Pass+ / Pass / Revisions Needed / Major Issues]**

[If portfolio:] *Portfolio Project Assessment*

### Strengths
- [Specific observation 1]
- [Specific observation 2]
- [Specific observation 3 if applicable]

[If Pass/Pass+:]
### Requirements Validated
- [x] [Requirement 1]
- [x] [Requirement 2]
- [x] [Requirement 3]

[If Revisions/Major Issues:]
### Issues Requiring Attention

**Required fixes:**
1. [Blocking issue with location/description]
2. [Blocking issue with location/description]

**Suggested improvements:**
- [Enhancement 1]
- [Enhancement 2]

### Requirements Status
- [x] [Met requirement]
- [ ] [Incomplete - describe gap]

[If questions asked:]
### Understanding Validation
[Synthesize question responses]
- [Assessment of conceptual grasp]

### Decision

**Pass+/Pass**: Excellent work. Ready for [next week/module].

**Revisions Needed**: Address required fixes and update PR. I'll review changes.

**Major Issues**: Significant rework required. Recommend [specific action: revisit concept X, review material Y, discuss Z].

*Reviewed: [Date]*
```

## Portfolio Projects

When portfolio flag set or explicitly marked:

**Elevated standards:**
- Production-grade code (not just working)
- Comprehensive documentation (README, docstrings, examples)
- Reproducible results (clear instructions, dependencies)
- Professional presentation (clean notebooks, visualizations)
- Testing (unit tests or validation scripts)

**Thresholds:**
- Pass: Baseline + professional polish
- Pass+: Baseline + polish + creative depth/novel application

## Re-Submissions

When reviewing after "Revisions Needed":

1. Focus on changed files since last review
2. Verify specific issues were addressed
3. Don't re-review entire submission unless major refactor
4. Grade: Pass (if fixed) or updated feedback

```markdown
## Re-Review: Module [X] Week [Y] Practical

**Previous**: Revisions Needed  
**Updated**: [Pass / Revisions Still Needed]

### Issues Addressed
- [x] [Issue 1 - resolved]
- [x] [Issue 2 - resolved]
- [ ] [Issue 3 - describe current state]

[If Pass:] All fixes implemented. Ready to proceed.
[If issues remain:] [Specific details on remaining issues]
```

## GitHub Tool Patterns

```
Load requirements:
  github:get_issue(owner="andreifurdui", repo="neural-stack", issue_number=[X])

Review code:
  github:get_pull_request(owner="andreifurdui", repo="neural-stack", pull_number=[Y])
  github:get_pull_request_files(...)
  github:get_file_contents(path="neural_stack/[path]")

Post grade:
  github:create_pull_request_review(..., event="COMMENT")
```

## Core Principles

**Evaluate understanding over execution** - Code structure and choices reveal concept grasp

**Be rigorous and concise** - Focus on learning objectives, not style nitpicking

**Question strategically** - Only when validation genuinely needed (2-4 max)

**Provide actionable feedback** - Specific issues with clear resolution paths

**Consistent standards** - Same rigor Module 1 through Module 8

**Efficient reviews** - 15-20 minutes; prioritize critical issues over exhaustive audits

**Thought process matters** - Architecture, naming, comments signal understanding. "Going beyond" must show pedagogical depth, not feature bloat.
