# AI Contributions

Starting in 2024, yapocis development has been significantly accelerated through AI assistance, primarily using Claude (Anthropic). This page documents the nature and scope of AI contributions to the project.

## Philosophy

AI assistance in yapocis follows these principles:

1. **Human direction** - The maintainer sets goals, reviews outputs, and makes architectural decisions
2. **AI acceleration** - Claude handles implementation details, boilerplate, and exploration
3. **Transparency** - AI contributions are tracked and documented
4. **Quality gates** - All AI-generated code goes through the same review and testing as human code

## Areas of AI Contribution

### Documentation
- This documentation site structure and content
- README improvements and modernization
- Inline code documentation and docstrings
- Architecture explanations

### Code Infrastructure
- Modern Python packaging (pyproject.toml, hatchling)
- Testing infrastructure improvements
- Performance tracking system design
- Validation framework implementation

### Backend Development
- CUDA backend implementation assistance
- Numba backend kernel translations
- Cross-backend compatibility testing
- Bug fixes across all backends

### Tooling
- CLI tools and scripts
- Development workflow automation
- Code quality tooling configuration

## What AI Doesn't Do

- **Architectural decisions** - The human maintainer designs the system
- **Algorithm design** - Novel image processing algorithms come from human research
- **Release decisions** - Version bumps, releases, and deployment are human-controlled
- **Final review** - All code is reviewed before merge

## Commit Attribution

AI-assisted commits include:

```
🤖 Generated with Claude Code (https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

## Statistics

*This section will be updated with contribution metrics as the project evolves.*

## Why Document This?

Transparency about AI assistance serves several purposes:

1. **Reproducibility** - Others can understand how the codebase evolved
2. **Trust** - Users know what they're getting
3. **Learning** - The AI-assisted development process may help others
4. **Accountability** - Clear attribution of human vs AI contributions
