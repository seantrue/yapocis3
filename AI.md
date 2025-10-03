# AI Contributions to Yapocis

This document tracks contributions made by AI assistants (primarily Claude) versus the human maintainer.

## Summary Statistics

- **Total AI-assisted commits**: ~8
- **Major features developed with AI assistance**: Metal backend fixes, coordinate system standardization, renamed truetrace module, validation framework fixes, numba backend completeness
- **Lines of code**: Tracking started 2025-10-03

## Detailed Contributions

### 2025-10-03: Validation Framework Fixes and Numba HSI Completion

**AI Contribution (Claude):**
- Fixed test_manual_validation_context test failure
  - Analyzed validation framework auto-promotion logic
  - Added cleanup of stale validation references
  - Enhanced test to verify both auto-promotion and subsequent validation
  - Improved test documentation and comments
- Completed Numba HSI backend implementation
  - Identified missing `hsi2rgb` function in numba/kernels/hsi.py
  - Implemented `hsi2rgb` matching OCL kernel interface
  - Maintained proper channel swizzle (r,b,g) for consistency
  - Verified all numba backend tests pass
- Lines changed: ~30 additions across 2 files

**Files modified:**
- `tests/test_validation_example.py` - Fixed validation test with proper cleanup and verification
- `yapocis/numba/kernels/hsi.py` - Added missing `hsi2rgb` function

**Testing:**
- All numba backend tests pass (18 tests)
- Validation framework tests pass
- HSI color conversion tests pass on numba backend

**Human Contribution:**
- Bug identification and reporting
- Testing and verification

---

### 2025-10-03: Rename sdfpath to truetrace + Numba documentation

**AI Contribution (Claude):**
- Renamed sdfpath module to truetrace across entire codebase
  - Files renamed: 5 (interface, library, OCL kernel, Metal kernel, Numba kernel)
  - Class renamed: SDFPath → TrueTrace
  - Interface declaration updated
  - Test file updated with new imports
  - Documentation updated in CLAUDE.md and README.md
- Updated README.md to document Numba backend
  - Added Numba to supported backends list
  - Documented CPU fallback functionality
  - Added test markers and dependencies
  - Added backend selection examples
- Created AI.md to track contributions

**Files modified:**
- `yapocis/interfaces/sdfpath.interface` → `truetrace.interface`
- `yapocis/lib/sdfpath.py` → `truetrace.py`
- `yapocis/kernels/sdfpath.mako` → `truetrace.mako`
- `yapocis/metalkernels/sdfpath.mako` → `truetrace.mako`
- `yapocis/numba/kernels/sdfpath.py` → `truetrace.py`
- `tests/test_truetrace_equiv.py` → `test_truetrace.py`
- `README.md`
- `CLAUDE.md`
- `AI.md` (created)

**Human Contribution:**
- Design decision to rename sdfpath to truetrace
- Review and approval of changes

---

### 2025-10 (earlier): Metal Backend Column-Major Coordinate System Fixes

**AI Contribution (Claude):**
- Analyzed OCL vs Metal coordinate system implementations
- Identified bugs in Metal helpers.mako (row-major vs column-major)
- Fixed Metal kernel.py width/height handlers
- Fixed graphics4.mako composite kernel coordinate swizzles (.xy/.zw → .yx/.wz)
- Fixed graphics.mako (single-channel) composite kernel swizzles
- Fixed abstract_graphics.py coerce_colors for single-channel graphics
- Created comprehensive test scripts in claude-debug/

**Files modified:**
- `yapocis/metalkernels/helpers.mako`
- `yapocis/metallic/kernel.py`
- `yapocis/metalkernels/graphics4.mako`
- `yapocis/metalkernels/graphics.mako`
- `yapocis/lib/abstract_graphics.py`

**Commits:**
- `bea0457` - Fix Metal composite kernel coordinate swizzle for column-major layout
- `0eb50ab` - Fix Metal graphics (single-channel) composite kernel coordinate swizzle
- Additional commit for coerce_colors fix

**Human Contribution:**
- Bug identification and reproduction
- Design decision: maintain column-major layout for 'older' branch
- Testing and validation of fixes

---

## Methodology

**AI Role:**
- Implementation of approved changes
- Code analysis and bug identification
- Documentation updates
- Test creation and debugging
- Commit message drafting

**Human Role:**
- Architecture decisions
- Requirements specification
- Final review and approval
- Git operations (when not delegated)
- Strategic direction

## Notes

- AI contributions are supervised and reviewed by the human maintainer
- All commits are attributed to the human maintainer in git history
- This tracking started 2025-10-03; earlier AI contributions are documented retroactively based on session summaries
