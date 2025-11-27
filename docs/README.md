# Documentation and Findings

This directory contains documentation of benchmark findings and investigations.

## Investigation Reports

### lax.scan GPU Performance (November 2025)

**Issue**: `lax.scan` with 10M iterations took 81s on GPU vs 0.06s on CPU

**Root Cause**: Kernel launch overhead, not CPU-GPU synchronization
- XLA generates 3 separate kernels per iteration (mul, add, dynamic_update_slice)
- Each kernel launch has ~2-3µs overhead
- 3 kernels × ~2-3µs = ~6-9µs per iteration (matches measured ~8µs)

**Evidence**:
1. TensorBoard profiler showed 1000 calls each for mul/add/dynamic_update_slice
2. Nsight Systems timeline showed characteristic pattern of tiny kernel launches with gaps
3. Time scales linearly with iteration count (constant per-iteration overhead)

**Solution**: Use `device=cpu` for sequential scalar operations

**Reference**: [QuantEcon PR #437](https://github.com/QuantEcon/lecture-python-programming.myst/pull/437)

---

## Adding New Findings

When documenting new benchmark findings:

1. Create a markdown file with the investigation details
2. Include:
   - Problem description
   - Root cause analysis
   - Evidence/data
   - Solution/workaround
   - References

3. Update the main README with a summary
