# CI/CD Integration

This directory contains example GitHub Actions workflows for running benchmarks on different hardware configurations.

## Workflows

### `benchmark.yml`

Example workflow that runs benchmarks on both CPU (GitHub Actions) and GPU (RunsOn) runners.

## RunsOn Setup

[RunsOn](https://runs-on.com/) provides self-hosted GPU runners for GitHub Actions. To use GPU runners:

1. Install the RunsOn GitHub App on your organization
2. Configure your AWS account with RunsOn
3. Use `runs-on: [runs-on, runner=gpu, image=quantecon-gpu-ubuntu2404]` in your workflow

### QuantEcon GPU AMI

QuantEcon maintains a custom AMI (`quantecon-gpu-ubuntu2404`) with:
- Ubuntu 24.04
- NVIDIA drivers
- CUDA toolkit
- Miniconda with JAX (CUDA support)
- Common scientific Python packages

## Usage

Copy the example workflow to your repository's `.github/workflows/` directory and customize as needed.

## Cost Considerations

- **GitHub Actions (CPU)**: Free for public repositories
- **RunsOn (GPU)**: Pay-per-use based on AWS instance costs
  - `g4dn.xlarge`: ~$0.526/hour (1 GPU, good for most benchmarks)
  - `g4dn.2xlarge`: ~$0.752/hour (1 GPU, more CPU/RAM)

Use the `if: github.event_name == 'workflow_dispatch'` pattern to make GPU runs manual-only and avoid unexpected costs.
