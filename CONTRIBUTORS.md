# Contributors

## Core Contributors

### Sean D. True
- **Role**: Creator and Lead Maintainer
- **Organization**: Swapwizard LLC
- **Email**: sean.true@gmail.com
- **Contributions**:
  - Original architecture and design (2011)
  - Core IDL-based interface system
  - Multi-backend architecture (OpenCL, CUDA, Metal, Numba)
  - High-level graphics and image processing libraries
  - Performance optimization and validation framework
  - Modern packaging infrastructure (2024-2025)

## AI Assistance

### Claude (Anthropic)
- **Role**: Development Assistant
- **Contributions**: See [AI.md](AI.md) for detailed tracking of AI-assisted development
  - Package modernization (hatchling, modern tooling)
  - Test infrastructure improvements
  - Performance tracking system (DuckDB integration)
  - Validation framework development
  - Documentation enhancements
  - Automated changelog system

## How to Contribute

We welcome contributions! Here's how you can help:

1. **Report Issues**: Use the [GitHub issue tracker](https://github.com/seantrue/yapocis3/issues)
2. **Submit Pull Requests**: Fork the repo and submit PRs for bug fixes or features
3. **Documentation**: Help improve documentation and examples
4. **Testing**: Add tests for new features or backends

### Contribution Guidelines

- Follow the existing code style (use `make format` and `make lint`)
- Write tests for new features
- Use conventional commits for changelog generation
- Update documentation as needed

### Development Setup

```bash
# Clone the repository
git clone https://github.com/seantrue/yapocis3.git
cd yapocis3

# Install development dependencies
make install-dev

# Run tests
make test

# Run code quality checks
make check
```

## Acknowledgments

- **PyOpenCL Community**: For the excellent OpenCL Python bindings
- **Andreas Klöckner**: For PyOpenCL and ongoing support
- **Numba Team**: For JIT compilation capabilities
- **Apple**: For Metal compute framework
- **NVIDIA**: For CUDA support

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.
