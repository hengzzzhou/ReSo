# Configuration Guide for ReSo

This document explains the configuration system for ReSo multi-agent mathematical reasoning framework.

## Configuration Files

### `config.ini` - Agent Pool Configuration
Defines specialized expert agents with their capabilities and prompts.

### `config_hyper.ini` - Hyperparameters  
Controls agent selection, exploration, and optimization behavior.

### `.env` - API Credentials
Contains API keys and endpoints for LLM providers.

## Agent Configuration

Each agent is defined with:
- **Model**: The underlying LLM (GPT-4, Gemini, etc.)
- **Role**: Expert specialization (MechanicsExpert, AlgebraExpert, etc.)  
- **Prompt**: Detailed system prompt defining expertise and behavior

### Available Expert Roles

#### Science Domains
- MechanicsExpert, ElectromagnetismExpert
- Thermodynamics&OpticsExpert
- InorganicChemistryExpert, OrganicChemistryExpert, PhysicalChemistryExpert

#### Mathematics Domains  
- AlgebraExpert, GeometryExpert
- Probability&StatisticsExpert

## Key Hyperparameters

### Agent Selection Weights
- `similarity_weight`: Importance of expertise relevance (default: 0.6)
- `reputation_weight`: Historical performance weight (default: 2.0)
- `cost_weight`: Cost consideration weight (default: 1.0)

### Selection Constraints
- `THRESHOLD`: Minimum similarity for selection (default: 0.4)
- `EXPLORATION_CONST`: Exploration vs exploitation balance (default: 0.3)
- `TOP_K`: Number of agents to select (default: 1)

### Training Settings
- `error_tolerance`: Numerical comparison tolerance (default: 0.15)
- `max_episodes`: Training episode limit (default: 1000)
- `save_frequency`: Checkpoint frequency (default: 100)

## Environment Setup

Required API keys in `.env`:
```bash
OAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key  
CLAUDE_API_KEY=your_claude_key
QWEN_API_KEY=your_qwen_key
DEEPSEEK_API_KEY=your_deepseek_key
```

See full documentation for detailed configuration options and best practices.