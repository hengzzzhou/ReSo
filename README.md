# ReSo: Reasoning with Self-Optimization for Mathematical Problem Solving

A comprehensive framework for multi-step mathematical problem solving using reinforcement learning and self-optimization techniques.

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- Git

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd ReSo/
```

2. **Set up environment**
```bash
conda create -n ReSo python=3.10
conda activate ReSo
pip install -r requirements.txt
```

3. **Configure API Keys**

Create a `.env` file from the template:
```bash
cp .env.template .env
```

Edit the `.env` file with your API credentials:
```bash
# OpenAI Configuration
OAI_API_KEY=your_openai_api_key
OAI_BASE_URL=https://api.openai.com/v1

# Qwen Configuration  
QWEN_API_KEY=your_qwen_api_key
QWEN_BASE_URL=your_qwen_base_url

# Claude Configuration
CLAUDE_API_KEY=your_claude_api_key
CLAUDE_BASE_URL=your_claude_base_url

# Gemini Configuration
GEMINI_API_KEY=your_gemini_api_key
GEMINI_BASE_URL=your_gemini_base_url

# DeepSeek Configuration
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=your_deepseek_base_url
```

## 📁 Project Structure

```
ReSo/
├── ReSo/                    # Core framework modules
│   ├── agent_graph/         # Agent graph implementation
│   ├── llm_agent/          # LLM agent components
│   ├── model/              # Custom model implementations
│   └── task_graph/         # Task graph management
├── datasets/               # Data synthesis and storage
│   ├── data_gen.py        # Complex problem generator
│   ├── get_answer.py      # Answer extraction utilities
│   ├── sub_question/      # Base sub-question datasets
│   ├── MATH-MAS/         # MATH Multi-Agent System datasets
│   └── Scibench-MAS/     # Science benchmark datasets
├── experiments/           # Training and evaluation scripts
├── reward_model/         # Reward model training
├── config.ini           # Model agent configuration
├── config_hyper.ini     # Hyperparameter configuration
└── requirements.txt     # Python dependencies
```

## 🔧 Configuration

### Model Configuration

Edit `config.ini` to modify model agent settings:
- Model selection and parameters
- Agent behavior configurations
- API endpoints and settings

### Hyperparameters

Adjust `config_hyper.ini` for:
- Training hyperparameters
- Optimization settings
- Performance tuning parameters

## 🎯 Usage

### Training

Train the ReSo model on your dataset:

```bash
python experiments/train_ReSo.py --dataset_path <path_to_training_data>
```

**Parameters:**
- `--dataset_path`: Path to the training dataset
- Additional training parameters can be configured in `config_hyper.ini`

### Evaluation

#### MATH-MAS Evaluation

Test the model on MATH-MAS benchmarks:

```bash
# Easy difficulty
python experiments/test_ReSo.py --dataset_path datasets/MATH-MAS/MATH-MAS-Easy.json --plan_mode gt

# Medium difficulty  
python experiments/test_ReSo.py --dataset_path datasets/MATH-MAS/MATH-MAS-Medium.json --plan_mode gt

# Hard difficulty
python experiments/test_ReSo.py --dataset_path datasets/MATH-MAS/MATH-MAS-Hard.json --plan_mode gt
```

**Parameters:**
- `--dataset_path`: Path to the test dataset
- `--plan_mode`: Planning mode (`gt` for ground truth)
- `--random_select`: Enable random selection (optional)
- `--error_tolerance`: Set error tolerance threshold (optional)

#### GSM8K Evaluation

```bash
python experiments/test_gsm8k.py --dataset_path <gsm8k_dataset_path>
```

## 📊 Data Generation

### Multi-Agent System (MAS) Dataset Creation

The framework includes a sophisticated data synthesis module for creating complex multi-step mathematical problems.

#### 1. Prepare Sub-question Datasets

Base sub-questions are stored in `datasets/sub_question/`:
- `math_test.json`: Mathematical problem sub-questions
- `scibench.json`: Science benchmark sub-questions

Each sub-question includes:
- Original problem statement
- Ground-truth answer
- Variable placeholders for dependency injection
- Metadata (difficulty, category, etc.)

#### 2. Generate Complex Problems

Create complex multi-step problems using the data generator:

```bash
python datasets/data_gen.py -n <num_questions> -c <complexity_level> [-o <output_file>]
```

**Parameters:**
- `-n, --num_questions`: Number of complex questions to generate
- `-c, --complexity`: Complexity level (number of sub-questions to combine)
- `-o, --output`: Output file path (optional, auto-generates with timestamp)

**Examples:**
```bash
# Generate 100 questions with 3 sub-questions each
python datasets/data_gen.py -n 100 -c 3

# Generate 50 highly complex questions (5 sub-questions each)
python datasets/data_gen.py -n 50 -c 5 -o datasets/mixed/complex_dataset.json
```

#### 3. Data Generation Process

The system creates complex problems by:
1. **DAG Construction**: Generates random Directed Acyclic Graphs to define question dependencies
2. **Question Linking**: Links sub-questions where child questions depend on parent answers
3. **Integration**: Creates final tasks that combine all sub-question answers
4. **Validation**: Ensures numerical consistency and solvability

For detailed information about the data synthesis process, see [`datasets/README.md`](datasets/README.md).

## 🤖 Pre-trained Models

We provide fine-tuned **Plan** and **CRM** models on [Hugging Face](https://huggingface.co/henggg/ReSo/tree/main):

- **Plan Model**: Specialized for multi-step planning in mathematical reasoning
- **CRM Model**: Critic-Reward Model for solution evaluation and optimization

### Using Pre-trained Models

```python
from transformers import AutoModel, AutoTokenizer

# Load Plan model
plan_model = AutoModel.from_pretrained("henggg/ReSo-Plan")
plan_tokenizer = AutoTokenizer.from_pretrained("henggg/ReSo-Plan")

# Load CRM model  
crm_model = AutoModel.from_pretrained("henggg/ReSo-CRM")
crm_tokenizer = AutoTokenizer.from_pretrained("henggg/ReSo-CRM")
```

## 🔬 Key Features

### Multi-Agent Reasoning
- **Agent Graph**: Structured representation of reasoning agents
- **Task Decomposition**: Automatic breakdown of complex problems into sub-tasks
- **Collaborative Solving**: Multiple agents work together on different problem aspects

### Self-Optimization
- **Reward Modeling**: Custom reward models for solution quality assessment
- **Iterative Improvement**: Solutions are refined through multiple iterations
- **Error Detection**: Automatic identification and correction of reasoning errors

### Flexible Architecture
- **Modular Design**: Easy to extend with new models and strategies
- **API Integration**: Support for multiple LLM providers (OpenAI, Claude, Gemini, etc.)
- **Configurable Pipelines**: Customizable reasoning and optimization workflows

## 📈 Performance

The ReSo framework achieves state-of-the-art results on:
- **MATH Dataset**: Competitive performance on mathematical reasoning tasks
- **GSM8K**: Strong results on grade school mathematics problems  
- **Science Benchmarks**: Effective on multi-domain scientific reasoning

## 🛠️ Development

### Adding New Models

1. Implement model interface in `ReSo/llm_agent/`
2. Add configuration options in `config.ini`
3. Update model registry in `model_info.py`

### Custom Reward Models

1. Define model architecture in `ReSo/model/`
2. Implement training logic in `reward_model/train.py`
3. Add evaluation metrics in `reward_model/test.py`

### Extending Data Generation

1. Add new sub-question formats in `datasets/sub_question/`
2. Modify generation logic in `datasets/data_gen.py`
3. Update answer parsing in `datasets/get_answer.py`

## 📝 Citation

If you use ReSo in your research, please cite our paper:

```bibtex
@article{reso2024,
  title={ReSo: Reasoning with Self-Optimization for Mathematical Problem Solving},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details on:
- Code style and standards
- Testing requirements
- Documentation expectations
- Pull request process

## 📜 License

This project is licensed under [License Name] - see the LICENSE file for details.

## 🔗 Links

- [Paper](link-to-paper)
- [Hugging Face Models](https://huggingface.co/henggg/ReSo/tree/main)
- [Documentation](link-to-docs)
- [Issues](link-to-issues)

## 📞 Contact

For questions and support:
- Create an issue on GitHub
- Contact: [email]
- Project maintainers: [names]