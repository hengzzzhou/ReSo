
## Quickstart

**Clone the repo**

```bash
cd ReSo/
```

**Install packages**
```
conda create -n ReSo python=3.10
conda activate ReSo
pip install --r requirements.txt
```

**You should add API keys in `.env.template` and change its name to `.env`**

```python
QWEN_API_KEY=your_qwen_api_key
QWEN_BASE_URL=your_qwen_base_url
OAI_API_KEY=your_oai_api_key
OAI_BASE_URL=your_oai_base_url
CLAUDE_API_KEY=your_claude_api_key
CLAUDE_BASE_URL=your_claude_base_url
GEMINI_API_KEY=your_gemini_api_key
GEMINI_BASE_URL=your_gemini_base_url
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=your_deepseek_base_url
```



**Run the following commands to reproduce our experiments in the paper**

**train**

```bash
python ReSo/experiments/train_ReSo.py
```



**MATH-MAS**

```bash
python experiments/test_ReSo.py --dataset_path datasets/MATH-MAS/MATH-MAS-Easy.json --plan_mode gt [--random_select] [--error_tolerance <value>]
```
--dataset_path: Path to the Math-MAS dataset JSON file
--plan_mode: Planning mode (gt for ground truth, gpt for GPT-based, or open source model).
--random_select: If specified, enables random agent selection.
--error_tolerance: Error tolerance for answer comparison (default: 0.01)





