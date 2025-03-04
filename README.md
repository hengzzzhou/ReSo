
## Quickstart

**Clone the repo**

```bash
cd ReSo/
```

**Install packages**
```
conda create -n ReSo python=3.10
conda activate ReSo
pip install -r requirements.txt
```


**Configure API Keys:**  
   Add the required API keys in the `.env.template` file and rename it to `.env`.  
   In the `.env` file, you can fill in the corresponding API keys and base URLs according to the following template:

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

**Adjust Model Agent Information:**  
   If you need to modify the model agent information, please edit the `config.ini` file.

**Adjust Hyperparameters:**  
   To change the hyperparameter configuration, please edit the `config_hyper.ini` file.




**Run the following commands to reproduce our experiments in the paper**

**train**

```bash
python ReSo/experiments/train_ReSo.py --dataset_path 
```



**MATH-MAS**

```bash
python experiments/test_ReSo.py --dataset_path datasets/MATH-MAS/MATH-MAS-Easy.json --plan_mode gt [--random_select] [--error_tolerance <value>]
```

### Data-Mas Generation Process

1. **Subtask Dataset Preparation**  
   The processed subtasks are stored in the `datasets/sub_question` directory, containing:  
   - Original problem statements  
   - Ground-truth answers  
   - Replaced unknown variables (placeholders)  
   - Variable-substituted questions  

   You can extract appropriate unknown variables using either:  
   - *Regular expressions* for pattern-based extraction  
   -  Use *LLM*(GPT-4o) to label

2. **Data Generation Execution**  
   Run the data generation script with customizable parameters:  
   ```bash
   python datasets/data_gen.py -n 91 -c 3
   ```
   - `-n`: Number of instances to generate
   - `-c`: Context complexity level 









