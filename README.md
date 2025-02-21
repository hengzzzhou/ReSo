## Run the following commands to reproduce our experiments in the paper

### **MATH-MAS**
```bash
python experiments/test_ReSo.py --dataset_path <path_to_math_mas_dataset> --plan_mode <mode> [--random_select] [--error_tolerance <value>]
```
--dataset_path: Path to the Math-MAS dataset JSON file (default: datasets/MATH-MAS/MATH-MAS-Easy.json).
--plan_mode: Planning mode (gt for ground truth, gpt for GPT-based, or custom task graph parsing).
--random_select: If specified, enables random agent selection.
--error_tolerance: Error tolerance for answer comparison (default: 0.01)

### **train**
```bash
python ReSo/experiments/train_ReSo.py
```

