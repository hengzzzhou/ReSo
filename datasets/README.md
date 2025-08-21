# Data Synthesis Module

This module provides tools for generating complex multi-step mathematical problems from simple sub-questions using a Directed Acyclic Graph (DAG) structure.

## Overview

The data synthesis system creates complex mathematical problems by:
1. Loading base sub-questions from existing datasets
2. Generating random DAG structures to define dependencies between questions
3. Linking questions where child questions depend on answers from parent questions
4. Creating final integration tasks that combine all sub-question answers

## Files

### Core Scripts

- **`data_gen.py`** - Main data generation script that creates complex multi-step problems
- **`get_answer.py`** - Utilities for parsing and extracting numerical answers from various text formats

### Data Files

- **`sub_question/`** - Directory containing base sub-question datasets
  - `math_test.json` - Mathematics sub-questions from various domains
  - `scibench.json` - Science benchmark sub-questions
- **`mixed/`** - Directory for generated complex problem datasets
- **Source datasets** in parent directories (MATH-MAS, Scibench-MAS)

## Usage

### Generating Complex Problems

Use the `data_gen.py` script to generate complex multi-step mathematical problems:

```bash
python datasets/data_gen.py -n <num_questions> -c <complexity> [-o <output_file>]
```

**Parameters:**
- `-n, --num_questions`: Number of complex questions to generate (required)
- `-c, --complexity`: Complexity level - number of sub-questions to combine (required)  
- `-o, --output`: Output file path (optional, auto-generates if not provided)

**Examples:**

```bash
# Generate 100 questions with complexity level 3
python datasets/data_gen.py -n 100 -c 3

# Generate 50 questions with complexity level 5, save to specific file
python datasets/data_gen.py -n 50 -c 5 -o datasets/mixed/my_complex_dataset.json
```

### Output Format

Generated datasets contain complex problems with the following structure:

```json
{
  "source": "Counting & Probability Algebra",
  "Q_ID": ["question_id_1", "question_id_2", ...],
  "complexity": 3,
  "dag": {"0": [1], "1": [2], "2": []},
  "node_info": {
    "0": {
      "source": "Counting & Probability",
      "question_id": "question_id_1",
      "problem": "Original problem text...",
      "problem_UNK": "Modified problem with UNK placeholders...",
      "answer_val": 42.0,
      "in_edges": [],
      "out_edges": [1]
    }
  },
  "problem_text": "Complete multi-step problem description...",
  "problem_text_sort": "Problems sorted by topological order...",
  "answer_number": "final_answer",
  "gt_plan": ["step1", "step2", ...],
  "gt_subtask": [answer1, answer2, ..., final_answer]
}
```

### Answer Extraction

The `get_answer.py` module provides utilities for extracting numerical answers:

```python
from get_answer import parse_math_answer, parse_gpqa_answer, load_dataset

# Parse mathematical answer from boxed notation
answer = parse_math_answer("The solution is \\boxed{42}")

# Load dataset
dataset = load_dataset("path/to/dataset.json")

# Check numerical equivalence
is_correct = is_numerically_equivalent("42.0", "42", tolerance=1e-6)
```

## Algorithm Details

### DAG Generation

The system generates random Directed Acyclic Graphs where:
1. Each node (except the first) has at least one parent to ensure connectivity
2. Additional random edges are added to create complex dependencies
3. Topological ordering ensures proper question sequencing

### Question Linking

Child questions are modified so their unknown values depend on parent answers:
- Parent answers are summed
- An adjustment factor is calculated: `adjustment = child_value - sum(parent_answers)`
- The child question's UNK placeholder is replaced with a dependency description

### Final Integration

All sub-question answers are combined using multiplication:
- Final question: "Calculate Answer[0] * Answer[1] * ... * Answer[n-1]"
- Final answer: Product of all sub-question answers

## Data Format Requirements

### Sub-question Format

Base sub-questions should follow this JSON structure:

```json
{
  "problem": "Original problem statement",
  "question_id": "unique_identifier",
  "answer_number": 42.0,
  "q_vals": 72.0,
  "problem_UNK": "Problem with UNK(a constant can be caculuted by other answers) placeholder",
  "type": "Problem category",
  "level": "Difficulty level"
}
```

### Key Fields

- **`problem`**: Original problem statement
- **`problem_UNK`**: Modified version with UNK placeholder for dependency injection
- **`q_vals`**: Numerical value that can be replaced by parent dependencies
- **`answer_number`**: Correct numerical answer
- **`question_id`**: Unique identifier for the question

## Dependencies

The module requires:
- Python 3.7+
- Standard libraries: `json`, `random`, `re`, `math`, `argparse`, `collections`, `typing`

## Output Files

Generated datasets are saved as JSON files with timestamps:
- Default naming: `mix_math_test_{complexity}_{timestamp}.json`
- Custom naming: Use `-o` parameter to specify output path

## Notes

- The system ensures no duplicate question combinations by tracking used Q_ID sets
- Questions are presented both in original order and topologically sorted order
- The magnitude checking ensures reasonable numerical relationships between parent and child questions
- All generated problems include ground truth plans and subtask answers for evaluation