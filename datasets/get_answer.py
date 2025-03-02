import os
import re
import math
import json
def load_dataset(dataset_path):
    """
    Load dataset from the specified JSON file.

    Args:
        dataset_path (str): Path to the dataset JSON file.

    Returns:
        list: Loaded dataset as a list of dictionaries.
    """
    if os.path.exists(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            print(f"Dataset loaded successfully with {len(dataset)} samples.")
        return dataset
def load_gsmhard_dataset(filepath):
    dataset = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line.strip())
                dataset.append(record)
    return dataset
def remove_not(x):
    match_number = re.compile('[\$]?\ *10\^[{]?\ *-?[0-9]+\ *[}]?\ *[\$]?')
    result=re.findall(match_number, x)
    if len(result) !=0:
        return re.split(match_number, x)[-1]
    return None
def equiv(model_output, answer, hold):
    model_output=model_output.replace(',', '')
    try:
        ans=float(answer)
        first=math.isclose(float(model_output.strip()), ans, rel_tol=hold)
    except:
        first=False
    try: 
        model=model_output.strip().split()[0]
        second=math.isclose(float(model.strip()), ans, rel_tol=hold)
    except:
        second=False
    if first or second:
        return True
    return False
def parse_not(inputs):
    try:
        if not inputs:
            return '',''
        if '\times' in inputs:
            x,ab=inputs.split('\times')
        elif '\\times' in inputs:
            x,ab=inputs.split('\\times')
        elif '*' in inputs:
            x,ab=inputs.split('*')
        else:
            return inputs
        return x,ab
    except:
        return '',''

def cal_not(inputs):
    
    try:
        x,ab=list(inputs)
        match_number = re.compile('10\^[{]?\ *-?[0-9]+\ *[}]?')
        ab=re.findall(match_number, ab)[0]
        ab=ab[ab.find('^')+1:]
        if '{' in ab:
            ab=ab[ab.find('{')+1:]
        if '}' in ab:
            ab=ab[:ab.find('}')]
        x=x.strip()
        out=float(x)*10**float(ab)
        # print(float(x)*10**float(ab))
        return str(out)
    except:
        print('cal_not error')
    return inputs

def remove_boxed(s):
    left = "oxed{" #change
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        content = s[len(left):-1]

        match = re.search(r"-?\d+\.\d+|-?\d+", content)
        if match:
            return match.group()
        else:
            return None
    except:
        return None
def last_boxed_only_string(string):
        idx = string.rfind("oxed") #change
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None
        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx == None:
            retval = None
        else:
            retval = string[idx:right_brace_idx + 1]

        return retval
def parse_math_answer(raw_string):
    return remove_boxed(last_boxed_only_string(raw_string))