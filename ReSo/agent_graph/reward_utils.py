from datasets.get_answer import equiv, parse_math_answer

def reward_model(answer, gt_answer):
    """
    Evaluate the answer against the ground truth answer.
    
    Returns 1 if the answer is correct (within tolerance), otherwise 0.
    """
    predict_answer = parse_math_answer(answer)
    try:
        is_solved = equiv(predict_answer, gt_answer, 0.05)
    except Exception:
        is_solved = False
    return 1 if is_solved else 0
