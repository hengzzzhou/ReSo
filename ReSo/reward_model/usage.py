import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model.modeling_qwen2_rm import Qwen2ForProcessRewardModel


data = {
    "instruction": "**Previous Steps**: []\n\nyour current sub-question: A $4.0$-foot by $8$-foot rectangular piece of plywood will be cut into $4$ congruent rectangles with no wood left over and no wood lost due to the cuts. What is the positive difference, in feet, between the greatest possible perimeter of a single piece and the least possible perimeter of a single piece?. The answer is recorded as Answer[1]\n\n Conclude the answer by stating 'The answer is therefore \\boxed{[ANSWER]}'",
    "answer": "To solve this problem, we need to determine the possible dimensions of the congruent rectangles that can be obtained by cutting a 4-foot by 8-foot piece of plywood into 4 pieces, and then find the greatest and least possible perimeters of these rectangles.\n\n### Step 1: Determine Possible Dimensions\n\nThe area of the original plywood is \\(4 \\times 8 = 32\\) square feet. Since the plywood is cut into 4 congruent rectangles, each rectangle must have an area of \\(\\frac{32}{4} = 8\\) square feet.\n\nThe dimensions of a rectangle with area 8 can be any pair of factors of 8. The possible integer pairs are:\n- \\(1 \\times 8\\)\n- \\(2 \\times 4\\)\n\n### Step 2: Calculate Perimeters\n\nFor each pair of dimensions, calculate the perimeter of the rectangle:\n- For dimensions \\(1 \\times 8\\), the perimeter is \\(2(1 + 8) = 18\\) feet.\n- For dimensions \\(2 \\times 4\\), the perimeter is \\(2(2 + 4) = 12\\) feet.\n\n### Step 3: Find the Positive Difference\n\nThe greatest possible perimeter is 18 feet, and the least possible perimeter is 12 feet. The positive difference between these perimeters is:\n\\[ 18 - 12 = 6 \\text{ feet} \\]\n\nThe answer is therefore \\(\\boxed{6}\\)."
}
base_model = '/path/to/base/model'
lora_model = '/path/to/lora/model'
device = 'auto'


tokenizer = AutoTokenizer.from_pretrained(base_model)
special = tokenizer.encode('<extra_0>')[0]
model = Qwen2ForProcessRewardModel.from_pretrained(
    base_model,
    device_map=device,
    torch_dtype=torch.bfloat16
).eval()
model.load_adapter(lora_model)


conversation = tokenizer.apply_chat_template([
        {"role": "user", "content": data["instruction"]},
        {"role": "assistant", "content": data["answer"]}
    ], tokenize=False)
inputs = tokenizer.encode(conversation)
inputs.append(special)
inputs = torch.tensor([inputs], device=model.device)
with torch.no_grad():
    logits = model(inputs, return_dict=True).logits
score = F.softmax(logits, dim=-1)[0, -1, 1].item()
print(f"score: {score}")
