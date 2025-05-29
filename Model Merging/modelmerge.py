import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

smol_model = "unsloth/Llama-3.2-1B-Instruct"
big_model = "unsloth/Llama-3.2-3B-Instruct"

model1 = AutoModelForCausalLM.from_pretrained(smol_model, device_map="auto", torch_dtype=torch.bfloat16)
model2 = AutoModelForCausalLM.from_pretrained(big_model, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(smol_model)


class LlamaMerge(nn.Module):
    def __init__(self, model1, model2):
        super(LlamaMerge, self).__init__()
        self.model1 = model1
        self.model2 = model2
    
    def print_info(self, model):
        for k, v in model.state_dict().items():
            print(k, v.shape)
    
    def resize_weights(self, W1, W2):
        if W1.shape != W2.shape:
            W1 = torch.nn.functional.interpolate(W1.unsqueeze(0).unsqueeze(0), size=W2.shape[1:], mode='bilinear', align_corners=False)
            W1 = W1.squeeze(0).squeeze(0)
        return W1, W2
    
    def merge_weights(self, model1, model2, num_layers=5, method="linear"):
        for i in range(num_layers):
            if method == "linear":
                model1 = self.merge_weights_linear(model1, model2, num_layers, alpha=0.3)
            elif method == "slerp":
                model1 = self.merge_weights_slerp(model1, model2, num_layers, interpolation_param=0.3)
        return model1

    def merge_weights_linear(self, model1, model2, num_layers, alpha=0.3):
        for i in range(num_layers):
            q_1, k_1, v_1 = model1.layers[i].self_attn.q_proj.weight, model1.layers[i].self_attn.k_proj.weight, model1.layers[i].self_attn.v_proj.weight
            q_2, k_2, v_2 = model2.layers[i].self_attn.q_proj.weight, model2.layers[i].self_attn.k_proj.weight, model2.layers[i].self_attn.v_proj.weight

            if q_1.shape != q_2.shape:
                q_1, q_2 = self.resize_weights(q_1, q_2)
            if k_1.shape != k_2.shape:
                k_1, k_2 = self.resize_weights(k_1, k_2)
            if v_1.shape != v_2.shape:
                v_1, v_2 = self.resize_weights(v_1, v_2)

            q_merged = alpha * q_1 + (1 - alpha) * q_2
            k_merged = alpha * k_1 + (1 - alpha) * k_2
            v_merged = alpha * v_1 + (1 - alpha) * v_2

            model1.layers[i].self_attn.q_proj.weight.data = q_merged
            model1.layers[i].self_attn.k_proj.weight.data = k_merged
            model1.layers[i].self_attn.v_proj.weight.data = v_merged

            gate_1, up_1, down_1 = model1.layers[i].mlp.gate_proj.weight, model1.layers[i].mlp.up_proj.weight, model1.layers[i].mlp.down_proj.weight
            gate_2, up_2, down_2 = model2.layers[i].mlp.gate_proj.weight, model2.layers[i].mlp.up_proj.weight, model2.layers[i].mlp.down_proj.weight

            if gate_1.shape != gate_2.shape:
                gate_1, gate_2 = self.resize_weights(gate_1, gate_2)
            if up_1.shape != up_2.shape:
                up_1, up_2 = self.resize_weights(up_1, up_2)
            if down_1.shape != down_2.shape:
                down_1, down_2 = self.resize_weights(down_1, down_2)

            gate_merged = alpha * gate_1 + (1 - alpha) * gate_2
            up_merged = alpha * up_1 + (1 - alpha) * up_2
            down_merged = alpha * down_1 + (1 - alpha) * down_2

            model1.layers[i].mlp.gate_proj.weight.data = gate_merged
            model1.layers[i].mlp.up_proj.weight.data = up_merged
            model1.layers[i].mlp.down_proj.weight.data = down_merged

        return model1

    def merge_weights_slerp(self, model1, model2, num_layers, interpolation_param=0.3):
        for i in range(num_layers):
            q_1, k_1, v_1 = model1.layers[i].self_attn.q_proj.weight, model1.layers[i].self_attn.k_proj.weight, model1.layers[i].self_attn.v_proj.weight
            q_2, k_2, v_2 = model2.layers[i].self_attn.q_proj.weight, model2.layers[i].self_attn.k_proj.weight, model2.layers[i].self_attn.v_proj.weight

            if q_1.shape != q_2.shape:
                q_1, q_2 = self.resize_weights(q_1, q_2)
            if k_1.shape != k_2.shape:
                k_1, k_2 = self.resize_weights(k_1, k_2)
            if v_1.shape != v_2.shape:
                v_1, v_2 = self.resize_weights(v_1, v_2)

            q_merged = slerp(q_1, q_2, interpolation_param)
            k_merged = slerp(k_1, k_2, interpolation_param)
            v_merged = slerp(v_1, v_2, interpolation_param)

            model1.layers[i].self_attn.q_proj.weight.data = q_merged
            model1.layers[i].self_attn.k_proj.weight.data = k_merged
            model1.layers[i].self_attn.v_proj.weight.data = v_merged

            gate_1, up_1, down_1 = model1.layers[i].mlp.gate_proj.weight, model1.layers[i].mlp.up_proj.weight, model1.layers[i].mlp.down_proj.weight
            gate_2, up_2, down_2 = model2.layers[i].mlp.gate_proj.weight, model2.layers[i].mlp.up_proj.weight, model2.layers[i].mlp.down_proj.weight

            if gate_1.shape != gate_2.shape:
                gate_1, gate_2 = self.resize_weights(gate_1, gate_2)
            if up_1.shape != up_2.shape:
                up_1, up_2 = self.resize_weights(up_1, up_2)
            if down_1.shape != down_2.shape:
                down_1, down_2 = self.resize_weights(down_1, down_2)

            gate_merged = slerp(gate_1, gate_2, interpolation_param)
            up_merged = slerp(up_1, up_2, interpolation_param)
            down_merged = slerp(down_1, down_2, interpolation_param)

            model1.layers[i].mlp.gate_proj.weight.data = gate_merged
            model1.layers[i].mlp.up_proj.weight.data = up_merged
            model1.layers[i].mlp.down_proj.weight.data = down_merged

        return model1


def slerp(W1, W2, t):
    if W1.shape != W2.shape:
        raise ValueError("Shape mismatch between weights.")

    W1_flat = W1.flatten()
    W2_flat = W2.flatten()

    dot = torch.dot(F.normalize(W1_flat), F.normalize(W2_flat))
    dot = torch.clamp(dot, -1.0, 1.0)

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    coeff1 = torch.sin((1 - t) * theta) / sin_theta
    coeff2 = torch.sin(t * theta) / sin_theta

    return coeff1 * W1 + coeff2 * W2


# Merge the models' weights (using linear interpolation)
llama_merge = LlamaMerge(model1, model2)
merged_model = llama_merge.merge_weights(model1, model2, num_layers=5, method="linear")
# Prepare the input text
input_text = "Once upon a time"

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt").to(model1.device)

# Perform inference with the merged model
with torch.no_grad():
    generated_ids = merged_model.generate(inputs['input_ids'], max_length=50)

# Decode the generated text
output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(output_text)
