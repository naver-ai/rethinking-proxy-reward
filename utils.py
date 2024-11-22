import math
import torch
from numbers import Number


def significant(x: Number, ndigits=2) -> Number:
    """
    Cut the number up to its `ndigits` after the most significant
    """
    if isinstance(x, torch.Tensor):
        x = x.item()

    if not isinstance(x, Number) or math.isnan(x) or x == 0:
        return x

    return round(x, ndigits - int(math.floor(math.log10(abs(x)))))


def strip_response_tensors(response_tensors, tokenizer, strip_strings=[], max_length=768):
    STOP_SEQUENCE = '[STOP_SEQUENCE]'
    padding_side_default = tokenizer.padding_side
    responses = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)
    if not strip_strings:
        return response_tensors, responses

    try:
        append_eos = []
        for idx, (rt, r) in enumerate(zip(response_tensors, responses)):
            response = re.sub(f"({'|'.join(strip_strings)})", STOP_SEQUENCE, r.strip("\n "))
            if STOP_SEQUENCE in response:
                response = response.split(STOP_SEQUENCE, 1)[0].strip("\n ")
                append_eos.append(True)
            elif rt[-1] == tokenizer.eos_token_id:
                append_eos.append(True)
            else:
                append_eos.append(False)
            responses[idx] = response.replace("�", "").strip()

        tokenizer.padding_side = "right"
        retokenized_tensors = tokenizer(responses,
                                        padding="longest",
                                        max_length=max_length,
                                        add_special_tokens=False,
                                        return_tensors="pt").input_ids
        outputs = []
        for add_eos, tensor in zip(append_eos, retokenized_tensors):
            leng = tensor.ne(tokenizer.pad_token_id)
            pad_start = leng.sum().item()
            if add_eos:
                if pad_start >= tensor.size(-1):
                    eos_tensor = torch.tensor([tokenizer.eos_token_id])
                    tensor = torch.cat([tensor, eos_tensor], -1)
                else:
                    tensor[pad_start] = tokenizer.eos_token_id
                pad_start += 1
            tensor = tensor[:pad_start].to(response_tensors[0].device)
            outputs.append(tensor)
        tokenizer.padding_side = padding_side_default
    except Exception as e:
        tokenizer.padding_side = padding_side_default
        return response_tensors, responses
    return outputs, responses