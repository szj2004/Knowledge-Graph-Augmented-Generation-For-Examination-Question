import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# def api_answer( prompt: str):
#     prompt=prompt+"在结果前加入一个标记(##答案)"
#     print(prompt)
#     model_path = "/mnt/yunpan/deepseek/models/dir/resource/model/DeepSeek-R1-Distill-Qwen-14B"
#     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     # 生成文本
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=200,
#         temperature=0.7,
#         top_p=0.9
#     )
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     print(11)
#     print(response)
#     return response
####硅基流动api调用
from openai import OpenAI
def api_answer( prompt: str):
    prompt = prompt
    client = OpenAI(
        api_key = "sk-vrnyojexauzlilufysynpaobvurzjrvemkvonywophkttoab",
        base_url = "https://api.siliconflow.cn/v1",
    )
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        )

        output = chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(e)
    return output

if __name__ == '__main__':
    print(3)
