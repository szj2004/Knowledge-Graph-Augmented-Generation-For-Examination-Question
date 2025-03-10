"""
调用gpt3.5
"""

from openai import OpenAI
def query_chatgpt_model( prompt: str):
    client = OpenAI(
        api_key = "sk-alSpLVParko5PgY0C7DcFa2c7e02455aA8E567250a18B6C8",
        base_url = "https://one-api.aiporters.com/v1",
    )
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-3.5-turbo",
        )

        output = chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(e)
    return output




if __name__ == '__main__':
    prompt = '公元前841年，周厉王贪财好利，为政暴虐，引发了“国人暴动”。周厉王出逃，大臣召公、周公共同执政，史称“共和行政”。公元前771年，西北游牧民族犬戎乘西周王室内乱，攻破镐京，杀死周幽王，西周灭亡。 展示3种对该段话进行关系提取的推理思路，并输出推理结果。'
    prompt = '请给出3种关系提取的推理思路。'
    res = query_chatgpt_model(prompt)
    print(res)