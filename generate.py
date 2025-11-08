import os
from openai import OpenAI
from typing import Literal

class DatasetGenerator:
    def __init__(
        self,
        url:str = 'https://api.deepseek.com',
        api_key_environ_name:str = 'DEEPSEEK_API_KEY',
        default_model:str = 'deepseek-chat'
    ) -> None:
        self.client = OpenAI(
            api_key = os.environ.get(api_key_environ_name),
            base_url = url
        )
        self.default_model = default_model
    
    def __generate_question(self, history:list, model:str, lang:Literal['cn', 'en'] = 'cn') -> dict:
        prompt = "请根据以下对话生成一个有深度的、高质量的、相关的问题。（只要一个，直接写出）" if lang == 'cn' \
                 else "Please generate a relevant, meaningful question based on the following conversation. (Just one, write it directly)"
        
        messages = history + [{'role':'user', 'content':prompt}]
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False
        )
        return response.choices[0].message    # Return the generated question message
    
    def __generate_answer(self, history:list, model:str) -> dict:
        response = self.client.chat.completions.create(
            model=model,
            messages=history,
            stream=False
        )
        return response.choices[0].message    # Return the generated answer message
