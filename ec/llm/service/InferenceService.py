import os
from openai import OpenAI

from ec.llm.utils.const import TEMPERATURE, MAX_TOKENS, CLEAN_TEXT


class InferenceService:
    def __init__(self):
        self.__model = os.getenv('OPENAI_MODEL', 'text-davinci-003')
        self.__openai_client = OpenAI()
        self.__prompt_template = 'Comvierte el numero de 32 a binario'

    def __inference(self, prompt):
        return CLEAN_TEXT(self.__openai_client.completions.create(
            model=self.__model,
            prompt=prompt,
            max_tokens=100,
            temperature=0.1
        ).choices[0].text)

    def invoke(self, year: str) -> str:
        prompt = self.__prompt_template.format(year=year)
        return self.__inference(prompt)