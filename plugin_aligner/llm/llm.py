import torch
from openai import OpenAI
from fastchat.model import load_model, get_conversation_template
import logging
import time
import concurrent.futures
from vllm import LLM as vllm
from vllm import SamplingParams
from LLM_MMR.utils.constants import *
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import google.generativeai as genai

class LLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def generate(self, prompt):
        raise NotImplementedError("LLM must implement generate method.")

    def predict(self, sequences):
        raise NotImplementedError("LLM must implement predict method.")


class OpenAILLM(LLM):
    def __init__(self,
                 model_path,
                 api_key=None,
                 system_message=None
                ):
        super().__init__()

        if not api_key.startswith('sk-'):
            raise ValueError('OpenAI API key should start with sk-')
        self.client = OpenAI(api_key = api_key)
        self.model_path = model_path
        self.system_message = system_message if system_message is not None else "You are a helpful assistant."

    def generate(self, prompt, temperature=0, max_tokens=512, n=1, max_trials=10, failure_sleep_time=5):
        for _ in range(max_trials):
            try:
                results = self.client.chat.completions.create(
                    model=self.model_path,
                    messages=[
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                )
                return [results.choices[i].message.content for i in range(n)]
            except Exception as e:
                logging.warning(
                    f"OpenAI API call failed due to {e}. Retrying {_+1} / {max_trials} times...")
                time.sleep(failure_sleep_time)

        return [" " for _ in range(n)]

    def generate_batch(self, prompts, temperature=0, max_tokens=512, n=1, max_trials=10, failure_sleep_time=5):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate, prompt, temperature, max_tokens, n,
                                       max_trials, failure_sleep_time): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results
            
    def predict(self, sequences, question):
        black_list = get_black_list()
        # check if any black list in the sequences
        for black in black_list:
            if black in sequences[0]:
                return [0]
        # verify whether the generation is related with the target
        split = question.split()
        cnt = 0
        for word in split:
            if word in sequences[0]:
                cnt += 1
        # if cnt/len(split) < 0.5:
        #     return [0]
        prediction = self.generate(sequences[0])
        for i in range(len(prediction)):
            try:
                prediction[i] = int(prediction[i])
            except:
                prediction[i] = 0
        # print the sequences if the prediction is harmful
        if prediction[0] == 1:
            print(sequences[0])
        return prediction

