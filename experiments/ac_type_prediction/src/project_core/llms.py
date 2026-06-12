"""
Groq API access to Llama3 model
"""

from groq import Groq


class GroqClassifier:
    """
    Class for Llama3 classifiers, accessed through Groq API
    """

    def __init__(self, api_key, model_name):
        """
        Args:
            api_key: str, API key for Groq API
            model_name: str, name of a chosen LLM model hosted on Groq, e.g. "llama-3.1-8b-instant"
        """
        self.client = Groq(api_key=api_key)
        print('Groq client set up')
        self.model_name = model_name

    def get_llm_completion_zero(self, user_prompt, system_prompt="", temperature=1, max_out_tokens=512):
        """
        Call to Groq API to get model completion in zero-shot prompt setting
        Args:
            user_prompt: str User prompt: task instruction including TARGET essay
            system_prompt: str (optional) System prompt: description of LLM-role and context
        Return:
            out_text: str LLM output text
        """
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=temperature,
            max_completion_tokens=max_out_tokens,
            top_p=1,
            stream=False,
            stop=None,
        )
        out_text = completion.choices[0].message.content
        return out_text

    def get_llm_completion_few_shot(self, user_prompt_target, user_prompt_demos, assistant_out_demos,
                                    system_prompt="", temperature=1, max_out_tokens=512):
        """
         Call to Groq API to get model completion in few-shot prompt setting
        Args:
            user_prompt_target: str User prompt: task instruction including TARGET essay
            user_prompt_demos: list(str) User prompts with each DEMO essay, index-aligned with assistant_out_demos
            assistant_out_demos: list(str) demo outputs, index-aligned with user_prompt_demos
            system_prompt: str (optional) System prompt: description of LLM-role and context
        Return:
            out_text: str LLM output text
        """
        message_system_prompt = [{"role": "system", "content": system_prompt}]
        message_user_prompt_target = [{"role": "user", "content": user_prompt_target}]
        assert len(user_prompt_demos) == len(assistant_out_demos), print("Unequal no. of demo ins and outs")
        message_demos = []
        for user_prompt_demo, demo_out in zip(user_prompt_demos, assistant_out_demos):
            prompt = {"role": "user", "content": user_prompt_demo}
            message_demos.append(prompt)
            out = {"role": "assistant", "content": demo_out}
            message_demos.append(out)
        messages = message_system_prompt + message_demos + message_user_prompt_target
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_out_tokens,
            top_p=1,
            stream=False,
            stop=None,
        )
        out_text = completion.choices[0].message.content
        return out_text

