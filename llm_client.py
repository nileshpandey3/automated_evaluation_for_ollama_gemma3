from ollama import ChatResponse, chat

class LLMClient:
    def __init__(self, model, raw=False) -> None:
        self.model = model
        self.raw = raw

    def prompt(self, input):
        response: ChatResponse = chat(model=self.model, messages=[
        {
            'role': 'user',
            'content': input,
        },
        ])
        if response is not None:
            if self.raw:
                return (response)
            return (response['message']['content'])
