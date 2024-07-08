import os
from groq import Groq


class GroqManager:
    def __init__(self):
        self.qroq_client = Groq(api_key=os.environ.get("GROQ_KEY"))

    def generate_response(self, context: str, question: str):
        return self.qroq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a great assistant that helps with sharing knowledge about medical plants. "
                               "Given the following sections in brackets from an encyclopedia, answer the question using "
                               "only that information, outputted in .md format. If you are unsure and answer is not "
                               "explicitly written in given sections, say 'Sorry, I don't know how to help with that question.'"
                               "Please answer by providing the whole content of sections that are suitable. "
                               "Context sections: "
                               f"{context}"
                },
                {
                    "role": "user",
                    "content": question,
                }
            ],
            model="llama3-8b-8192",
        )
