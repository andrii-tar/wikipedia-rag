import os

from litellm import completion

MODEL = "ollama/llama3.2"
API_BASE_URL = "http://192.168.2.216:11434"

MODEL = "groq/llama3-8b-8192"
os.environ['GROQ_API_KEY'] = "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"

def invoke_model(query, context="", temperature=0.7) -> str:
    partial_message = ""

    messages = [
        {
            "role": "user",
            "content": query
        },
        {
            "role": "system",
            "content": context
        }
    ]
    print(messages)

    for chunk in completion(
            model=MODEL,
            #api_base=API_BASE_URL,
            messages=messages,
            temperature=temperature,
            top_p=.9,
            stream=True):

        if chunk['choices'][0]['delta']['content']:
            partial_message += chunk['choices'][0]['delta']['content']

    return partial_message


if __name__ == "__main__":
    invoke_model("who are you?", "Dima is a student")
    """
    response = completion(
        model="ollama/llama3.2",
        messages=[{"content": "Hello, how are you?", "role": "user"}],
        api_base="http://192.168.2.216:11434",
        stream=False,
    )
    # print(response)

    partial_message = ""
    for chunk in completion(
            model="ollama/llama3.2",
            api_base="http://192.168.2.216:11434",
            messages=[{"content": "Hello, how are you?", "role": "user"}],
            max_new_tokens=512,
            temperature=0.7,
            top_k=100,
            top_p=.9,
            repetition_penalty=1.18,
            stream=True):
        print(chunk)

        if chunk['choices'][0]['delta']['content']:
            partial_message += chunk['choices'][0]['delta']['content'] # extract text from streamed litellm chunks
    print ("\n\n",partial_message)
    
    """
