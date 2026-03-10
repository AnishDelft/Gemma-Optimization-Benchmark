from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="none",  # vLLM doesn't require a real key
)

stream = client.chat.completions.create(
    model="gemma-3-4b-quant",
    messages=[
        {"role": "user", "content": "Explain quantization in simple terms"}
    ],
    max_tokens=256,
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()
