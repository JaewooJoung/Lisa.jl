using Lisa

# Assuming the first command line argument is the text you want to pass to the model
text_to_process = ARGS[1]

#model = load_gguf_model("/from/your/link/llama-2-7b-chat.Q6_K.gguf") #use  "/from/your/link" download from hugginssite(https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)
model = load_gguf_model("/media/crux/3806cda1-26b3-4214-8a59-aebe26c8c73a/vogen/LISA/llama-2-7b-chat.Q6_K.gguf") #use  "/from/your/link" download from hugginssite(https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)

@time println(sample(model, text_to_process))