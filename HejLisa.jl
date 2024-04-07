using Lisa

# Assuming the first command line argument is the text you want to pass to the model
text_to_process = ARGS[1]

# "Lisa.brain" 파일에서 텍스트를 읽습니다.
text = read("Lisa.brain", String)

# 문자 기반 토크나이저를 생성합니다.
tokenizer = CharTokenizer(text)

# 텍스트를 토큰화합니다.
tokens = encode(text, tokenizer)

# 모델 구성을 설정합니다. config = ModelConfig(dim=64, hidden_dim=96, n_layers=4, n_heads=4, n_kv_heads=4, vocab_size=length(tokenizer.id_to_token), seq_len=128)
config = ModelConfig(dim=128, hidden_dim=96, n_layers=4, n_heads=8, n_kv_heads=8, vocab_size=length(tokenizer.id_to_token), seq_len=256)

# killed my computer. 
#config = ModelConfig(dim=4096, hidden_dim=11008, n_layers=32, n_heads=32, n_kv_heads=16, vocab_size=length(tokenizer.id_to_token), seq_len=256)
#=   dim         = 4096,
  hidden_dim  = 11008,
  n_layers    = 32,
  n_heads     = 32,
  n_kv_heads  = 32,
  vocab_size  = 32000,
  seq_len     = 512,
=#

# 모델을 훈련합니다.
weights = train(config, tokens; n_tokens=4_000_000, batch_size=4)

# 훈련된 가중치를 사용하여 언어 모델을 생성합니다.
model = LanguageModel(config, tokenizer, weights)

# 모델을 사용하여 텍스트 샘플을 생성합니다.
generated_text = sample(model, text_to_process; stop_on_special_token=false, bos_token=false)

# 생성된 텍스트를 출력합니다.
println(generated_text)
