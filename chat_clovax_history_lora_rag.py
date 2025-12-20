from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================
# Set route
# ==========================
#BASE_MODEL_PATH = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
BASE_MODEL_PATH = "./HyperCLOVAX-SEED-Text-Instruct-1.5B"
LORA_PATH       = "./HyperCLOVAX-history-lora-fp16"
DOCS_PATH       = "nh_raw.jsonl"   # 01_crawl_nh_raw.py 결과 파일


# ==========================
# Document loading & search preparation
# ==========================
def load_docs(path=DOCS_PATH):
    docs = []
    texts = []

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"The file {path} does not exist. Please crawl it first.")

    with p.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            text = d.get("text", "").strip()
            if not text:
                continue
            docs.append(d)
            texts.append(text)

    print(f"[RAG] Loaded {len(docs)} docs from {path}")

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2)
    )
    tfidf = vectorizer.fit_transform(texts)
    print("[RAG] Built TF-IDF matrix:", tfidf.shape)

    return docs, vectorizer, tfidf


def search_docs(query, docs, vectorizer, tfidf_matrix, k=3):
    """
    Returns the top k documents by TF-IDF cosine similarity for a user query.
    """
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix)[0]
    best_idx = scores.argsort()[::-1][:k]
    return [docs[i] for i in best_idx], [scores[i] for i in best_idx]


# ==========================
# Model loading (Base + LoRA)
# ==========================
def load_model_and_tokenizer():
    print("[Model] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("[Model] Loading base model (fp16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        dtype=torch.float16,
        device_map="auto",
    )

    print("[Model] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    if torch.cuda.is_available():
        model.to("cuda")
        print("[Model] Moved model to CUDA.")

    return model, tokenizer


# ==========================
# Main Chat Loop
# ==========================
def main():
    # 1) RAG preparation
    docs, vectorizer, tfidf_matrix = load_docs(DOCS_PATH)

    # 2) Model/Tokenizer Loading
    model, tokenizer = load_model_and_tokenizer()

    # 3) Reset conversation history
    chat_history = [
        {"role": "tool_list", "content": ""},
        {
            "role": "system",
            "content": (
                "- The name of the AI language model is \"CLOVA X\" and it was created by Naver.\n"
                "- You have been fine-tuned on Korean history texts from the National Institute of Korean History "
                "(NIKH) '우리역사넷' 신편 한국사.\n"
                "- When answering historical questions, rely primarily on the provided reference texts. "
                "If the answer is not in the references, say that you do not know rather than guessing."
            ),
        },
    ]

    print("Welcome to CLOVA X (History LoRA + RAG). Type 'exit' to exit.\n")

    while True:
        try:
            user_input = input("User: ")
        except EOFError:
            print("\nExit.")
            break

        if user_input.lower().strip() in ["exit", "quit"]:
            print("Exit.")
            break

        # --------------------------
        # 1) Search for related text in our history net documents
        # --------------------------
        retrieved_docs, scores = search_docs(
            user_input, docs, vectorizer, tfidf_matrix, k=3
        )

        context_chunks = []
        for d, s in zip(retrieved_docs, scores):
            # Use only the front part, not too long (adjust if necessary)
            snippet = d["text"][:5000]
            title = d.get("title", "")
            url = d.get("url", "")
            context_chunks.append(
                f"[Source Title] {title}\n[URL] {url}\n\n{snippet}"
            )

        context_text = "\n\n------------------------------\n\n".join(context_chunks)

        # --------------------------
        # 2) Configure the actual user messages to be sent to the model
        # --------------------------
        augmented_user_input = (
            "다음은 국사편찬위원회 우리역사넷(신편 한국사)에서 가져온 참고 자료이다.\n"
            "아래 참고 자료와 사용자 질문을 바탕으로, 우리역사넷의 서술과 모순되지 않도록 신중하게 한국어로 답하라.\n"
            "가능하면 개념을 차근차근 설명하고, 여러 문단으로 나누어 **상세하게** 서술하라.\n"
            "짧게 답하지 말고, 필요한 경우 예시와 배경 설명도 덧붙여라.\n"
            "참고 자료에 없는 내용은 추측하지 말고, 모른다고 대답해도 된다.\n\n"
            f"=== 참고 자료 시작 ===\n{context_text}\n=== 참고 자료 끝 ===\n\n"
            f"질문: {user_input}"
        )

        # Add to history
        chat_history.append({"role": "user", "content": augmented_user_input})

        # --------------------------
        # 3) Tokenize & Generate
        # --------------------------
        inputs = tokenizer.apply_chat_template(
            chat_history,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=2048,   # Number of new tokens to be created
                do_sample=True,
                top_p=0.4,
                temperature=0.8,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Cut out the prompt part from the entire output and take only the newly generated part.
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_len:]
        ai_response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        print(f"CLOVA X: {ai_response}\n")

        # Add assistant responses to conversation history
        chat_history.append({"role": "assistant", "content": ai_response})


if __name__ == "__main__":
    main()
