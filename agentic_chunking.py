from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from dotenv import load_dotenv

load_dotenv()

# ==============================
# Load FLAN-T5
# ==============================
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_answer(prompt, max_new_tokens=300):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ==============================
# Tesla text to chunk
# ==============================
tesla_text = """Tesla's Q3 Results
Tesla reported record revenue of $25.2B in Q3 2024.
The company exceeded analyst expectations by 15%.
Revenue growth was driven by strong vehicle deliveries.
Model Y Performance  
The Model Y became the best-selling vehicle globally, with 350,000 units sold.
Customer satisfaction ratings reached an all-time high of 96%.
Model Y now represents 60% of Tesla's total vehicle sales.
Production Challenges
Supply chain issues caused a 12% increase in production costs.
Tesla is working to diversify its supplier base.
New manufacturing techniques are being implemented to reduce costs."""

# ==============================
# Prompt
# ==============================
prompt = f"""Split this text into logical chunks at natural topic boundaries.
Put "<<<SPLIT>>>" between each chunk. Keep related sentences together.

Text:
{tesla_text}

Return the text with <<<SPLIT>>> markers:"""

# ==============================
# Generate
# ==============================
print("🤖 Asking AI to chunk the text...")
marked_text = generate_answer(prompt)

# ==============================
# Parse chunks
# ==============================
chunks = marked_text.split("<<<SPLIT>>>")

clean_chunks = []
for chunk in chunks:
    cleaned = chunk.strip()
    if cleaned:
        clean_chunks.append(cleaned)

# ==============================
# Fallback: split by known topic headings
# ==============================
if len(clean_chunks) <= 1:
    print("⚠️ Model didn't use markers, falling back to heading-based splitting...")
    import re
    # Split at lines that look like headings (short lines, no period at end)
    parts = re.split(r'\n(?=[A-Z][^\n]{0,40}\n)', tesla_text)
    clean_chunks = [p.strip() for p in parts if p.strip()]

# ==============================
# Show results
# ==============================
print("\n🎯 AGENTIC CHUNKING RESULTS:")
print("=" * 50)
for i, chunk in enumerate(clean_chunks, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()