from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

@app.on_event("startup")
def load_models():
    """Load all models during server startup"""
    global tokenizer_1, model_1, tokenizer_2, model_2, ai_detector
    
    try:
        print("⚡ Loading AI models...")
        
        # DistilGPT2 model
        tokenizer_1 = AutoTokenizer.from_pretrained("distilgpt2")
        model_1 = AutoModelForCausalLM.from_pretrained("distilgpt2")
        
        # GPT-2 Medium model
        tokenizer_2 = AutoTokenizer.from_pretrained("gpt2-medium")
        model_2 = AutoModelForCausalLM.from_pretrained("gpt2-medium")
        
        # AI Detection pipeline
        ai_detector = pipeline('text-classification', 
                              model="roberta-base-openai-detector")
        
        print("✅ Models loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")

def calculate_perplexity(model, tokenizer, text):
    """Calculate perplexity score for given text"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return torch.exp(outputs.loss).item()

@app.post("/detect")
async def detect_text(request: TextRequest):
    try:
        text = request.text
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Empty text input")
        
        # Calculate perplexities
        ppl1 = calculate_perplexity(model_1, tokenizer_1, text)
        ppl2 = calculate_perplexity(model_2, tokenizer_2, text)
        ppl_ratio = ppl1 / ppl2
        
        # Get AI detection score
        detector_result = ai_detector(text)[0]
        detector_label = detector_result['label']
        detector_confidence = detector_result['score']
        
        # Calculate AI probability score (0-100)
        ppl_score = min(100, max(0, (ppl_ratio - 1) * 50))  # Scale perplexity ratio
        detector_score = detector_confidence * 100 if detector_label == "Fake" else (1 - detector_confidence) * 100
        
        final_score = (ppl_score * 0.3 + detector_score * 0.7)  # Weighted average
        
        # Classification logic
        if final_score > 70:
            classification = "Likely AI-Generated"
        elif final_score > 30:
            classification = "Uncertain"
        else:
            classification = "Likely Human-Written"

        explanation = f"Final AI probability: {final_score:.2f}%. " \
                      f"This is based on perplexity analysis ({ppl_score:.2f}%) " \
                      f"and AI detection model ({detector_score:.2f}%). " \
                      f"The AI detector labeled it as '{detector_label}' with {detector_confidence:.2%} confidence."

        return {
            "classification": classification,
            "ai_probability": round(final_score, 2),
            "perplexity_ratio": round(ppl_ratio, 2),
            "detector_label": detector_label,
            "detector_confidence": round(detector_confidence, 4),
            "explanation": explanation
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/")
def health_check():
    return {"status": "active", "model": "ai-text-detector"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")
