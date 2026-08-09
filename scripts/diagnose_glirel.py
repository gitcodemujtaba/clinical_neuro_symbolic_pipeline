"""
Standalone diagnostic. Isolates GLiREL from the rest of the pipeline to see
its actual raw output (no 0.5 threshold, no post-hoc constraint filter) on
a simple sentence with two obviously-related clinical entities. If this
still comes back empty at threshold=0.0, the problem is upstream of scoring
(labels/ner format). If it comes back with real (label, score) pairs but
all scores are low, it's a threshold/calibration issue -- the model just
isn't confident on zero-shot clinical relation labels at 0.5.
"""
import spacy
from glirel import GLiREL

print("Loading GLiREL...")
model = GLiREL.from_pretrained("jackboyla/glirel-large-v0")

try:
    nlp = spacy.load("en_core_sci_sm")
except Exception:
    import en_core_sci_sm
    nlp = en_core_sci_sm.load()

text = "Patient was given aspirin for chest pain."
doc = nlp(text)
tokens = [t.text for t in doc]
print("Tokens:", list(enumerate(tokens)))

aspirin_span = doc.char_span(text.index("aspirin"), text.index("aspirin") + len("aspirin"), alignment_mode="expand")
chest_pain_span = doc.char_span(text.index("chest pain"), text.index("chest pain") + len("chest pain"), alignment_mode="expand")
print("aspirin span (token idx):", aspirin_span.start, aspirin_span.end)
print("chest pain span (token idx):", chest_pain_span.start, chest_pain_span.end)

ner = [
    [aspirin_span.start, aspirin_span.end - 1, "Medication", "aspirin"],
    [chest_pain_span.start, chest_pain_span.end - 1, "Symptom", "chest pain"],
]
labels = ["treated with", "indicates", "causes", "located in", "measured by"]

print("\n--- Calling predict_relations with threshold=0.0 ---")
relations = model.predict_relations(tokens, labels, threshold=0.0, ner=ner, top_k=5)
print(f"\n{len(relations)} relation(s) returned (all scores, since threshold=0.0):")
for r in sorted(relations, key=lambda x: x["score"], reverse=True):
    print(f"  {r['head_text']} --[{r['label']}]--> {r['tail_text']}  score={r['score']:.4f}")

print("\n--- For comparison: GLiREL's own README example, unmodified ---")
text2 = 'Derren Nesbitt had a history of being cast in "Doctor Who", having played villainous warlord Tegana in the 1964 First Doctor serial "Marco Polo".'
import spacy as spacy2
nlp2 = spacy2.load("en_core_web_sm") if spacy2.util.is_package("en_core_web_sm") else nlp
doc2 = nlp2(text2)
tokens2 = [t.text for t in doc2]
labels2 = ['country of origin', 'licensed to broadcast to', 'father', 'followed by', 'characters']
ner2 = [[26, 27, 'PERSON', 'Marco Polo'], [22, 23, 'Q2989412', 'First Doctor']]
relations2 = model.predict_relations(tokens2, labels2, threshold=0.0, ner=ner2, top_k=1)
print(f"{len(relations2)} relation(s) on the README's own example (sanity check the model/library work at all):")
for r in sorted(relations2, key=lambda x: x["score"], reverse=True):
    print(f"  {r['head_text']} --[{r['label']}]--> {r['tail_text']}  score={r['score']:.4f}")
