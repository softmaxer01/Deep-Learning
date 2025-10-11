from bpe import BPE

with open("input.txt","r") as r:
  text = r.read()


bpt = BPE(text,270)

val_text = "hello i am smruti, i am an undergrad math student at nit-rkl..."

print(f"tokens: {bpt.encode(val_text)}")

# print(f"text: {bpt.decode(bpt.encode(val_text))}")
# print(bpt.vocab)

