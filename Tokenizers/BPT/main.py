from bpt import BPT

with open("/home/smruti/Desktop/git repos/Deep-Learning/Tokenizers/BPT/input.txt","r") as r:
  text = r.read()


bpt = BPT(text,270)

val_text = "hello i am smruti, i am an undergrad math student at nit-rkl..."

print(f"tokens: {bpt.encode(val_text)}")

print(f"text: {bpt.decode(bpt.encode(val_text))}")

