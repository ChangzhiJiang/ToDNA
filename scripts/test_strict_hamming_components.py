import random, subprocess, tempfile
from pathlib import Path
from prepare_strict_independent_v1 import canon
def test(binary):
 rng=random.Random(918);seq=[''.join(rng.choices('ACGT',k=200)) for _ in range(60)]
 seq += [seq[0],seq[1].translate(str.maketrans('ACGT','TGCA'))[::-1]]
 for edits in [1,19,20,21,40]:
  s=list(seq[2]);
  for i in rng.sample(range(200),edits):s[i]=next(c for c in 'ACGT' if c!=s[i])
  seq.append(''.join(s))
 parent=list(range(len(seq)))
 def f(i):
  while parent[i]!=i:i=parent[i]
  return i
 for i in range(len(seq)):
  for j in range(i):
   rc=seq[j].translate(str.maketrans('ACGT','TGCA'))[::-1]
   if min(sum(a!=b for a,b in zip(seq[i],seq[j])),sum(a!=b for a,b in zip(seq[i],rc)))<=20:parent[f(i)]=f(j)
 with tempfile.TemporaryDirectory() as d:
  inp=Path(d)/'in';out=Path(d)/'out';inp.write_text('\n'.join(seq)+'\n');subprocess.run([str(binary),str(inp),str(out)],check=True)
  labels=list(map(int,out.read_text().split()))
  assert all((labels[i]==labels[j])==(f(i)==f(j)) for i in range(len(seq)) for j in range(len(seq)))
 print('PASS: exhaustive brute-force component comparison, duplicates, RC, threshold and transitivity')
if __name__=='__main__':
 import sys
 test(Path(sys.argv[1]))
