Payload-free MobileNetV1-class conversion through Const-SSA.

The source is a committed `model.json`, not a `.pt2` archive: this invocation
has no parameter bytes to preload. Canonicalization must still fold the weight
relayouts and inference batch norms symbolically, then Native4D must carry the
same Const-SSA exports. `quick` verification is allowed to be inconclusive on
large activation clusters, but it must report no refutation.

  $ ../bin/native_graph.exe visualize --model mobilenetv1_100_model.json --limits small --verify-symbolic quick --output session.json
  native_graph: the encoded document is over the ceiling
  [123]

The small profile refuses the encoded session without writing a partial file.
The default profile admits it and checks the intended conversion and verdicts.

  $ test ! -e session.json
  $ ../bin/native_graph.exe visualize --model mobilenetv1_100_model.json --verify-symbolic quick --output session.json
  $ ./cram_probe.exe mobilenetv1 session.json
  native4d: available
  verification: available
  g/native/001 nodes=115 batch_norm=0 add=0 sqrt=0
  g/native4d/000 nodes=115 batch_norm=0 add=0 sqrt=0
  refuted: False
