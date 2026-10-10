Gate 8's frozen Native4D corpus must choose the same canonical graph whether
it begins at the committed payload-free model.json or at the payload-backed
release archive.  Selecting both canonical Native and Native4D projections
keeps source metadata and the archive-only payload cache out of the comparison.

  $ ../bin/native_graph.exe visualize --model mobilenetv2_050_model.json --output source.json
  $ ../bin/native_graph.exe visualize --model "$PT2_DATA/mobilenetv2_050/mobilenetv2_050.pt2" --output archive.json
  $ ../bin/native_graph.exe visualize --model "$PT2_DATA/mobilenetv2_050/mobilenetv2_050.pt2" --fold --output preloaded.json
  $ ./cram_probe.exe graph-equality source.json archive.json preloaded.json
  g/native/001 source=archive:True archive=preloaded:True
  g/native4d/000 source=archive:True archive=preloaded:True
