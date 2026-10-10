Export a Model Explorer session with --generated-js, over the committed
mobilenetv2_050 model.json.

UNGATED for the same reason me_visualize_json_cram.t is: model.json is a
submodule checkout, not a downloaded weight.

Off by default: no js/js_truncated/js_unavailable attribute anywhere, and the
capability reads not_requested.

  $ ../bin/native_graph.exe visualize --model model.json --output off.json
  $ ./cram_probe.exe js-off off.json
  js attrs 0 not_requested

A bare --generated-js is the full optimization pipeline, attached to every
out<i> node of every operator detail graph -- expr/g/native/001/n0's out0
among them, the first canonical operator.

  $ ../bin/native_graph.exe visualize --model model.json --generated-js --output optimized.json
  $ ./cram_probe.exe js-present optimized.json
  available present
  True False False 193

=raw is the unoptimized program for the same node -- present, and a
different text from the optimized one above.

  $ ../bin/native_graph.exe visualize --model model.json --generated-js=raw --output raw.json
  $ ./cram_probe.exe js-raw optimized.json raw.json
  raw len 485 differs from optimized True

A named subset (here, only unit_loops -- this node is a Permute, simple
enough that fold/simplify/guards/cse/hoist are each individually no-ops on
it, so unit_loops alone is what isolates a third, distinct text from the
other two) is neither the full pipeline's nor the empty one's.

  $ ../bin/native_graph.exe visualize --model model.json --generated-js=unit_loops --output custom.json
  $ ./cram_probe.exe js-custom optimized.json raw.json custom.json
  custom differs from optimized True and from raw True

An unknown pass name is rejected, not silently ignored. NO_COLOR, for the reason
me_visualize_json_cram.t gives: cmdliner quotes the option name only when
styling is off, and cram strips ANSI, so the golden would otherwise depend on
the caller's $TERM.

  $ NO_COLOR=1 ../bin/native_graph.exe visualize --model model.json --generated-js=not_a_pass --output bad.json
  Usage: native_graph visualize [--help] [OPTION]…
  native_graph: option '--generated-js': unknown optimization pass "not_a_pass"
                (known: unit_loops, fold, simplify, guards, cse, hoist,
                collapse)
  [124]

Every other part of the document is untouched: diffing --generated-js's
session against the flag-off one, with every js/js_truncated/js_unavailable
attribute stripped out first (and the one capability row's payload, which
necessarily differs), is empty.

  $ ./cram_probe.exe js-equality off.json optimized.json
  identical once the js attributes and capability are stripped True
