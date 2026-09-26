Export a Model Explorer session with --generated-js, over the committed
mobilenetv2_050 model.json.

UNGATED for the same reason me_visualize_json_cram.t is: model.json is a
submodule checkout, not a downloaded weight.

Off by default: no js/js_truncated/js_unavailable attribute anywhere, and the
capability reads not_requested.

  $ ../bin/native_graph.exe visualize --model model.json --output off.json
  $ python3 -c "
  > import json
  > s = json.load(open('off.json'))
  > count = sum(1 for g in s['graphCollections'][0]['graphs']
  >             for n in g['nodes'] for a in (n.get('attrs') or [])
  >             if a['key'] in ('js', 'js_truncated', 'js_unavailable'))
  > cap = [c for c in s['capabilities'] if c['key'] == 'feature:generated_js'][0]
  > print('js attrs', count, cap['status']['state'])"
  js attrs 0 not_requested

A bare --generated-js is the full optimization pipeline, attached to every
out<i> node of every operator detail graph -- expr/g/native/001/n0's out0
among them, the first canonical operator.

  $ ../bin/native_graph.exe visualize --model model.json --generated-js --output optimized.json
  $ python3 -c "
  > import json
  > s = json.load(open('optimized.json'))
  > cap = [c for c in s['capabilities'] if c['key'] == 'feature:generated_js'][0]
  > print(cap['status']['state'], cap['status']['payload']['kind'])
  > g = {g['id']: g for g in s['graphCollections'][0]['graphs']}['expr/g/native/001/n0']
  > n = {n['id']: n for n in g['nodes']}['out0']
  > a = {x['key']: x['value'] for x in n['attrs']}
  > print('js' in a, 'js_truncated' in a, 'js_unavailable' in a, len(a['js']))"
  available present
  True False False 193

=raw is the unoptimized program for the same node -- present, and a
different text from the optimized one above.

  $ ../bin/native_graph.exe visualize --model model.json --generated-js=raw --output raw.json
  $ python3 -c "
  > import json
  > def js_of(path):
  >     s = json.load(open(path))
  >     g = {g['id']: g for g in s['graphCollections'][0]['graphs']}['expr/g/native/001/n0']
  >     n = {n['id']: n for n in g['nodes']}['out0']
  >     return {x['key']: x['value'] for x in n['attrs']}['js']
  > opt = js_of('optimized.json')
  > raw = js_of('raw.json')
  > print('raw len', len(raw), 'differs from optimized', raw != opt)"
  raw len 485 differs from optimized True

A named subset (here, only unit_loops -- this node is a Permute, simple
enough that fold/simplify/guards/cse/hoist are each individually no-ops on
it, so unit_loops alone is what isolates a third, distinct text from the
other two) is neither the full pipeline's nor the empty one's.

  $ ../bin/native_graph.exe visualize --model model.json --generated-js=unit_loops --output custom.json
  $ python3 -c "
  > import json
  > def js_of(path):
  >     s = json.load(open(path))
  >     g = {g['id']: g for g in s['graphCollections'][0]['graphs']}['expr/g/native/001/n0']
  >     n = {n['id']: n for n in g['nodes']}['out0']
  >     return {x['key']: x['value'] for x in n['attrs']}['js']
  > opt = js_of('optimized.json')
  > raw = js_of('raw.json')
  > custom = js_of('custom.json')
  > print('custom differs from optimized', custom != opt, 'and from raw', custom != raw)"
  custom differs from optimized True and from raw True

An unknown pass name is rejected, not silently ignored.

  $ ../bin/native_graph.exe visualize --model model.json --generated-js=not_a_pass --output bad.json
  Usage: native_graph visualize [--help] [OPTION]…
  native_graph: option '--generated-js': unknown optimization pass "not_a_pass"
                (known: unit_loops, fold, simplify, guards, cse, hoist,
                collapse)
  [124]

Every other part of the document is untouched: diffing --generated-js's
session against the flag-off one, with every js/js_truncated/js_unavailable
attribute stripped out first (and the one capability row's payload, which
necessarily differs), is empty.

  $ python3 -c "
  > import json
  > def strip(doc):
  >     for g in doc['graphCollections'][0]['graphs']:
  >         for n in g['nodes']:
  >             n['attrs'] = [a for a in (n.get('attrs') or [])
  >                           if a['key'] not in ('js', 'js_truncated', 'js_unavailable')]
  >     doc['capabilities'] = [c for c in doc['capabilities']
  >                            if c['key'] != 'feature:generated_js']
  >     return doc
  > off = strip(json.load(open('off.json')))
  > opt = strip(json.load(open('optimized.json')))
  > print('identical once the js attributes and capability are stripped', off == opt)"
  identical once the js attributes and capability are stripped True
