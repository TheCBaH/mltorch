The catalogue is verified before creating the flattened view. No Python, torch,
network, or producer checkout is needed for these tests.

  $ tool="$PWD/../bin/transformers_source.exe"
  $ mkdir -p 'source space/item/models' 'source space/item/data/constants' 'source space/item/data/weights' flat
  $ files='captures.json cases.json contract.json data/constants/model_constants_config.json data/weights/model_weights_config.json models/model.json models/op_facts.json'
  $ for file in $files; do printf '{}\n' > "source space/item/$file"; done
  $ digest=$(sha256sum 'source space/item/models/model.json' | cut -d ' ' -f 1)
  $ {
  > printf '{"schema_version":1,"artifacts":[{"artifact_id":"a/b","path":"item","graph_sha256":"%s","files":{' "$digest"
  > sep=''
  > for file in $files; do printf '%s"%s":{"size":3,"sha256":"%s"}' "$sep" "$file" "$digest"; sep=,; done
  > printf '}}]}\n'
  > } > 'source space/catalogue.json'
  $ cp 'source space/catalogue.json' good.json
  $ "$tool" inventory "$PWD/source space" flat inventory.json
  $ test -L flat/a--b && echo 'verified seven files and created view'
  verified seven files and created view
  $ printf '{"model":"a--b","native_builds":true,"native4d_converts":false,"kernel_converts":true}\n' > admission.jsonl
  $ "$tool" summary inventory.json admission.jsonl run.json source-pin consumer-pin 0
  $ cat run.json
  {
    "catalogue_sha256": "0fc6659f7ec1df7ef81dc54635f58810591023bcd970c0cabe861e89cb3d7d87",
    "verified_files": 7,
    "consumer_commit": "consumer-pin",
    "consumer_workspace_changes": "0",
    "producer_commit": "source-pin",
    "source_gitlink": "source-pin",
    "rows": 1,
    "native_builds": 1,
    "native4d_converts": 0,
    "kernel_converts": 1
  }
  $ printf '' > admission.jsonl
  $ "$tool" summary inventory.json admission.jsonl run.json source-pin consumer-pin 0
  transformers_source: admission coverage differs from inventory
  [2]

File corruption of the same size and changed extents are independently refused.

  $ mkdir rejected
  $ printf '[]\n' > 'source space/item/models/model.json'
  $ "$tool" inventory "$PWD/source space" rejected refused.json 2>&1 | sed "s|$PWD/||g"
  transformers_source: sha256 differs: source space/item/models/model.json
  $ test ! -e rejected/a--b && test ! -e refused.json
  $ printf '{}\n\n' > 'source space/item/models/model.json'
  $ "$tool" inventory "$PWD/source space" rejected refused.json 2>&1 | sed "s|$PWD/||g"
  transformers_source: size differs: source space/item/models/model.json
  $ printf '{}\n' > 'source space/item/models/model.json'

Path escapes, duplicate IDs, flatten collisions and repeated JSON members fail.

  $ sed 's|"path":"item"|"path":"../item"|' good.json > 'source space/catalogue.json'
  $ "$tool" inventory "$PWD/source space" rejected refused.json
  transformers_source: unsafe relative path: ../item
  [2]
  $ sed 's|"path":"item"|"path":"/item"|' good.json > 'source space/catalogue.json'
  $ "$tool" inventory "$PWD/source space" rejected refused.json
  transformers_source: unsafe relative path: /item
  [2]
  $ row=$(sed 's/^.*"artifacts":\[//; s/\]}$//' good.json)
  $ printf '{"schema_version":1,"artifacts":[%s,%s]}\n' "$row" "$row" > 'source space/catalogue.json'
  $ "$tool" inventory "$PWD/source space" rejected refused.json
  transformers_source: duplicate artifact ID: a/b
  [2]
  $ cp -R 'source space/item' 'source space/other'
  $ row2=$(printf '%s' "$row" | sed 's|"a/b"|"a--b"|; s|"path":"item"|"path":"other"|')
  $ printf '{"schema_version":1,"artifacts":[%s,%s]}\n' "$row" "$row2" > 'source space/catalogue.json'
  $ "$tool" inventory "$PWD/source space" rejected refused.json
  transformers_source: duplicate flattened artifact ID: a--b
  [2]
  $ sed 's/"schema_version":1/"schema_version":1,"schema_version":1/' good.json > 'source space/catalogue.json'
  $ "$tool" inventory "$PWD/source space" rejected refused.json
  transformers_source: duplicate JSON member: schema_version
  [2]
  $ sed 's/"schema_version":1/"schema_version":2/' good.json > 'source space/catalogue.json'
  $ "$tool" inventory "$PWD/source space" rejected refused.json
  transformers_source: unsupported catalogue schema
  [2]
  $ cp good.json 'source space/catalogue.json'
  $ mv 'source space/item/models/model.json' saved-model.json
  $ ln -s "$PWD/saved-model.json" 'source space/item/models/model.json'
  $ "$tool" inventory "$PWD/source space" rejected refused.json 2>&1 | sed "s|$PWD/||g"
  transformers_source: non-regular path: source space/item/models/model.json
  $ rm 'source space/item/models/model.json'
  $ mv saved-model.json 'source space/item/models/model.json'

Checkout checks derive the pin from the consumer index and reject before Dune.
The miniature consumer also proves that an empty directory inside a parent Git
worktree cannot masquerade as an initialized producer.

  $ mkdir -p consumer/scripts consumer/modules/devcontainer.transformers
  $ cp ../scripts/transformers-admission.sh consumer/scripts/
  $ git -C consumer init -q
  $ git -C 'source space' init -q
  $ git -C 'source space' add .
  $ git -C 'source space' -c user.name=Test -c user.email=test@example.test commit -qm initial
  $ pin=$(git -C 'source space' rev-parse HEAD)
  $ git -C consumer update-index --add --cacheinfo "160000,$pin,modules/devcontainer.transformers"
  $ sh consumer/scripts/transformers-admission.sh missing out
  transformers admission: missing checkout; run git submodule update --init modules/devcontainer.transformers
  [2]
  $ sh consumer/scripts/transformers-admission.sh consumer/modules/devcontainer.transformers out
  transformers admission: uninitialized checkout; run git submodule update --init modules/devcontainer.transformers
  [2]
  $ printf 'dirty' >> 'source space/catalogue.json'
  $ sh consumer/scripts/transformers-admission.sh 'source space' out
  transformers admission: modified producer inputs:  M catalogue.json
  [2]
  $ cp good.json 'source space/catalogue.json'
  $ printf 'dirty' >> 'source space/item/models/model.json'
  $ sh consumer/scripts/transformers-admission.sh 'source space' out
  transformers admission: modified producer inputs:  M item/models/model.json
  [2]
  $ git -C 'source space' -c user.name=Test -c user.email=test@example.test commit -qam changed
  $ sh consumer/scripts/transformers-admission.sh 'source space' out 2>&1 | sed -E 's/[0-9a-f]{40}/PIN/g'
  transformers admission: producer is at PIN, expected gitlink PIN; run git submodule update --init modules/devcontainer.transformers
