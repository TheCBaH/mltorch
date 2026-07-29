Version-indexed ids: what the type checker must reject.

Each negative case is paired with a control that must still compile. Without the
controls a broken harness — a wrong include path, say — would reject everything
and look like a pass. See .ai/native_transform_versioning.md.

  $ R=../../..
  $ check() { sh $R/test/native/version_safety.sh $R/test/native/version_probe.exe "$1" "$2"; }

A relation runs one way. Asking it to carry a destination id forward, or a
source id backward, is the collision fixed by hand at map_verify.ml:316.

  $ check "forward given a dst edge" "ignore (forward m d)"
  forward given a dst edge: rejected
  $ check "forward given a src edge" "ignore (forward m s)"
  forward given a src edge: COMPILES
  $ check "backward given a src edge" "ignore (backward m s)"
  backward given a src edge: rejected
  $ check "backward given a dst edge" "ignore (backward m d)"
  backward given a dst edge: COMPILES

An id belongs to the snapshot it was drawn from, so it cannot be resolved in the
other graph's universe even though both hold the same raw numbers.

  $ check "src edge in the dst universe" "ignore (Universe.find (Snapshot.edges sb) (raw s) = Some s)"
  src edge in the dst universe: rejected

Brand.fresh hands back a packed brand, so a caller can mint their own version
but never an existing snapshot's. A universe built under a self-minted brand
therefore yields ids that unify with nothing the snapshots produced.

  $ check "universe forged at a foreign tag" "let Brand.Pack mine = Brand.fresh () in ignore (forward m (match Universe.find (Universe.create mine (raws (Set.singleton s))) (raw s) with Some x -> x | None -> s))"
  universe forged at a foreign tag: rejected

A recipe's edit metadata is typed too. A value claim runs source-to-target and
only substitution is symmetric, so stating a claim the other way round — a fresh
edge where the source belongs — does not compile.

  $ check "claim stated source-to-target" "ignore (Recipe.run Recipe.(let* x = existing (Tensor_id.of_int 0) in let* f = fresh (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1) in replace ~remove:[] ~insert:[] ~claims:[ (x, Fresh f, Correspondence.Identical) ] ()) sa (Id_supply.of_graph g))"
  claim stated source-to-target: COMPILES
  $ check "claim stated backwards" "ignore (Recipe.run Recipe.(let* x = existing (Tensor_id.of_int 0) in let* f = fresh (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1) in replace ~remove:[] ~insert:[] ~claims:[ (f, Preserved x, Correspondence.Identical) ] ()) sa (Id_supply.of_graph g))"
  claim stated backwards: rejected

Erasing a tag stays available: printers, the PT2 lens and Ground_eval all need
raw ids. It is the other direction that has no entry point.

  $ check "raw projection off either side" "ignore (raw s : Tensor_id.t); ignore (raw d : Tensor_id.t)"
  raw projection off either side: COMPILES
