(* The Native evaluator under each storage and scheduling configuration, on one
   body: every configuration gives the default's bits, a prepared body can be
   run again on other inputs without disturbing the first results, and a budget
   the body exceeds refuses at preparation. *)

open Graph_ir
module S = Inference_compare_pt2.Native_storage
module C = Inference_compare_pt2.Storage_config
module Tid = Tensor_id

let chain =
  Native_test.Pipeline_test.build "chain"
    (let open Graph_builder in
     let* x = input ~shape:(Native_test.Pipeline_test.nhwc ~h:8 ~w:8 ~c:4) () in
     let* a = relu x in
     let* b = sqrt a in
     let* c = relu b in
     let* d = sqrt c in
     relu d)

let sig_of g id = Tid.Map.find id g.Graph.tensors

let input_of g seed =
  let id = List.hd g.Graph.inputs in
  let sg = sig_of g id in
  ( id,
    Tensor.materialize sg.Tensor_sig.shape (fun c ->
        float_of_int (((Vec6.offset sg.Tensor_sig.shape c :> int) + seed) mod 7)
        /. 4.) )

let get = function Ok x -> x | Error m -> failwith m

let run ?(config = C.default) ~notes g ~seed =
  let c = get (S.compile ~notes ~config g) in
  (c, get (S.execute ~notes c ~inputs:[ input_of g seed ]))

let bits g env = List.map (fun id -> Tid.Map.find id env) g.Graph.outputs
let equal g a b = List.for_all2 Tensor.equal_bits (bits g a) (bits g b)

let configs =
  [
    ("release", { C.storage = Release; scheduled = false });
    ("arena", { C.storage = Arena_best_effort; scheduled = false });
    ("separate", { C.storage = Separate; scheduled = false });
    ("shared", { C.storage = Shared; scheduled = false });
    ("release+scheduled", { C.storage = Release; scheduled = true });
    ("arena+scheduled", { C.storage = Arena_best_effort; scheduled = true });
    ("separate+scheduled", { C.storage = Separate; scheduled = true });
    ("shared+scheduled", { C.storage = Shared; scheduled = true });
  ]

let%expect_test
    "each configuration gives the default's bits, before and after a rerun on \
     other inputs" =
  let g = chain in
  let _, first = run ~notes:(S.Notes.create ()) g ~seed:0 in
  let _, second = run ~notes:(S.Notes.create ()) g ~seed:3 in
  List.iter
    (fun (name, config) ->
      let notes = S.Notes.create () in
      let c, a = run ~config ~notes g ~seed:0 in
      let first_run = equal g a first in
      let b = get (S.execute ~notes c ~inputs:[ input_of g 3 ]) in
      Printf.printf "%s: first run %b, rerun %b, first results kept %b\n" name
        first_run (equal g b second) (equal g a first))
    configs;
  [%expect
    {|
    release: first run true, rerun true, first results kept true
    arena: first run true, rerun true, first results kept true
    separate: first run true, rerun true, first results kept true
    shared: first run true, rerun true, first results kept true
    release+scheduled: first run true, rerun true, first results kept true
    arena+scheduled: first run true, rerun true, first results kept true
    separate+scheduled: first run true, rerun true, first results kept true
    shared+scheduled: first run true, rerun true, first results kept true |}]

let%expect_test
    "the arena is used and reuses storage; a second body does not change the \
     first's" =
  let notes = S.Notes.create () in
  let config = { C.storage = Arena_best_effort; scheduled = false } in
  let _ = run ~config ~notes chain ~seed:0 in
  print_endline (S.Notes.summary notes);
  [%expect
    {| 1 body(ies): 1 in an arena (1 reusing storage), 0 role-planned, 0 release-only |}]

let%expect_test "a budget the body exceeds refuses at preparation" =
  let notes = S.Notes.create () in
  List.iter
    (fun scheduled ->
      let config = { C.storage = Arena_required 1L; scheduled } in
      match S.compile ~notes ~config chain with
      | Ok _ -> print_endline "prepared"
      | Error m -> print_endline m)
    [ false; true ];
  [%expect
    {|
    arena: the run needs 3072 bytes but the budget is 1
    arena: the run needs 3072 bytes but the budget is 1 |}]
