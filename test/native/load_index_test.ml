(* [Direct.load_index]/[Symbolic.load_index]: [index.Tensor]'s runtime-gather
   primitive at the [SEMANTICS] level, exercised directly -- no [Graph_ir] op
   exists for it yet (Gate 3). *)

let s1c n = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:n
let zero = Dim.index 0
let v c = Vec6.make ~n:zero ~t:zero ~d:zero ~h:zero ~w:zero ~c

let i64_of values =
  Tensor.materialize_i64
    (s1c (List.length values))
    (fun c -> List.nth values (Dim.to_int (Vec6.get c Axis.C)))

let%expect_test
    "Direct.load_index: exact read and ATen-style negative-index normalization"
    =
  let idx = i64_of [ 2L; -1L; -8L ] in
  let extent = Dim.extent 8 in
  let show c =
    Fmt.pr "c=%d -> %d@." (Dim.to_int c)
      (Dim.to_int (Direct.load_index idx (v c) ~extent))
  in
  show (Dim.index 0);
  show (Dim.index 1);
  show (Dim.index 2);
  [%expect {|
    c=0 -> 2
    c=1 -> 7
    c=2 -> 0
    |}]

(* [Err.Exn.E], not a bare [Invalid_argument] and not a returned [Error] --
   round 8's resolution: [Direct]'s pixels already raise for other invariant
   violations, so a dtype mismatch or an out-of-range gather value follows the
   identical, already-established contract. *)
let catch f =
  try
    ignore (f ());
    "no exception"
  with Err.Exn.E e -> Format.asprintf "raised: %a" Err.Exn.pp_kind e

let%expect_test "Direct.load_index: raises Err.Exn.E on a non-I64 source" =
  let not_i64 = Tensor.materialize (s1c 1) (fun _ -> 1.0) in
  Fmt.pr "%s@."
    (catch (fun () -> Direct.load_index not_i64 (v zero) ~extent:(Dim.extent 8)));
  [%expect {| raised: Data source must be I64, got f32 |}]

let%expect_test
    "Direct.load_index: raises Err.Exn.E on an out-of-range gather value, \
     positive and negative" =
  let idx = i64_of [ 3L; -4L ] in
  let extent = Dim.extent 3 in
  Fmt.pr "%s@."
    (catch (fun () -> Direct.load_index idx (v (Dim.index 0)) ~extent));
  Fmt.pr "%s@."
    (catch (fun () -> Direct.load_index idx (v (Dim.index 1)) ~extent));
  [%expect
    {|
    raised: gather index 3 out of range [-3, 2]
    raised: gather index -4 out of range [-3, 2]
    |}]

(* [Symbolic.load_index] is construction only -- no check at build time, per
   round 3's design ("resolution happens once, where the value becomes
   concrete... never at Symbolic's construction step"). Grounding it through
   [Ground_eval]/[resolve_data_source] belongs to the Gate 2 ground_eval
   tests; this just confirms the node it builds. *)
let%expect_test "Symbolic.load_index: construction only, prints as a Data node"
    =
  let sg =
    Tensor_sig.create ~id:(Tensor_id.of_int 0) ~name:"" ~shape:(s1c 3)
      ~fmt:(Payload.Fmt Payload.I64) ()
  in
  let e = Symbolic.load_index sg Symbolic.out_vec ~extent:(Dim.extent 8) in
  Fmt.pr "%a@." (Expr.Pp.index ~names:(fun _ -> "r")) e;
  [%expect {| data(t0[N,T,D,H,W,C],8) |}]
