(* [Index.Data]: the runtime-gather primitive `index.Tensor` needs -- reading
   a source's own stored value at a coordinate and using it as an index
   component. Exercised at the [Expr] level, independent of any [native]
   graph node (which doesn't exist until Gate 3): a hand-built [resolve_data]
   closure stands in for [Env.load_index]/[Ground_eval.resolve_data_source]. *)

open Expr

let s0 = Source.create 0
let s1 = Source.create 1
let origin = Coord.of_fn (fun _ -> 0)
let no_reducers _ = None

(* A tiny in-memory "tensor": maps a source and a coordinate's C component to
   a raw int64, the same shape [Env.load_index]/[resolve_data_source] have. *)
let resolve_data table src (c : int Coord.t) : (int64, [> `Missing ]) Err.t =
  match List.assoc_opt (Source.to_int src, c.Coord.c) table with
  | Some v -> Err.return v
  | None -> Err.fail `Missing

let pp_res fmt r =
  Core.Pretty.err_result ~ok:Fmt.int
    ~error:(fun fmt -> function
      | #Eval.index_error as e -> Eval.pp_index_error fmt e
      | `Missing -> Fmt.string fmt "missing")
    fmt r

let ev table e =
  Err.Escape.with_escape @@ fun esc ->
  Eval.eval_index esc
    ~widen:(fun (e : Eval.index_error) ->
      (e :> [ Eval.index_error | `Missing ]))
    ~output:origin ~reducers:no_reducers ~resolve_data:(resolve_data table) e

let%expect_test
    "eval_index: Data reads the resolver's raw value and normalizes it exactly \
     like ATen" =
  let data = Index.data s0 (Coord.of_fn (fun _ -> Index.zero)) 8 in
  let table v = [ ((Source.to_int s0, 0), v) ] in
  Fmt.pr "in range positive: %a@." pp_res (ev (table 3L) data);
  Fmt.pr "in range zero:     %a@." pp_res (ev (table 0L) data);
  Fmt.pr "negative -1:       %a@." pp_res (ev (table (-1L)) data);
  Fmt.pr "negative -8:       %a@." pp_res (ev (table (-8L)) data);
  [%expect
    {|
    in range positive: 3
    in range zero:     0
    negative -1:       7
    negative -8:       0
    |}]

let%expect_test
    "eval_index: Data rejects an out-of-range raw value, positive and negative"
    =
  let data = Index.data s0 (Coord.of_fn (fun _ -> Index.zero)) 8 in
  let table v = [ ((Source.to_int s0, 0), v) ] in
  Fmt.pr "%a@." pp_res (ev (table 8L) data);
  Fmt.pr "%a@." pp_res (ev (table (-9L)) data);
  [%expect
    {|
    gather index 8 out of range [-8, 7]
    gather index -9 out of range [-8, 7]
    |}]

let%expect_test
    "eval_index: Data's own resolver failure propagates through the caller's \
     row" =
  let data = Index.data s0 (Coord.of_fn (fun _ -> Index.zero)) 8 in
  Fmt.pr "%a@." pp_res (ev [] data);
  [%expect {| missing |}]

let%expect_test
    "eval_index: Data's coordinate is itself evaluated, so a nested Data \
     resolves too" =
  (* Read source 1 at whatever source 0's own value resolves to, then use
     THAT as the gather value. *)
  let inner = Index.data s0 (Coord.of_fn (fun _ -> Index.zero)) 8 in
  let outer =
    Index.data s1 (Coord.set (Coord.of_fn (fun _ -> Index.zero)) Axis.C inner) 8
  in
  let table = [ ((Source.to_int s0, 0), 2L); ((Source.to_int s1, 2), 5L) ] in
  Fmt.pr "%a@." pp_res (ev table outer);
  [%expect {| 5 |}]

let%expect_test
    "Fold.sources sees a Data-embedded source inside a Load's coordinate" =
  let data = Index.data s1 (Coord.of_fn (fun _ -> Index.zero)) 8 in
  let e =
    Value.load s0 (Coord.set (Coord.of_fn (fun _ -> Index.zero)) Axis.C data)
  in
  Fmt.pr "%a@."
    (Fmt.list ~sep:(Fmt.any ",") Source.pp)
    (Source.Set.elements (Fold.sources e));
  [%expect {| t0,t1 |}]

let%expect_test "Rewrite.map_sources rewrites a Data node's own source too" =
  let data = Index.data s1 (Coord.of_fn (fun _ -> Index.zero)) 8 in
  let e =
    Value.load s0 (Coord.set (Coord.of_fn (fun _ -> Index.zero)) Axis.C data)
  in
  let bump s = Source.create (Source.to_int s + 10) in
  let e' = Rewrite.map_sources bump e in
  Fmt.pr "%a@."
    (Fmt.list ~sep:(Fmt.any ",") Source.pp)
    (Source.Set.elements (Fold.sources e'));
  [%expect {| t10,t11 |}]

let%expect_test
    "Rewrite.freshen/Check.value reach a reducer nested inside a Data \
     coordinate" =
  let open Builder.Syntax in
  let e =
    Builder.run
      (Builder.reduction ~kind:Reduction.Sum ~lo:Index.zero ~hi:(Index.const 4)
         (fun r ->
           let+ () = Builder.return () in
           Value.load s0
             (Coord.set
                (Coord.of_fn (fun _ -> Index.zero))
                Axis.C
                (Index.data s1 (Coord.of_fn (fun _ -> r)) 8))))
  in
  Fmt.pr "%a@." Pp.value e;
  [%expect {| sum(r1=0..4: t0[0,0,0,0,0,data(t1[r1,r1,r1,r1,r1,r1],8)]) |}];
  Fmt.pr "well scoped: %a@."
    (Core.Pretty.err_result ~ok:(Fmt.any "ok") ~error:Check.pp_error)
    (Check.value e);
  [%expect {| well scoped: ok |}]
