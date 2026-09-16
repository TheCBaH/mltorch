(* [Eval_direct4]'s dtype-preserving [Reshape4]/[Permute4] dispatch, the
   Native4D twin of [Eval_direct]'s own P5.2 fix (`reshape_i64_test.ml`/
   `permute_i64_test.ml`): the default arm's delegation to Native's
   [Reshape.Reshape.Compute(S).pixel]/[Permute.Permute.Compute(S).pixel]
   reads through [SEMANTICS.load], which round-trips every format through
   [Payload.get_float] and is silently lossy for an exact I64 value above
   float's 2^53 mantissa. [Builder.reshape4]/[permute4] now thread the
   operand's I64 format onto the output edge, and [Eval_direct4]'s new
   arms route an I64 source through Native's own [Compute_i64] functors
   instead -- never touching float. Native4D is a required execution route
   per the plan's own section 1 ("Native Direct, Native Symbolic, ...
   Native4D Direct/Symbolic, and the JSOO general evaluator"), and this
   pair had no I64 dispatch at all before this session. *)

open Native4d

let read t ~n ~h ~w ~c =
  Err.or_raise
    ~pp_error:(fun fmt (`Wrong_format (Payload.Fmt f)) ->
      Fmt.pf fmt "wrong format %a" Payload.pp_fmt f)
    (Tensor.read_i64_at6 t (function
      | Axis.N -> n
      | Axis.H -> h
      | Axis.W -> w
      | Axis.C -> c
      | Axis.T | Axis.D -> 0))

let%expect_test
    "direct4: I64 reshape4 [H=2 W=3 C=1] -> [W=6] stays exact past 2^53" =
  let in_shape = Shape4.of_ints ~n:1 ~h:2 ~w:3 ~c:1 in
  let out_shape = Shape4.of_ints ~n:1 ~h:1 ~w:6 ~c:1 in
  let g =
    Builder.build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape:in_shape ~fmt:Payload.(Fmt I64) () in
       reshape4 out_shape x)
    |> Err.or_raise ~pp_error:Builder.pp_error
  in
  (* Row-major over [H,W]: value = 9_007_199_254_740_993L + (h*3+w), so every
     one of the 6 cells sits strictly above 2^53 and is distinct -- a float
     round trip would collapse the first two. *)
  let x =
    Tensor.materialize_i64 (Shape4.to_vec6 in_shape) (fun c ->
        Int64.add 9_007_199_254_740_993L
          (Int64.of_int
             ((Dim.to_int (Vec6.get c Axis.H) * 3)
             + Dim.to_int (Vec6.get c Axis.W))))
  in
  let env =
    Eval_direct4.run g ~inputs:(List.combine g.Graph.Graph.inputs [ x ])
    |> Err.or_raise ~pp_error:Eval_direct4.pp_error
  in
  let out = Tensor_id.Map.find (List.hd g.Graph.Graph.outputs) env in
  Fmt.pr "%Ld,%Ld,%Ld,%Ld,%Ld,%Ld@."
    (read out ~n:0 ~h:0 ~w:0 ~c:0)
    (read out ~n:0 ~h:0 ~w:1 ~c:0)
    (read out ~n:0 ~h:0 ~w:2 ~c:0)
    (read out ~n:0 ~h:0 ~w:3 ~c:0)
    (read out ~n:0 ~h:0 ~w:4 ~c:0)
    (read out ~n:0 ~h:0 ~w:5 ~c:0);
  [%expect
    {| 9007199254740993,9007199254740994,9007199254740995,9007199254740996,9007199254740997,9007199254740998 |}]

let%expect_test "direct4: I64 permute4 swaps H and W and stays exact past 2^53"
    =
  let in_shape = Shape4.of_ints ~n:1 ~h:2 ~w:3 ~c:1 in
  let g =
    Builder.build
      ~outputs:(fun o -> [ o ])
      (let open Builder in
       let* x = input ~shape:in_shape ~fmt:Payload.(Fmt I64) () in
       permute4 (Ops4.Permute4.of_fn (function H -> W | W -> H | a -> a)) x)
    |> Err.or_raise ~pp_error:Builder.pp_error
  in
  let x =
    Tensor.materialize_i64 (Shape4.to_vec6 in_shape) (fun c ->
        Int64.add 9_007_199_254_740_993L
          (Int64.of_int
             ((Dim.to_int (Vec6.get c Axis.H) * 3)
             + Dim.to_int (Vec6.get c Axis.W))))
  in
  let env =
    Eval_direct4.run g ~inputs:(List.combine g.Graph.Graph.inputs [ x ])
    |> Err.or_raise ~pp_error:Eval_direct4.pp_error
  in
  let out = Tensor_id.Map.find (List.hd g.Graph.Graph.outputs) env in
  (* Transposed: out[w,h] = in[h,w], so reading along the new H axis (old W)
     at each new W position (old H) recovers the same distinct values. *)
  Fmt.pr "%Ld,%Ld,%Ld,%Ld,%Ld,%Ld@."
    (read out ~n:0 ~h:0 ~w:0 ~c:0)
    (read out ~n:0 ~h:1 ~w:0 ~c:0)
    (read out ~n:0 ~h:2 ~w:0 ~c:0)
    (read out ~n:0 ~h:0 ~w:1 ~c:0)
    (read out ~n:0 ~h:1 ~w:1 ~c:0)
    (read out ~n:0 ~h:2 ~w:1 ~c:0);
  [%expect
    {| 9007199254740993,9007199254740994,9007199254740995,9007199254740996,9007199254740997,9007199254740998 |}]
