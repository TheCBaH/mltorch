(* [Eval_direct]'s dtype-preserving [Reshape] dispatch (this session's P5.2
   slice): the default [Reshape.Reshape.Compute(Direct).pixel] arm reads
   through [SEMANTICS.load], which round-trips every format through
   [Payload.get_float] and is therefore silently lossy for an exact I64
   value above float's 2^53 mantissa. [eval_direct.ml]'s new [Reshape] arm
   branches on the operand's declared signature format and routes an I64
   source through [Reshape.Reshape.Compute_i64], which reads via
   [Tensor.read_i64_at6] instead -- never touching float. See the
   implementation tracker's P5.2 note. *)

open Graph_ir
open Graph_direct_fixtures

let%expect_test
    "Direct graph: I64 reshape [H=2 W=3 C=1] -> [C=6] stays exact past 2^53" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"reshape_i64" ~outputs:(fun r -> [ r ])
          @@
          let* x =
            input ~shape:(s 1 1 1 2 3 1) ~name:"x" ~fmt:Payload.(Fmt I64) ()
          in
          reshape ~name:"out" { Reshape.Reshape.shape = s 1 1 1 1 1 6 } x)
    in
    (* Row-major over [H,W]: value = 9_007_199_254_740_993L + (h*3+w), so
       every one of the 6 cells sits strictly above 2^53 and is distinct --
       a float round trip would collapse the first two (9_007_199_254_740_993
       and ...994 both round to the same float). *)
    let x =
      Tensor.materialize_i64 (s 1 1 1 2 3 1) (fun c ->
          Int64.add 9_007_199_254_740_993L
            (Int64.of_int
               ((Dim.to_int (Vec6.get c Axis.H) * 3)
               + Dim.to_int (Vec6.get c Axis.W))))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect
    {|
    out = tensor i64 [C=6] {9007199254740993, 9007199254740994, 9007199254740995, 9007199254740996, 9007199254740997, 9007199254740998}
    |}]

(* [Graph_builder.reshape] itself had a companion defect this same session:
   [op1]'s default output edge format is F32 (the "arithmetic outputs are
   F32" convention documented at the top of graph_builder.ml), and unlike
   [unbind]/[split_with_sizes] (which explicitly retain their operand's
   format/quant), [reshape] never opted in -- so a Reshape node's OWN
   declared [Tensor_sig.fmt] was always F32, even when its operand (and
   therefore, via [Eval_direct]'s I64 dispatch above, its actual computed
   tensor) is I64. That mismatch is latent for a single reshape (nothing
   reads the mismatched sig), but a SECOND reshape consuming the first one's
   output looks up exactly that declared sig, not the runtime payload, to
   pick its own dispatch branch -- so it would wrongly take the float
   [Schedule.evaluate]/[Tensor.materialize] path (always F32), silently
   truncating past 2^53 for the second time, defeating the whole point of
   [Compute_i64] one hop later. Fixed by threading [~fmt]/[~quant] from the
   operand's own signature in [reshape], mirroring [unbind]. *)
let%expect_test
    "Direct graph: chained I64 reshapes -- the intermediate edge's own \
     declared signature stays I64, not the op1 default" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"reshape_i64_chain" ~outputs:(fun (mid, out) ->
              [ mid; out ])
          @@
          let* x =
            input ~shape:(s 1 1 1 2 3 1) ~name:"x" ~fmt:Payload.(Fmt I64) ()
          in
          let* mid = reshape { Reshape.Reshape.shape = s 1 1 1 1 1 6 } x in
          let* out = reshape { Reshape.Reshape.shape = s 1 1 1 1 2 3 } mid in
          return (mid, out))
    in
    let mid_id, out_id =
      match g.Graph.outputs with [ a; b ] -> (a, b) | _ -> assert false
    in
    let x =
      Tensor.materialize_i64 (s 1 1 1 2 3 1) (fun c ->
          Int64.add 9_007_199_254_740_993L
            (Int64.of_int
               ((Dim.to_int (Vec6.get c Axis.H) * 3)
               + Dim.to_int (Vec6.get c Axis.W))))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    let (Payload.Fmt mid_fmt) =
      (Tensor_id.Map.find mid_id g.Graph.tensors).Tensor_sig.fmt
    in
    let mid_tensor = Tensor_id.Map.find mid_id env in
    let out_tensor = Tensor_id.Map.find out_id env in
    Err.return (Payload.fmt_name mid_fmt, mid_tensor, out_tensor)
  in
  let pp_ok ppf (mid_fmt_name, mid_tensor, out_tensor) =
    Format.fprintf ppf "mid declared fmt = %s@.mid = %a@.out = %a" mid_fmt_name
      Tensor.pp mid_tensor Tensor.pp out_tensor
  in
  Format.printf "%a@." (pp_result pp_ok) result;
  [%expect
    {|
    mid declared fmt = i64
    mid = tensor i64 [C=6] {9007199254740993, 9007199254740994, 9007199254740995, 9007199254740996, 9007199254740997, 9007199254740998}
    out = tensor i64 [W=2 C=3] {9007199254740993, 9007199254740994, 9007199254740995, 9007199254740996, 9007199254740997, 9007199254740998}
    |}]

let%expect_test
    "Direct graph: I64 reshape preserves format through a non-flattening target"
    =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"reshape_i64_2d" ~outputs:(fun r -> [ r ])
          @@
          let* x =
            input ~shape:(s 1 1 1 2 3 1) ~name:"x" ~fmt:Payload.(Fmt I64) ()
          in
          reshape ~name:"out" { Reshape.Reshape.shape = s 1 1 1 3 2 1 } x)
    in
    let x =
      Tensor.materialize_i64 (s 1 1 1 2 3 1) (fun c ->
          Int64.add 9_007_199_254_740_993L
            (Int64.of_int
               ((Dim.to_int (Vec6.get c Axis.H) * 3)
               + Dim.to_int (Vec6.get c Axis.W))))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect
    {|
    out = tensor i64 [H=3 W=2 C=1] {9007199254740993, 9007199254740994, 9007199254740995, 9007199254740996, 9007199254740997, 9007199254740998}
    |}]
