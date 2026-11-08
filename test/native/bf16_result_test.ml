(* A BF16-declared result ([To_copy] to [Bfloat16]): the per-node reference,
   the stored and fused kernel runs and [Stage_program.ground] all store
   [Expr.Value.bf16_value] of the binary32 value, the one definition (round to
   nearest even on the high half, every NaN the canonical quiet [0x7FC0]). The
   inputs are the cases that tell a rounding apart: ties either way, a value
   just past a tie, subnormals, the overflow to infinity, infinities, and NaNs
   whose payload sits in the low half or carries the sign. *)

open Graph_ir

let inputs =
  [|
    (0x3F808000l, "tie, even below");
    (0x3F818000l, "tie, odd below");
    (0x3F808001l, "past a tie");
    (0x00008000l, "subnormal tie to zero");
    (0x80018000l, "negative subnormal tie up");
    (0x7F7FFFFFl, "largest finite");
    (0xFF800000l, "-infinity");
    (0x7FFFFFFFl, "NaN, all payload");
    (0xFFC00001l, "negative NaN");
    (0x7F800001l, "NaN, low payload");
  |]

let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:(Array.length inputs)

let x =
  Tensor.materialize shape (fun c ->
      Int32.float_of_bits (fst inputs.(Dim.to_int (Vec6.get c Axis.C))))

let build =
  Graph_builder.(
    build ~name:"bf16_result" ~outputs:(fun r -> [ r ])
    @@
    let* x = input ~shape ~name:"x" () in
    let* h = to_copy ~name:"h" Pointwise.To_copy.Bfloat16 x in
    to_copy ~name:"out" Pointwise.To_copy.Float h)
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let h_id =
  match List.nth build.Graph.nodes 0 with
  | { Node.outputs = id :: _; _ } -> id
  | _ -> assert false

let out_id = List.hd build.Graph.outputs

let bind id =
  if Tensor_id.equal id (List.hd build.Graph.inputs) then Some x else None

(* the high half of a cell's binary32 pattern: a bfloat16 cell's bits *)
let half t c =
  Int32.to_int
    (Int32.shift_right_logical
       (Int32.bits_of_float
          (Tensor.read t (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c)))
       16)
  land 0xFFFF

let row name t =
  Fmt.pr "%-8s" name;
  Array.iteri (fun c _ -> Fmt.pr " %04x" (half t c)) inputs;
  Fmt.pr "@."

let fmt_of (Tensor.Tensor t) = Payload.fmt_name t.Tensor.payload.Payload.fmt

let%expect_test "every engine stores the one bfloat16 rounding" =
  Fmt.pr "defined ";
  Array.iter
    (fun (b, _) ->
      Fmt.pr " %04x"
        (Int32.to_int
           (Int32.shift_right_logical
              (Int32.bits_of_float
                 (Expr.Value.bf16_value (Int32.float_of_bits b)))
              16)
        land 0xFFFF))
    inputs;
  Fmt.pr "@.";
  let env =
    Eval_direct.run build ~inputs:[ (List.hd build.Graph.inputs, x) ]
    |> Err.or_raise ~pp_error:Eval_direct.pp_error
  in
  let h = Tensor_id.Map.find h_id env in
  Fmt.pr "reference h: %s@." (fmt_of h);
  row "per-node" h;
  row "  out" (Tensor_id.Map.find out_id env);
  let kernel =
    Kernel_adapt.of_stage_program (Eval_symbolic.run build)
    |> Err.or_raise ~pp_error:Kernel_adapt.pp_error
  in
  let stored =
    Kernel_eval.run kernel ~bind |> Err.or_raise ~pp_error:Kernel_eval.pp_error
  in
  Fmt.pr "stored h: %s@." (fmt_of (Tensor_id.Map.find h_id stored));
  row "stored" (Tensor_id.Map.find h_id stored);
  row "  out" (Tensor_id.Map.find out_id stored);
  let plan, decisions = Fusion_plan.plan kernel in
  List.iter (Fmt.pr "%a@." Fusion_plan.Decision.pp) decisions;
  let fused =
    Kernel_eval.run_plan plan ~bind
    |> Err.or_raise ~pp_error:Kernel_eval.pp_error
  in
  row "fused" (Tensor_id.Map.find h_id fused);
  row "  out" (Tensor_id.Map.find out_id fused);
  let grounded =
    Stage_program.ground (Eval_symbolic.run build) ~bind:(fun id ->
        Option.get (bind id))
    |> Err.or_raise ~pp_error:Stage_program.pp_error
  in
  row "grounded" (Tensor_id.Map.find h_id grounded);
  [%expect
    {|
    defined  3f80 3f82 3f81 0000 8002 7f80 ff80 7fc0 7fc0 7fc0
    reference h: bf16
    per-node 3f80 3f82 3f81 0000 8002 7f80 ff80 7fc0 7fc0 7fc0
      out    3f80 3f82 3f81 0000 8002 7f80 ff80 7fc0 7fc0 7fc0
    stored h: bf16
    stored   3f80 3f82 3f81 0000 8002 7f80 ff80 7fc0 7fc0 7fc0
      out    3f80 3f82 3f81 0000 8002 7f80 ff80 7fc0 7fc0 7fc0
    reject t1: t1->t2 is not a pointwise site (t1->t2 is outside the supported site class)
    fused    3f80 3f82 3f81 0000 8002 7f80 ff80 7fc0 7fc0 7fc0
      out    3f80 3f82 3f81 0000 8002 7f80 ff80 7fc0 7fc0 7fc0
    grounded 3f80 3f82 3f81 0000 8002 7f80 ff80 7fc0 7fc0 7fc0 |}]

(* The double rounding is the definition: a binary64 value just past a
   bfloat16 tie, which binary32 rounds onto the tie, stores as the even
   neighbour, not the one a single rounding from binary64 would give. *)
let%expect_test "binary64 to binary32 to bfloat16" =
  let x = 1. +. ldexp 1. (-8) +. ldexp 1. (-30) in
  Fmt.pr "%08lx@." (Int32.bits_of_float (Expr.Value.bf16_value x));
  [%expect {| 3f800000 |}]
