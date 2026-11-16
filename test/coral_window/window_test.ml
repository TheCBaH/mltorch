(* The binary32 counterexample for a window kernel with no overlap: 2x2,
   stride 2, no padding, two input channels, all weights 1, no bias; channel 0
   is [[1,0],[0,0]], channel 1 is [[2^24,-2^24],[0,0]].

   The reference reduces in nested sums, channel, then kernel row, then kernel
   column, each with its own accumulator, so channel 1 sums to 0 on its own and
   the output is 1. A flat sum over the same taps in the same lexical order
   absorbs the 1 into 2^24 and gives 0. A window kernel must return 1; a
   recognizer that read the window as a flat product must refuse it. *)

open Graph_ir
module F = Native_test.Graph_fixtures

let graph () =
  F.build "window_counterexample"
    Graph_builder.(
      let* x = input ~shape:(F.nhwc ~h:2 ~w:2 ~c:2) () in
      let* w = constant ~shape:(F.s 1 1 1 2 2 2) () in
      conv2d
        {
          Conv.Conv2d.h = F.conv_axis ~kernel:2 ~stride:2 ~pad:0;
          w = F.conv_axis ~kernel:2 ~stride:2 ~pad:0;
          in_channels = Dim.extent 2;
          groups = Op_config.Pos.of_int 1;
        }
        ~x ~weight:w ())

(* NHWC offsets: (h * 2 + w) * 2 + c. *)
let taps = [| 1.; 16777216.; 0.; -16777216.; 0.; 0.; 0.; 0. |]

let tensor_of values (sg : Tensor_sig.t) =
  Tensor.materialize sg.Tensor_sig.shape (fun c ->
      values (Vec6.offset sg.Tensor_sig.shape c :> int))

let bundle g =
  Err.or_raise ~pp_error:Loop_ir.Loop_bundle.pp_error
    (Loop_ir.Loop_bundle.build g)

let sig_of (g : graph) id = Tensor_id.Map.find id g.Graph.tensors

(* The ordered binary32 oracle of the graph: the Loop interpreter at binary32,
   the golden of the ordered policy. *)
let oracle ?accumulate g =
  let b = bundle g in
  let bind id values = (id, tensor_of values (sig_of g id)) in
  let constants = List.map (fun id -> bind id (fun _ -> 1.)) b.constants in
  let inputs = List.map (fun id -> bind id (Array.get taps)) b.inputs in
  Err.or_raise ~pp_error:Loop_ir.Loop_bundle_oracle.pp_error
    (Loop_ir.Loop_bundle_oracle.run ?accumulate b
       ~constants:(fun id -> List.assoc_opt id constants)
       ~inputs:(fun id -> List.assoc_opt id inputs))

(* The window as one contraction over its eight taps in the nested order (channel,
   row, column): the flat product the convolution is not. *)
let flat numerics =
  let d =
    Result.get_ok
      (Err.payload
         (Contraction.make ~batch:1L ~m:1L ~n:1L ~k:8L
            ~a:{ Contraction.Operand.batch = 0L; outer = 8L; k = 1L }
            ~b:{ Contraction.Operand.batch = 0L; outer = 1L; k = 1L }
            ~c:{ Contraction.Output.batch = 0L; row = 1L; col = 1L }
            ~format:Contraction.Element.F32 ~seed:Contraction.Seed.Zero
            ~numerics))
  in
  let out = ref nan in
  Contraction.eval d
    ~a:(fun i ->
      let i = Int64.to_int i in
      taps.((i mod 4 * 2) + (i / 4)))
    ~b:(fun _ -> 1.)
    ~c:(fun _ -> 0.)
    ~write:(fun _ v -> out := v);
  !out

let%expect_test "nested sums give 1, the flat sum 0" =
  Fmt.pr "nested (ordered binary32 oracle): %a@."
    (Fmt.list ~sep:Fmt.comma Tensor.pp)
    (oracle (graph ()));
  Fmt.pr "flat, ordered: %g@." (flat Contraction.Numerics.Ordered);
  Fmt.pr "flat, fused: %g@." (flat Contraction.Numerics.Fused_sequential);
  [%expect
    {|
    nested (ordered binary32 oracle): tensor f32 [C=1] {1}
    flat, ordered: 0
    flat, fused: 0 |}]

(* The recognizer reads only nested sums of extent 1 as a product, so the window
   is refused rather than turned into the flat form. *)
let%expect_test "the recognizer refuses the window" =
  let b = bundle (graph ()) in
  List.iter
    (fun inv ->
      match Ssa_backends.program Ssa_backends.Pipeline.Exact inv with
      | Error e -> Fmt.pr "%s@." e
      | Ok p ->
          List.iter
            (function
              | Ssa_ir.Ssa_contract.Outcome.Found f ->
                  Fmt.pr "found %a@." Contraction.pp
                    f.Ssa_ir.Ssa_contract.Found.desc
              | Ssa_ir.Ssa_contract.Outcome.Refused r ->
                  Fmt.pr "refused %s@." (Ssa_ir.Ssa_contract.Refusal.name r))
            (Ssa_ir.Ssa_contract.census ~numerics:Contraction.Numerics.Ordered
               (Ssa_ir.Ssa_plan.binary32_scalar p)))
    b.invocations;
  [%expect {|
    refused bounds
    refused bounds
    refused non_affine |}]
