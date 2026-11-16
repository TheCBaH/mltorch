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

(* ---- pool order and divisor, padding as a skipped tap ---------------------

   The same oracle over three more graphs. The values are read from a flat
   array in NHWC order; [constants] feeds the bundle's constants in creation
   order. *)

let eval g ~input ~constants =
  let b = bundle g in
  let bind values id = (id, tensor_of (Array.get values) (sig_of g id)) in
  let cs = List.map2 (fun id v -> bind v id) b.constants constants in
  let xs = List.map (bind input) b.inputs in
  match
    Err.or_raise ~pp_error:Loop_ir.Loop_bundle_oracle.pp_error
      (Loop_ir.Loop_bundle_oracle.run b
         ~constants:(fun id -> List.assoc_opt id cs)
         ~inputs:(fun id -> List.assoc_opt id xs))
  with
  | [ y ] -> y
  | _ -> failwith "one output"

let values ?(channels = 1) ~w y =
  let (Tensor.Tensor t) = y in
  let n = (Vec6.numel t.Tensor.shape :> int) in
  Array.init n (fun i ->
      Tensor.read_at_raw32 y (function
        | Axis.H -> i / channels / w
        | W -> i / channels mod w
        | C -> i mod channels
        | N | T | D -> 0))

let pool ~count_include_pad ~kernel ~stride ~pad:padding ~h ~w =
  let hw v = { Core.Geometry.Hw.h = v; w = v } in
  F.build "pool"
    Graph_builder.(
      let* x = input ~shape:(F.nhwc ~h ~w ~c:1) () in
      avg_pool2d
        {
          Pool.AvgPool2d.ceil_mode = false;
          count_include_pad;
          kernel = hw (Dim.extent kernel);
          stride = hw (Op_config.Pos.of_int stride);
          pad = hw (Op_config.Nonneg.of_int padding);
        }
        x)

(* [[2^24, 1], [3, -2^24]] in one window: row by row, each row summed, then the
   rows, gives 3; column by column 5; one flat sum in row order 4. The pool
   sums the rows' sums, and divides last. *)
let%expect_test "average pool: nested row sums, then the divisor" =
  let w = [| 16777216.; 1.; 3.; -16777216. |] in
  let r32 = F32_math.round32 in
  let ( +. ) a b = r32 (a +. b) in
  Fmt.pr "row-nested %g, column-nested %g, flat %g@."
    (w.(0) +. w.(1) +. (w.(2) +. w.(3)))
    (w.(0) +. w.(2) +. (w.(1) +. w.(3)))
    (w.(0) +. w.(1) +. w.(2) +. w.(3));
  let y =
    values ~w:1
      (eval
         (pool ~count_include_pad:true ~kernel:2 ~stride:2 ~pad:0 ~h:2 ~w:2)
         ~input:w ~constants:[])
  in
  Fmt.pr "pool: %g (the row-nested sum, 3, over 4)@." y.(0);
  [%expect
    {|
    row-nested 3, column-nested 5, flat 4
    pool: 0.75 (the row-nested sum, 3, over 4) |}]

(* A 2x2 window at stride 2 and padding 1 over a 2x2 input of ones: each
   output window holds one real element. The divisor is the kernel area, 4,
   with the padding counted, and the window's real size, 1, without. *)
let%expect_test "average pool: the divisor under padding" =
  List.iter
    (fun count_include_pad ->
      let y =
        values ~w:2
          (eval
             (pool ~count_include_pad ~kernel:2 ~stride:2 ~pad:1 ~h:2 ~w:2)
             ~input:[| 1.; 1.; 1.; 1. |] ~constants:[])
      in
      Fmt.pr "count_include_pad=%b: %a@." count_include_pad
        (Fmt.array ~sep:(Fmt.any " ") Fmt.float)
        y)
    [ true; false ];
  [%expect
    {|
    count_include_pad=true: 0.25 0.25 0.25 0.25
    count_include_pad=false: 1 1 1 1 |}]

(* Padding is not an added zero tap: a window clips to the input, so a padded
   tap is never multiplied. Weight infinity at the corner tap would turn an
   added zero into NaN (0 * inf); the output at that corner stays finite. *)
let%expect_test "convolution padding skips the tap" =
  let g =
    F.build "padded"
      Graph_builder.(
        let* x = input ~shape:(F.nhwc ~h:2 ~w:2 ~c:1) () in
        let* w = constant ~shape:(F.s 1 1 1 2 2 1) () in
        conv2d
          {
            Conv.Conv2d.h = F.conv_axis ~kernel:2 ~stride:1 ~pad:1;
            w = F.conv_axis ~kernel:2 ~stride:1 ~pad:1;
            in_channels = Dim.extent 1;
            groups = Op_config.Pos.of_int 1;
          }
          ~x ~weight:w ())
  in
  let y =
    values ~w:3
      (eval g ~input:[| 1.; 2.; 3.; 4. |]
         ~constants:[ [| infinity; 1.; 1.; 1. |] ])
  in
  Fmt.pr "3x3 output: %a@." (Fmt.array ~sep:(Fmt.any " ") Fmt.float) y;
  [%expect {| 3x3 output: 1 3 2 4 inf inf 3 inf inf |}]
