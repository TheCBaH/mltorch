(* Native SOURCE graphs, one per row of the Native4D domain contract table in
   .ai/native4d_plan.md. Named after the property each exercises, following
   test/native/graph_fixtures.ml.

   These are all plain [Graph_ir.graph]s: stage 0 checks the partiality contract
   against the Native dialect, before any Native4D IR exists. *)

let s n t d h w c = Vec6.shape ~n ~t ~d ~h ~w ~c
let nhwc ~n ~h ~w ~c = s n 1 1 h w c
let chan c = s 1 1 1 1 1 c

let build name m =
  Graph_builder.build ~name ~outputs:(fun o -> [ o ]) m
  |> Core.or_raise (fun ppf e ->
      Fmt.pf ppf "fixture %s: %a" name Graph_builder.pp_error e)

let conv_axis ~kernel : Conv.Conv2d.axis_window =
  {
    kernel = Dim.extent kernel;
    stride = Op_config.Pos.of_int 1;
    pad_before = Op_config.Nonneg.of_int 0;
    pad_after = Op_config.Nonneg.of_int 0;
    dilation = Op_config.Pos.of_int 1;
  }

let conv_params ~in_channels ~groups : Conv.Conv2d.params =
  {
    h = conv_axis ~kernel:1;
    w = conv_axis ~kernel:1;
    in_channels = Dim.extent in_channels;
    groups = Op_config.Pos.of_int groups;
  }

let hw n : 'a Op_config.Hw.t = { h = n; w = n }

let convolution_params ~transposed ~groups : Conv.Convolution.params =
  {
    stride = hw (Op_config.Pos.of_int 1);
    padding = hw (Op_config.Nonneg.of_int 0);
    dilation = hw (Op_config.Pos.of_int 1);
    transposed;
    output_padding = hw (Op_config.Nonneg.of_int 0);
    groups = Op_config.Pos.of_int groups;
  }

(* 1x1 weights: [Cout,1,1,1,1,Cin/groups]. *)
let weight ~out_channels ~in_per_group = s out_channels 1 1 1 1 in_per_group

let perm_of pairs =
  Permute.Permute.of_fn (fun a ->
      Option.value (List.assoc_opt a pairs) ~default:a)

(* ---- the four-axis invariant ---------------------------------------------- *)

let all_four_axis () =
  build "all_four_axis"
    (let open Graph_builder in
     let* x = input ~shape:(nhwc ~n:2 ~h:4 ~w:4 ~c:3) () in
     relu x)

let non_unit_t () =
  build "non_unit_t"
    (let open Graph_builder in
     let* x = input ~shape:(s 1 2 1 4 4 3) () in
     relu x)

let non_unit_d () =
  build "non_unit_d"
    (let open Graph_builder in
     let* x = input ~shape:(s 1 1 2 4 4 3) () in
     relu x)

(* A captured weight with a non-4D shape that no node reads. Outside both halves
   of the shape rule, so accepted — the lowerer omits it. *)
let unread_constant_non_4d () =
  build "unread_constant_non_4d"
    (let open Graph_builder in
     let* x = input ~shape:(nhwc ~n:1 ~h:4 ~w:4 ~c:3) () in
     let* _dead = constant ~shape:(s 1 1 5 4 4 3) () in
     relu x)

(* The companion to [unread_constant_non_4d]: the same non-4D constant, but READ.
   Without this row the accepting case proves nothing — a shape rule that skipped
   every constant would pass it just as happily. Broadcast makes the sum non-4D
   too, but the constant is the lower id and so the reported one. *)
let read_constant_non_4d () =
  build "read_constant_non_4d"
    (let open Graph_builder in
     let* x = input ~shape:(nhwc ~n:1 ~h:4 ~w:4 ~c:3) () in
     let* w = constant ~shape:(s 1 1 5 1 1 1) () in
     add x w)

(* A non-4D USER input that nothing reads. DCE keeps it — the signature is
   externally meaningful — so it must be rejected rather than omitted. *)
let unused_input_non_4d () =
  build "unused_input_non_4d"
    (let open Graph_builder in
     let* x = input ~shape:(nhwc ~n:1 ~h:4 ~w:4 ~c:3) () in
     let* _unused = input ~shape:(s 1 1 5 4 4 3) () in
     relu x)

(* ---- axis-naming ops ------------------------------------------------------ *)

let mean_over_hw ~keepdim ~n () =
  build "mean_over_hw"
    (let open Graph_builder in
     let* x = input ~shape:(nhwc ~n ~h:4 ~w:4 ~c:3) () in
     mean { Reduce.Mean.dims = Axis.[ H; W ]; keepdim } x)

let mean_over_d () =
  build "mean_over_d"
    (let open Graph_builder in
     let* x = input ~shape:(nhwc ~n:1 ~h:4 ~w:4 ~c:3) () in
     mean { Reduce.Mean.dims = [ Axis.D ]; keepdim = true } x)

let permute_hw () =
  build "permute_hw"
    (let open Graph_builder in
     let* x = input ~shape:(nhwc ~n:1 ~h:4 ~w:2 ~c:3) () in
     permute (perm_of Axis.[ (H, W); (W, H) ]) x)

let permute_c_onto_d () =
  build "permute_c_onto_d"
    (let open Graph_builder in
     let* x = input ~shape:(nhwc ~n:1 ~h:4 ~w:4 ~c:3) () in
     permute (perm_of Axis.[ (D, C); (C, D) ]) x)

let rms_norm_over dims () =
  build "rms_norm"
    (let open Graph_builder in
     let* x = input ~shape:(nhwc ~n:1 ~h:4 ~w:4 ~c:3) () in
     rms_norm { Norm.RmsNorm.dims; eps = 1e-5 } ~x ())

(* ---- batch norm ----------------------------------------------------------- *)

let batch_norm_on ?(dynamic = false) channel () =
  build "batch_norm"
    (let open Graph_builder in
     let* x = input ~shape:(nhwc ~n:1 ~h:4 ~w:4 ~c:3) () in
     let stat () =
       if dynamic then
         (* A node output, not a captured constant: nothing to precompute from. *)
         let* c = input ~shape:(chan 3) () in
         relu c
       else constant ~shape:(chan 3) ()
     in
     let* running_mean = stat () in
     let* running_var = stat () in
     batch_norm
       { Norm.BatchNorm.channel; eps = 1e-5 }
       ~x ~running_mean ~running_var ())

(* A parameter vector shorter than the axis it scales. Nothing upstream rejects
   it because nothing VALIDATES it: [Norm.BatchNorm.output_shape] is a function
   of the input shape alone, so the parameters' extents are never compared
   against the normalized axis and [Graph_view] has nothing to check. Native's
   own evaluation does not survive it either — [BatchNorm.Compute] reads each
   parameter at the OUTPUT's channel index and [Tensor.read] is strict — so the
   graph is broken for both dialects; the difference is only that conversion
   discovers it earlier. *)
let batch_norm_short_stats () =
  build "batch_norm_short_stats"
    (let open Graph_builder in
     let* x = input ~shape:(nhwc ~n:1 ~h:1 ~w:1 ~c:2) () in
     let* running_mean = constant ~shape:(chan 1) () in
     let* running_var = constant ~shape:(chan 1) () in
     batch_norm
       { Norm.BatchNorm.channel = Axis.C; eps = 1e-5 }
       ~x ~running_mean ~running_var ())

(* ---- convolution --------------------------------------------------------- *)

let conv2d_grouped ~groups ~in_channels ~out_channels () =
  build "conv2d_grouped"
    (let open Graph_builder in
     let* x = input ~shape:(nhwc ~n:1 ~h:4 ~w:4 ~c:in_channels) () in
     let* w =
       constant
         ~shape:(weight ~out_channels ~in_per_group:(in_channels / groups))
         ()
     in
     conv2d (conv_params ~in_channels ~groups) ~x ~weight:w ())

let convolution_grouped ~transposed ~groups ~channels () =
  build "convolution_grouped"
    (let open Graph_builder in
     let* x = input ~shape:(nhwc ~n:1 ~h:4 ~w:4 ~c:channels) () in
     (* Transposed weight is [Cin,1,1,Kh,Kw,Cout/groups]. *)
     let* w =
       constant
         ~shape:
           (weight ~out_channels:channels ~in_per_group:(channels / groups))
         ()
     in
     convolution (convolution_params ~transposed ~groups) ~x ~weight:w ())

(* ---- bmm ----------------------------------------------------------------- *)

(* mat2 in a format f32 cannot hold exactly. The legalization MATERIALIZES the
   permuted mat2 before the convolution reads it, where Native's Bmm reads it
   directly, and every op output in this engine is f32 — so those values would
   be silently rounded while the map still claimed Identical. *)
let bmm_lossy_operand () =
  build "bmm_lossy_operand"
    (let open Graph_builder in
     let* a = input ~shape:(s 1 1 1 1 2 3) () in
     let* b = input ~shape:(s 1 1 1 1 3 4) ~fmt:(Payload.Fmt Payload.I64) () in
     bmm a b)

let bmm_batch batch () =
  build "bmm_batch"
    (let open Graph_builder in
     (* input[H=batch,W=rows,C=contract] x mat2[H=batch,W=contract,C=cols] *)
     let* a = input ~shape:(s 1 1 1 batch 2 3) () in
     let* b = input ~shape:(s 1 1 1 batch 3 4) () in
     bmm a b)

(* ---- max pool with indices ------------------------------------------------ *)

let pool_params : Pool.MaxPool2d.params =
  {
    kernel = hw (Dim.extent 2);
    stride = hw (Op_config.Pos.of_int 2);
    pad = hw (Op_config.Nonneg.of_int 0);
  }

(* The index edge routed into a [Discard] sink: dead, so stage 1 can rewrite the
   op away. Rejected here as unsupported, not as a live index. *)
let maxpool_indices_discarded () =
  build "maxpool_indices_discarded"
    (let open Graph_builder in
     let* x = input ~shape:(nhwc ~n:1 ~h:4 ~w:4 ~c:3) () in
     let* values, indices = max_pool2d_with_indices pool_params x in
     let* () = discard indices in
     return values)

(* The index edge consumed by a real op: nothing can remove it. *)
let maxpool_indices_live () =
  build "maxpool_indices_live"
    (let open Graph_builder in
     let* x = input ~shape:(nhwc ~n:1 ~h:4 ~w:4 ~c:3) () in
     let* values, indices = max_pool2d_with_indices pool_params x in
     let* live = relu indices in
     add values live)

(* ---- linear -------------------------------------------------------------- *)

(* Weight is [Out,1,1,1,1,In] — already a 1x1 convolution weight. *)
let linear_layer () =
  build "linear"
    (let open Graph_builder in
     let* x = input ~shape:(nhwc ~n:1 ~h:1 ~w:1 ~c:8) () in
     let* w = constant ~shape:(s 4 1 1 1 1 8) () in
     linear { Linear.Linear.in_features = Dim.extent 8 } ~x ~weight:w ())
