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
  |> Err.or_raise ~pp_error:(fun ppf e ->
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

(* Both affine operands present, so the fixture also carries the two extra
   edges the lowerer has to map. [dims] is the parameter under test, and the
   affine shape is DERIVED from it -- the operands carry the normalized extents
   and 1 everywhere else, so a fixture hard-coding [chan] would be rejected by
   the shape rule for any multi-axis [dims] rather than reaching the domain
   check it exists to exercise. *)
let layer_norm_over dims () =
  let x_shape = nhwc ~n:1 ~h:4 ~w:4 ~c:3 in
  let affine =
    List.fold_left
      (fun acc a -> Vec6.set acc a (Vec6.get x_shape a))
      (s 1 1 1 1 1 1) dims
  in
  build "layer_norm"
    (let open Graph_builder in
     let* x = input ~shape:x_shape () in
     let* w = constant ~shape:affine () in
     let* b = constant ~shape:affine () in
     layer_norm { Norm.LayerNorm.dims; eps = 1e-5 } ~x ~weight:w ~bias:b ())

(* Small enough that [Tensor.pp] prints every element, so the golden carries the
   values and not only a shape.

   What it CANNOT see: [verify_test]'s [native_vs_four] fills every input from
   the same counter, so weight and bias arrive holding the same values and a
   lowering that swapped them is invisible here. That swap is caught by the
   lowering golden in lower_test.ml and by the structural verifier -- both were
   observed failing under it -- and this fixture is for the axis conversion and
   the reduction, not for the operand order. *)
let layer_norm_tiny () =
  build "layer_norm"
    (let open Graph_builder in
     (* Graph INPUTS, not constants: [native_vs_four] binds inputs and has no
        payload to bind a constant with, and the operand wiring is what this
        fixture is for. *)
     let* x = input ~shape:(s 1 1 1 1 2 3) () in
     let* w = input ~shape:(chan 3) () in
     let* b = input ~shape:(chan 3) () in
     layer_norm
       { Norm.LayerNorm.dims = [ Axis.C ]; eps = 1e-5 }
       ~x ~weight:w ~bias:b ())

(* Two groups of two channels each, the same shape [layer_norm_tiny] above
   uses but for [group_norm]'s channel-window reduction instead of
   [layer_norm]'s whole-channel one. *)
let group_norm_tiny () =
  build "group_norm"
    (let open Graph_builder in
     let* x = input ~shape:(s 1 1 1 1 2 4) () in
     let* w = input ~shape:(chan 4) () in
     let* b = input ~shape:(chan 4) () in
     group_norm
       {
         Norm.GroupNorm.channel = Axis.C;
         groups = Op_config.Pos.of_int 2;
         eps = 1e-5;
       }
       ~x ~weight:w ~bias:b ())

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

(* [Batched_matmul] is rejected unconditionally, the same argument as [Sdpa]
   (its own landing note, `.ai/matmul_softmax_design.md` §5): unlike [Bmm]'s
   single-batch escape hatch, D names no [Ops4] axis at any extent, including
   1, so there is no configuration this legalizes to. *)
let batched_matmul () =
  build "batched_matmul"
    (let open Graph_builder in
     (* input[D=batch,H=heads,W=rows,C=contract] x
        mat2[D=batch,H=heads,W=contract,C=cols] *)
     let* a = input ~shape:(s 1 1 2 2 2 3) () in
     let* b = input ~shape:(s 1 1 2 2 3 4) () in
     batched_matmul a b)

(* [Sdpa] is rejected unconditionally (op8-impl.md F8), so any admissible
   shape exercises it -- unlike [Bmm]'s single-batch escape hatch, there is no
   configuration this legalizes to. *)
let sdpa () =
  build "sdpa"
    (let open Graph_builder in
     (* query/key/value[D=batch,H=heads,W=sequence,C=head_dim] *)
     let* q = input ~shape:(s 1 1 2 2 3 4) () in
     let* k = input ~shape:(s 1 1 2 2 3 4) () in
     let* v = input ~shape:(s 1 1 2 2 3 4) () in
     sdpa
       { Attention.Sdpa.scale = Attention.Sdpa.Scale.Default }
       ~query:q ~key:k ~value:v ())

(* [Softmax] has no [Ops4] counterpart at any axis (unlike [Mean]/[Amax]/
   [Vector_norm], which have a [*_keepdims] target to legalize onto), so any
   admissible axis exercises the plain "dialect does not have it" rejection
   -- see .ai/matmul_softmax_design.md §3. *)
let softmax_over axis () =
  build "softmax_over"
    (let open Graph_builder in
     let* x = input ~shape:(nhwc ~n:1 ~h:4 ~w:4 ~c:3) () in
     softmax { Reduce.Softmax.axis } x)

(* [Expand4] now exists: this fans H from 1 to 4 and stays four-axis
   throughout, so it converts -- see domain_test.ml's "expand always admits"
   and lower_test.ml's [expand4] cases for the typed-target rejection this
   fixture does NOT exercise. *)
let expand () =
  build "expand"
    (let open Graph_builder in
     let* x = input ~shape:(nhwc ~n:1 ~h:1 ~w:4 ~c:3) () in
     expand { Pointwise.Expand.size = nhwc ~n:1 ~h:4 ~w:4 ~c:3 } x)

(* Small enough that [Tensor.pp] prints every element -- [layer_norm_tiny]'s
   reason above. W fans from 1 to 3; C stays fixed at 2. *)
let expand_tiny () =
  build "expand"
    (let open Graph_builder in
     let* x = input ~shape:(s 1 1 1 1 1 2) () in
     expand { Pointwise.Expand.size = s 1 1 1 1 3 2 } x)

(* ---- max pool with indices ------------------------------------------------ *)

let pool_params : Pool.MaxPool2d.params =
  {
    ceil_mode = false;
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

(* ---- unbind -------------------------------------------------------------- *)

(* The dialect's first multi-output source op, and the one whose representable
   set is narrower than its axis check suggests.

   Dropping an axis shifts every axis OUTSIDE it one place outward, so unbinding
   H/W/C moves N's extent onto T — inside the dialect only when N=1. Unbinding N
   always converts, since nothing lies outside it. Neither rule is written down
   anywhere: [Shape4.of_vec6] on each inferred slice is what decides, which is
   exactly what these fixtures are here to show.

   [build] returns a single output, so these expose the whole slice list
   instead — the point is that all of them survive. *)
let unbind_all name ~shape axis =
  Graph_builder.build ~name ~outputs:Fun.id
    (let open Graph_builder in
     let* x = input ~shape () in
     unbind { Split.Unbind.axis } x)
  |> Err.or_raise ~pp_error:(fun ppf e ->
      Fmt.pf ppf "fixture %s: %a" name Graph_builder.pp_error e)

(* [Unbind]'s rank-preserving sibling. KEEPING the axis means there is no
   shift for the batch extent to land on T -- the N=1 precondition
   [unbind_c_batch2]/[unbind_n] turn on simply does not arise here, which is
   exactly what [split_with_sizes_w_batch2] below is here to show. *)
let split_with_sizes_all name ~shape axis sizes =
  Graph_builder.build ~name ~outputs:Fun.id
    (let open Graph_builder in
     let* x = input ~shape () in
     split_with_sizes { Split.Split_with_sizes.axis; sizes } x)
  |> Err.or_raise ~pp_error:(fun ppf e ->
      Fmt.pf ppf "fixture %s: %a" name Graph_builder.pp_error e)

let pad_graph name ~shape ~pads ~mode () =
  Graph_builder.build ~name
    ~outputs:(fun o -> [ o ])
    (let open Graph_builder in
     let* x = input ~shape () in
     pad { Pad.Pad.pads; mode } x)
  |> Err.or_raise ~pp_error:(fun ppf e ->
      Fmt.pf ppf "fixture %s: %a" name Graph_builder.pp_error e)

(* Pad on two dialect axes at once, one growing and one cropping, so a lowering
   that carried only the first entry or dropped the sign is visible in the
   printed graph rather than only in a numeric comparison. *)
let pad_hw_crop =
  pad_graph "pad_hw_crop" ~shape:(nhwc ~n:1 ~h:4 ~w:4 ~c:2)
    ~pads:
      [
        (Axis.H, { Pad.Pad.before = 1; after = -1 });
        (Axis.W, { Pad.Pad.before = 0; after = 2 });
      ]
    ~mode:(Pad.Pad.Constant 0.25)

(* Small enough that its WHOLE output prints: [Tensor.pp] truncates after eight
   elements, and on the fixture above every one of those eight is fill, which
   would leave a numeric golden that no wrong pixel map could fail. *)
let pad_tiny =
  pad_graph "pad_tiny" ~shape:(nhwc ~n:1 ~h:2 ~w:2 ~c:1)
    ~pads:[ (Axis.H, { Pad.Pad.before = 1; after = 0 }) ]
    ~mode:(Pad.Pad.Constant 9.)

(* The same op naming D, which the frame has and the dialect does not. Refused
   by the AXIS rule, so the diagnostic names D rather than reporting a tensor
   that happens to have extent there. *)
let pad_d =
  pad_graph "pad_d" ~shape:(s 1 1 3 2 2 2)
    ~pads:[ (Axis.D, { Pad.Pad.before = 1; after = 1 }) ]
    ~mode:Pad.Pad.Reflect

let slice_graph name ~shape ~axis ~start ~stop ~step () =
  Graph_builder.build ~name
    ~outputs:(fun o -> [ o ])
    (let open Graph_builder in
     let* x = input ~shape () in
     slice { Split.Slice.axis; start; stop; step = Op_config.Pos.of_int step } x)
  |> Err.or_raise ~pp_error:(fun ppf e ->
      Fmt.pf ppf "fixture %s: %a" name Graph_builder.pp_error e)

(* A non-unit step on a dialect axis. Small enough that its whole output prints,
   for the reason [pad_tiny] gives. *)
let slice_w =
  slice_graph "slice_w" ~shape:(nhwc ~n:1 ~h:2 ~w:5 ~c:1) ~axis:Axis.W ~start:1
    ~stop:5 ~step:2

(* The same op naming D, refused by the AXIS rule so the diagnostic names D. *)
let slice_d =
  slice_graph "slice_d" ~shape:(s 1 1 3 2 2 2) ~axis:Axis.D ~start:0 ~stop:2
    ~step:1

(* [fold_left], not [fold_right]: the monadic actions are threaded through
   STATE at run time, so building the accumulator on the wrong side reverses
   which shape gets the lower tensor id -- appending onto the end here is what
   keeps the graph's input order, [xs]'s order, and [shapes]' order all the
   same list. *)
(* [Select]'s own version of [unbind_all]: single output, so [build]'s default
   ([fun o -> [ o ]]) is fine as-is. Dropping the axis is the same shift
   [Unbind]'s comment gives -- an axis outside the one named moves one place
   toward T, so these fixtures probe the same N=1/N=2 boundary. *)
let select_graph name ~shape ~axis ~index () =
  Graph_builder.build ~name
    ~outputs:(fun o -> [ o ])
    (let open Graph_builder in
     let* x = input ~shape () in
     select { Split.Select.axis; index } x)
  |> Err.or_raise ~pp_error:(fun ppf e ->
      Fmt.pf ppf "fixture %s: %a" name Graph_builder.pp_error e)

let concat_graph name ~shapes ~axis () =
  Graph_builder.build ~name
    ~outputs:(fun o -> [ o ])
    (let open Graph_builder in
     let* xs =
       List.fold_left
         (fun acc shape ->
           let* rest = acc in
           let* x = input ~shape () in
           return (rest @ [ x ]))
         (return []) shapes
     in
     concat { Concat.Concat.axis } xs)
  |> Err.or_raise ~pp_error:(fun ppf e ->
      Fmt.pf ppf "fixture %s: %a" name Graph_builder.pp_error e)

(* Joined on a dialect axis, with operands of DIFFERENT extents along it. *)
let concat_w =
  concat_graph "concat_w"
    ~shapes:[ nhwc ~n:1 ~h:2 ~w:1 ~c:1; nhwc ~n:1 ~h:2 ~w:2 ~c:1 ]
    ~axis:Axis.W

(* The same op naming D, refused by the AXIS rule so the diagnostic names D. *)
let concat_d =
  concat_graph "concat_d" ~shapes:[ s 1 1 1 2 2 2; s 1 1 2 2 2 2 ] ~axis:Axis.D

(* N=1, so unbinding C leaves every slice four-axis. *)
let unbind_c_batch1 () =
  unbind_all "unbind_c_batch1" ~shape:(nhwc ~n:1 ~h:2 ~w:2 ~c:3) Axis.C

(* Batch 2, unbound along N: the outermost axis, so nothing shifts onto T. *)
let unbind_n () = unbind_all "unbind_n" ~shape:(nhwc ~n:2 ~h:2 ~w:2 ~c:3) Axis.N

(* Batch 2, unbound along C: N's extent lands on T, which the dialect has no
   form for. Rejected by the SHAPE rule, not by the axis rule — C is a perfectly
   legal axis to name. *)
let unbind_c_batch2 () =
  unbind_all "unbind_c_batch2" ~shape:(nhwc ~n:2 ~h:2 ~w:2 ~c:3) Axis.C

(* The motivating ViT node: rank five, unbound at dim 0. Right-aligned, dim 0 of
   a rank-five tensor IS the frame's T axis, so this is refused by the axis rule
   and names T — the actionable diagnostic, rather than a consequence like "some
   tensor has extent on T". *)
let unbind_rank5_t () =
  unbind_all "unbind_rank5_t" ~shape:(s 1 3 1 3 101 32) Axis.T

(* N=1, so selecting C leaves the one surviving slice four-axis -- [Select]'s
   own version of [unbind_c_batch1]. *)
let select_c_batch1 () =
  select_graph "select_c_batch1" ~shape:(nhwc ~n:1 ~h:2 ~w:2 ~c:3) ~axis:Axis.C
    ~index:1 ()

(* Batch 2, selected along N: the outermost axis, so nothing shifts onto T,
   the same reason [unbind_n] converts. *)
let select_n () =
  select_graph "select_n" ~shape:(nhwc ~n:2 ~h:2 ~w:2 ~c:3) ~axis:Axis.N
    ~index:0 ()

(* Batch 2, selected along C: N's extent lands on T, which the dialect has no
   form for -- [Select]'s own version of [unbind_c_batch2]. Rejected by the
   SHAPE rule, not the axis rule -- C is a perfectly legal axis to name. *)
let select_c_batch2 () =
  select_graph "select_c_batch2" ~shape:(nhwc ~n:2 ~h:2 ~w:2 ~c:3) ~axis:Axis.C
    ~index:1 ()

(* The same rank-five/dim-0 shape [unbind_rank5_t] uses, so the two ops' axis
   rejections are directly comparable: rank five, selected at dim 0, which
   right-aligns onto T -- refused by the AXIS rule, same as [Unbind]'s. *)
let select_rank5_t () =
  select_graph "select_rank5_t" ~shape:(s 1 3 1 3 101 32) ~axis:Axis.T ~index:0
    ()

(* Batch 2, split along W: unlike [unbind_c_batch2], KEEPING the axis means N
   never shifts, so batch 2 converts here where it does not there. *)
let split_with_sizes_w_batch2 () =
  split_with_sizes_all "split_with_sizes_w_batch2"
    ~shape:(nhwc ~n:2 ~h:2 ~w:4 ~c:3) Axis.W [ 1; 3 ]

(* The same rank-five/dim-0 shape [unbind_rank5_t] uses, so the two ops'
   axis rejections are directly comparable: rank five, split at dim 0, which
   right-aligns onto T -- refused by the AXIS rule, same as [Unbind]'s. *)
let split_with_sizes_rank5_t () =
  split_with_sizes_all "split_with_sizes_rank5_t" ~shape:(s 1 3 1 3 101 32)
    Axis.T [ 1; 2 ]
