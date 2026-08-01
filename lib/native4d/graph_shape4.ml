(* Shape dispatch for Native4D: the S-independent twin of [Eval_op4], and the
   Native4D counterpart of [Graph_shape].

   Every arm returns a [Shape4.t], which is the acceptance criterion
   .ai/native4d_plan.md sets for this stage — an inferred output that is not
   four-axis cannot escape into the IR. The arms themselves delegate to the
   NATIVE op's [output_shape]: the shape rules are identical, since the physical
   frame is identical (design §4.1), and restating them would be a second
   definition free to drift from the one the compute actually uses.

   So each arm is: unwrap operand signatures to [Vec6.shape], call Native, wrap
   the result back through [Shape4.of_vec6]. The wrap is not ceremony — it is
   where an op whose output leaves the dialect gets caught, which is how
   correction C1 (Mean keepdim=false putting the batch extent on D) is enforced
   without a special case anywhere. *)

open Op

type error =
  [ Shape_error.t | Shape4.error | `Missing_tensor_sig of Tensor_id.t ]

let pp_error fmt : [< error ] -> unit = function
  | #Shape_error.t as e -> Shape_error.pp fmt e
  | #Shape4.error as e -> Shape4.pp_error fmt e
  | `Missing_tensor_sig id -> Fmt.pf fmt "missing tensor sig %a" Tensor_id.pp id

let widen (r : ('a, [< error ]) Core.result) : ('a, error) Core.result =
  (r :> ('a, error) Core.result)

(* Native's window/broadcast rules answer in the six-axis frame; this is the one
   place the answer re-enters the dialect. *)
let four (r : (Vec6.shape, [< error ]) Core.result) =
  let open Core.Syntax in
  let* s = widen r in
  widen (Shape4.of_vec6 s)

(* Grouping is a constructor here, not a parameter, so the Native params these
   build are the only place a [groups] value exists at all. *)
let conv2d_params (p : Ops4.Conv_params.t) ~groups : Conv.Conv2d.params =
  {
    h = p.Ops4.Conv_params.h;
    w = p.Ops4.Conv_params.w;
    in_channels = p.Ops4.Conv_params.in_channels;
    groups = Op_config.Pos.of_int groups;
  }

let transposed_params (p : Ops4.Transposed_conv2d.params) :
    Conv.Convolution.params =
  {
    stride = p.Ops4.Transposed_conv2d.stride;
    padding = p.Ops4.Transposed_conv2d.padding;
    dilation = p.Ops4.Transposed_conv2d.dilation;
    transposed = true;
    output_padding = p.Ops4.Transposed_conv2d.output_padding;
    groups = Op_config.Pos.of_int 1;
  }

(* A four-axis permutation completed to the six-axis bijection Native's rule
   wants: T and D map to themselves, which is exactly what "does not use T or D
   as semantic axes" means once [Axis4.t] has made it unsayable otherwise. *)
let perm6 (perm : Ops4.Permute4.perm) : Permute.Permute.perm =
  Permute.Permute.of_fn (fun axis ->
      match Axis4.of_axis axis with
      | None -> axis (* T and D are fixed *)
      | Some a4 -> Axis4.to_axis (Ops4.Permute4.lookup perm a4))

let mean_params (p : Ops4.Mean_keepdims.params) : Reduce.Mean.params =
  {
    dims = List.map Axis4.to_axis p.Ops4.Mean_keepdims.dims;
    keepdim = true (* the dialect has no other form *);
  }

let rms_params (p : Ops4.Rms_norm.params) : Norm.RmsNorm.params =
  {
    dims = List.map Axis4.to_axis p.Ops4.Rms_norm.dims;
    eps = p.Ops4.Rms_norm.eps;
  }

(* Exhaustive with no default arm, as [Graph_shape] is. *)
let output_shape (op : Op.t)
    ~(sig_of : Tensor_ref.t -> (Tensor_sig.t, error) Core.result) :
    (Shape4.t list, error) Core.result =
  let open Core.Syntax in
  let shape r = sig_of r >>| fun sg -> sg.Tensor_sig.shape in
  let one r = r >>| fun s -> [ s ] in
  match op with
  | Add { Pointwise.Bin.a; b } ->
      let* a_shape = shape a in
      let* b_shape = shape b in
      one (four (Pointwise.Add.output_shape a_shape b_shape))
  | Div { Pointwise.Bin.a; b } ->
      let* a_shape = shape a in
      let* b_shape = shape b in
      one (four (Pointwise.Div.output_shape a_shape b_shape))
  | Mul { Pointwise.Bin.a; b } ->
      let* a_shape = shape a in
      let* b_shape = shape b in
      one (four (Pointwise.Mul.output_shape a_shape b_shape))
  | Sub { Pointwise.Bin.a; b } ->
      let* a_shape = shape a in
      let* b_shape = shape b in
      one (four (Pointwise.Sub.output_shape a_shape b_shape))
  | Add_scalar { Pointwise.Scalar_bin.x; _ } ->
      let* x_shape = shape x in
      one (four (Pointwise.Add_scalar.output_shape x_shape))
  | Div_scalar { Pointwise.Scalar_bin.x; _ } ->
      let* x_shape = shape x in
      one (four (Pointwise.Div_scalar.output_shape x_shape))
  | Clamp { Pointwise.Clamp.params; x } ->
      let* x_shape = shape x in
      one (four (Pointwise.Clamp.output_shape params x_shape))
  | Hardtanh { Pointwise.Hardtanh.x; _ } ->
      let* x_shape = shape x in
      one (four (Pointwise.Hardtanh.output_shape x_shape))
  | Relu { Pointwise.Relu.x } ->
      let* x_shape = shape x in
      one (four (Pointwise.Relu.output_shape x_shape))
  | Sqrt { Pointwise.Sqrt.x } ->
      let* x_shape = shape x in
      one (four (Pointwise.Sqrt.output_shape x_shape))
  | Avg_pool2d { Pool.AvgPool2d.params; x } ->
      let* x_shape = shape x in
      one (four (Pool.AvgPool2d.output_shape ~x_shape params))
  | Max_pool2d { Pool.MaxPool2d.params; x } ->
      let* x_shape = shape x in
      one (four (Pool.MaxPool2d.output_shape ~x_shape params))
  | Conv2d { Ops4.Conv_payload.params; x; weight; _ } ->
      let* x_shape = shape x in
      let* weight_shape = shape weight in
      one
        (four
           (Conv.Conv2d.output_shape ~x_shape ~weight_shape
              (conv2d_params params ~groups:1)))
  | Depthwise_conv2d { Ops4.Conv_payload.params; x; weight; _ } ->
      let* x_shape = shape x in
      let* weight_shape = shape weight in
      (* One group per input channel: the grouping the constructor names. *)
      let groups = Dim.to_int params.Ops4.Conv_params.in_channels in
      one
        (four
           (Conv.Conv2d.output_shape ~x_shape ~weight_shape
              (conv2d_params params ~groups)))
  | Transposed_conv2d { Ops4.Transposed_conv2d.params; x; weight; _ } ->
      let* x_shape = shape x in
      let* weight_shape = shape weight in
      one
        (four
           (Conv.Convolution.output_shape ~x_shape ~weight_shape
              (transposed_params params)))
  | Mean_keepdims { Ops4.Mean_keepdims.params; x } ->
      let* x_shape = shape x in
      one (four (Reduce.Mean.output_shape ~x_shape (mean_params params)))
  | Rms_norm { Ops4.Rms_norm.x; _ } ->
      let* x_shape = shape x in
      one (four (Norm.RmsNorm.output_shape ~x_shape))
  | Permute4 { Ops4.Permute4.perm; x } ->
      let* x_shape = shape x in
      one (four (Permute.Permute.output_shape ~x_shape (perm6 perm)))
  | Reshape4 { Ops4.Reshape4.params; _ } ->
      (* Already a [Shape4.t]: a reshape cannot leave the dialect, which is the
         whole reason the target is typed rather than validated. *)
      Core.return [ params.Ops4.Reshape4.shape ]
