(* Argument-decoding, permutation, and conv2d/pool2d parameter helpers for
   [Op_bridge], split from op_bridge.ml. Internal to the
   per-operation-family dispatch modules (op_bridge_pointwise.ml,
   op_bridge_linalg.ml, op_bridge_conv.ml, op_bridge_pool.ml,
   op_bridge_reduce.ml, op_bridge_norm.ml, op_bridge_attention.ml,
   op_bridge_shape.ml); not part of [Op_bridge]'s own external surface. *)

open Pytorch_types
open Op_bridge_error
module D = Interp_decode

let ( let* ) = Err.Syntax.( let* )
let return = Err.return
let fail = Err.fail
let decode_result r = Err.map_error (fun e -> `Decode e) r
let tensor_arg env node name = decode_result (D.tensor_arg_result env node name)

let tensors_arg env node name =
  decode_result (D.tensors_arg_result env node name)

let int_arg ?default node name =
  decode_result (D.int_arg_result ?default node name)

let ints_arg ?(default = []) node name =
  decode_result (D.ints_arg_result ~default node name)

let floats_arg ?(default = []) node name =
  decode_result (D.floats_arg_result ~default node name)

let bool_arg ?(default = false) node name =
  decode_result (D.bool_arg_result ~default node name)

let string_arg ~default node name =
  decode_result (D.string_arg_result ~default node name)

let packed_shape (Tensor.Tensor r) = r.shape

let packed_metadata (Tensor.Tensor r) =
  let fmt = Payload.Fmt r.payload.Payload.fmt in
  let quant =
    match r.payload.Payload.quant with
    | Payload.No_quant -> None
    | Payload.Quant q -> Some q
  in
  (fmt, quant)

(* Monadically allocate one input edge per packed tensor, left to right.
   The resulting ids match graph.inputs in insertion order. *)
let rec alloc_inputs = function
  | [] -> Graph_builder.return []
  | t :: rest ->
      let open Graph_builder in
      let fmt, quant = packed_metadata t in
      let* id = input ~shape:(packed_shape t) ~fmt ?quant () in
      let+ ids = alloc_inputs rest in
      id :: ids

(* Build a graph from [tensors] (native packed args) and [body] mapping input
   ids to output ids.  Returns the graph and its input bindings [(id, packed)]. *)
let build_g ~name tensors body =
  let* g =
    Graph_builder.build ~name ~outputs:Fun.id
      (let open Graph_builder in
       let* ids = alloc_inputs tensors in
       body ids)
    |> Err.map_error (fun e -> `Build e)
  in
  return (g, List.combine g.Graph_ir.Graph.inputs tensors)

let some_graph = function Ok x -> Some (Ok x) | Error e -> Some (Error e)

(* Convert an ATen tensor to native, prefixing errors with [arg_name]. *)
let native_of_aten arg_name t =
  Tensor_bridge.of_aten t
  |> Err.map_error ~pos:__POS__ (fun cause ->
      `Tensor_bridge { arg_name; cause })

let native_tensor_arg aten_env node name =
  let* tensor = tensor_arg aten_env node name in
  native_of_aten name tensor

let optional_tensor_present node name =
  match D.find_arg node name with
  | Some (Argument.Tensor _)
  | Some (Argument.Optional_tensor (OptionalTensorArgument.Tensor _)) ->
      true
  | _ -> false

let aten_rank t = Array.length (Aten_tensor.shape t)

(* [D.scalar_arg_result]/[scalar_opt_arg_result] hand back an [Aten_scalar.t],
   which also admits a Bool the native scalar domain has no meaning for — so the
   narrowing to float is explicit and checked rather than assumed. *)
let float_of_aten_scalar name = function
  | Aten_scalar.Int i -> Err.return (Int64.to_float i)
  | Aten_scalar.Float f -> Err.return f
  | Aten_scalar.Bool _ ->
      Err.fail (`Validation_failure (name ^ ": expected a numeric scalar"))

let scalar_arg ~default node name =
  let* s = decode_result (D.scalar_arg_result ~default node name) in
  float_of_aten_scalar name s

let scalar_opt_arg node name =
  let* s = decode_result (D.scalar_opt_arg_result node name) in
  match s with
  | None -> return None
  | Some s ->
      let* f = float_of_aten_scalar name s in
      return (Some f)

(* [add.Tensor]/[sub.Tensor] compute [self + alpha * other]. No model in this zoo
   serialises a non-default alpha, so it is unimplemented — but silently dropping
   it would make the bridge claim semantics it does not deliver, and this arm is
   the one an ATen-vs-native comparison runs through. Reject instead. *)
let reject_alpha node =
  let* alpha = scalar_opt_arg node "alpha" in
  match alpha with
  | None -> return ()
  | Some a when Float.equal a 1. -> return ()
  | Some a ->
      fail
        (`Validation_failure
           (Printf.sprintf "alpha=%g is not supported (only 1)" a))

(* A Tensor-typed argument the exporter serialised as a bare scalar — the
   native twin of [Interp_decode.tensor_or_scalar_arg], except that the native
   side routes the scalar into an op parameter instead of materialising it. *)
let tensor_or_scalar aten_env node name =
  match D.find_arg node name with
  | Some (Argument.Int i) -> return (`Scalar (float_of_int i))
  | Some (Argument.Float f) -> return (`Scalar f)
  | _ ->
      let* t = native_tensor_arg aten_env node name in
      return (`Tensor t)

(* --- Permutations for NCHW/NHWC relayout ---
   All are full 6-axis bijections; see .ai/native_aten_bridge_layout.md. *)

(* Right-aligned rank-4 NCHW (D=Nbatch, H=Cch, W=Hsp, C=Wsp) ->
   channel-last in the same outer frame (D=Nbatch, H=Hsp, W=Wsp, C=Cch). *)
let perm_nchw_to_nhwc : Permute.Permute.perm =
  let open Axis in
  [ (N, N); (T, T); (D, D); (H, W); (W, C); (C, H) ]

(* Channel-last in the same outer frame -> right-aligned rank-4 NCHW.
   Inverse of [perm_nchw_to_nhwc]. *)
let perm_nhwc_to_nchw : Permute.Permute.perm =
  let open Axis in
  [ (N, N); (T, T); (D, D); (H, C); (W, H); (C, W) ]

(* OIHW conv weight (D=Cout, H=Cin, W=Kh, C=Kw) ->
   native [N=Cout, H=Kh, W=Kw, C=Cin].  Unlike activations, weights must move
   Cout onto N because Conv2d.output_shape reads output channels from weight N. *)
let perm_oihw_to_conv_weight : Permute.Permute.perm =
  let open Axis in
  [ (N, D); (T, T); (D, N); (H, W); (W, C); (C, H) ]

(* Rank-2 addmm weight [In,Out] (W=In, C=Out) -> native [N=Out, C=In]. *)
let perm_addmm_weight : Permute.Permute.perm =
  let open Axis in
  [ (N, C); (T, T); (D, D); (H, H); (W, N); (C, W) ]

(* Rank-2 linear weight [Out,In] (W=Out, C=In) -> native [N=Out, C=In]. *)
let perm_linear_weight : Permute.Permute.perm =
  let open Axis in
  [ (N, W); (T, T); (D, D); (H, H); (W, N); (C, C) ]

(* --- Arg helpers shared by mean.dim, rms_norm, permute --- *)

(* [mean.dim] and [rms_norm] reference frame axes through [Aten_shape.axis_of_dim],
   so the dims must be derived from the ATen input's RANK (not the right-aligned
   6D shape), keeping reduced axes consistent with where [of_aten] places the
   data. [aten.mean.dim] treats a missing/None/empty dim list as "all dims", so
   the bridge normalizes all three spellings the same way. *)
(* The only approved route from a decoded dim to [Aten_shape.axis_of_dim],
   which asserts its precondition and raises: the range is checked HERE and
   reported as a typed row -- the design record's rule that a rank-sensitive
   rejection gets its own constructor rather than a [`Validation_failure]
   string. The ORIGINAL [dim] is reported, not the normalized one: a user who
   wrote -9 is better served by seeing -9. [op] names the calling arm, since
   that is the only thing distinguishing one caller's fault from another's.

   Only the DIM is judged here. A rank Native cannot hold is a different fault,
   and folding it in would report "invalid dimension 0 for rank 7" — where 0 is
   a perfectly good dimension and the rank is what went wrong. Every caller
   converts the operand first, so [Tensor_bridge]'s own [`Rank_out_of_range]
   has already fired for that case and [rank] is in [0,6] by the time this
   runs; the [rank < 1] arm then covers only a rank-0 operand, for which no dim
   is valid. *)
(* The checked, normalized INT underneath [dim_axis] -- exposed on its own
   because [transpose.int] needs the normalized position itself (to build a
   swap permutation), not the frame axis [dim_axis] converts it to. *)
let norm_dim ~op ~rank dim =
  let d = if dim < 0 then dim + rank else dim in
  if rank < 1 || d < 0 || d >= rank then
    fail (`Invalid_dim { Invalid_dim.op; dim; rank })
  else return d

let dim_axis ~op ~rank dim =
  let* d = norm_dim ~op ~rank dim in
  return (Aten_shape.axis_of_dim ~rank d)

(* [unsqueeze.default]'s [dim] is judged against rank+1 valid positions (the
   OUTPUT rank has one more axis than the operand), unlike every other
   [dim]-taking arm here, which judges against the operand's own rank via
   [norm_dim]. [Invalid_dim.rank] still reports the operand's real rank (its
   documented meaning, see the comment above); only the upper bound of the
   range check is relaxed by one to admit inserting past the last axis. *)
let norm_unsqueeze_dim ~op ~rank dim =
  let d = if dim < 0 then dim + rank + 1 else dim in
  if d < 0 || d > rank then fail (`Invalid_dim { Invalid_dim.op; dim; rank })
  else return d

let dims_arg node ~op ~rank name =
  match D.find_arg node name with
  | Some (Argument.Ints []) | Some (Argument.None _) | None ->
      return (Aten_shape.used_axes ~rank)
  | Some (Argument.Ints xs) -> Err.List.map (dim_axis ~op ~rank) xs
  | _ -> return (Aten_shape.used_axes ~rank)

(* Arithmetic arms materialize f32 outputs, so an i64 operand would silently
   become f32 and then fail the verifier's dtype pairing. [unbind] is excluded:
   it is a storage-preserving view operation and retains its input format. Read
   from the ATen tensor before conversion, so an error names the source dtype. *)
(* [Tensor_bridge.of_aten] right-aligns an ATen shape into the six-axis frame, so
   a bias declared [1,Cout] arrives indistinguishable from [Cout] and the shared
   [Graph_shape] check passes it -- while ATen refuses a bias that is not 1-D.
   The rank survives only on the ATen tensor, so it is read there. *)
let require_rank arg_name ~expected t =
  let got = Array.length (Aten_tensor.shape t) in
  if got = expected then return ()
  else fail (`Operand_rank { Operand_rank.arg_name; expected; got })

let require_f32 arg_name t =
  match Aten_tensor.scalar_type t with
  | Aten_scalar_type.Float -> return ()
  | dtype ->
      fail
        (`Unsupported_input_dtype { Unsupported_input_dtype.arg_name; dtype })

let trailing_axes ~rank ~k =
  let all = Aten_shape.used_axes ~rank in
  List.filteri (fun i _ -> i >= rank - k) all

(* [k <= rank] is checked BEFORE [trailing_axes], which has no guard of its own:
   [List.filteri] with a negative lower bound keeps every element, so an
   over-long normalized_shape silently normalized over the whole tensor. And the
   EXTENTS are compared, not just the count -- that is the check whose absence
   made a wrong shape a wrong answer rather than an error. *)
let normalized_dims ~op ~(x_shape : int array) ~normalized_shape =
  let rank = Array.length x_shape in
  let k = List.length normalized_shape in
  let* () =
    if k < 1 || k > rank then
      fail (`Normalized_rank { Normalized_rank.op; rank; got = k })
    else return ()
  in
  let expected = Array.to_list (Array.sub x_shape (rank - k) k) in
  let* () =
    if expected <> normalized_shape then
      fail
        (`Normalized_shape
           { Normalized_shape.op; expected; got = normalized_shape })
    else return ()
  in
  return (trailing_axes ~rank ~k)

(* rms_norm's [eps] is a [float?]: a float, an explicit none, or omitted. The
   catch-all this replaces read a bool, int, string or tensor `eps` as the
   default too -- so a malformed node the serialized importer refuses was
   accepted here, which is precisely the acceptance divergence Group 2 exists to
   close. *)
(* rms_norm's [eps] is a [float?]: a float, an explicit none, or omitted. The
   catch-all this replaces read a bool, int, string or tensor `eps` as the
   default too -- so a malformed node the serialized importer refuses was
   accepted here, which is precisely the acceptance divergence Group 2 exists to
   close. [`Float_opt] rather than [`Float], because a none IS accepted. *)
(* A REQUIRED [float]: [_native_batch_norm_legit_no_training]'s eps, whose schema
   has no [?]. Kept apart from [eps_arg] below because the two answer different
   questions, and one decoder answering both is how a null epsilon comes to be
   read as zero. *)
let float_arg ?default node name =
  decode_result (D.float_arg_result ?default node name)

(* A [float?] whose absence is genuinely "no value", not a default: [aten.pad]'s
   [value] means 0.0 in constant mode and must be ABSENT (or zero) in reflect,
   so the two cases have to stay distinguishable here rather than being
   collapsed by a decoder default. Contrast [eps_arg], which supplies one. *)
let float_opt_arg node name =
  match D.find_arg node name with
  | Some (Argument.Float f) -> return (Some f)
  | Some (Argument.Int i) -> return (Some (float_of_int i))
  | None | Some (Argument.None _) -> return None
  | Some a -> decode_result (D.wrong_kind name `Float_opt a)

(* Shared with the generated ATen dispatch and with [Native_interp] through
   [Interp_decode], so the [Sym_int] policy -- a resolved [Int n] is accepted, a
   [Name _] is refused as an unresolved symbol -- has ONE implementation rather
   than three that happen to agree. *)
let int_opt_arg node name = decode_result (D.int_opt_arg_result node name)

let eps_arg node name =
  match D.find_arg node name with
  | Some (Argument.Float f) -> return f
  | None | Some (Argument.None _) -> return Norm.RmsNorm.default_eps
  | Some a -> decode_result (D.wrong_kind name `Float_opt a)

(* Build a full 6D native permutation from an ATen [dims] list and the tensor
   rank.  For the [rank] used axes, [dims.(i)] is the ATen input dim for output
   position [i]; outer padding axes are the identity.

   [dims] must have exactly [rank] entries: a short or long list is caught
   downstream today ([Permute.output_shape] refuses the resulting
   non-bijection), so this check is a diagnostic improvement, not a hole it
   closes. *)
let native_perm_of_aten ~op ~rank dims =
  let* () =
    let got = List.length dims in
    if got <> rank then fail (`Dims_count { Dims_count.op; rank; got })
    else return ()
  in
  let used = Aten_shape.used_axes ~rank in
  let outer = List.filter (fun a -> not (List.mem a used)) Axis.all in
  let outer_perm = List.map (fun a -> (a, a)) outer in
  let* inner_perm =
    Err.List.map
      (fun (i, d) ->
        let* in_axis = dim_axis ~op ~rank d in
        return (Aten_shape.axis_of_dim ~rank i, in_axis))
      (List.mapi (fun i d -> (i, d)) dims)
  in
  return (outer_perm @ inner_perm)

(* --- Param helpers for conv2d / pool2d --- *)

(* Validate a 2-element int list as [h; w].  A single-element list is accepted
   as [v; v] (symmetric). *)
let hw2 name = function
  | [ h; w ] -> Err.return (h, w)
  | [ v ] -> Err.return (v, v)
  | values -> Err.fail (`Invalid_hw_arg { name; values })

(* Construct Conv2d.params from the ATen weight shape array
   (rank-4: [Cout,Cin/groups,Kh,Kw]) and validated config ints.
   Raises [Invalid_argument] on bad dims. *)
(* Every raw value is validated BEFORE it reaches an asserting constructor, and
   the fault it produces is [Op_config.Bad.t] -- the same row [Native_interp]
   reports for the same node. Containing the assertions in the arm's exception
   boundary is not enough on its own: it yields a [`Validation_failure] STRING,
   which a caller cannot classify without parsing prose, and which says nothing
   about which parameter of which op held what value. *)
let cfg = function Ok v -> return v | Error e -> fail (`Bad_config e)
let pos ~op ~param n = cfg (Op_config.Bad.pos ~op ~param n)
let nonneg ~op ~param n = cfg (Op_config.Bad.nonneg ~op ~param n)

let extent ~op ~param n =
  match Dim.extent_checked n with
  | Ok e -> return e
  | Error _ -> cfg (Error { Op_config.Bad.op; param; fault = `Not_positive n })

let conv_axis_window ~op ~kernel ~stride ~pad ~dilation :
    (Conv.Conv2d.axis_window, [> `Bad_config of Op_config.Bad.t ]) Err.t =
  let* kernel = extent ~op ~param:`Kernel_size kernel in
  let* stride = pos ~op ~param:`Stride stride in
  let* pad = nonneg ~op ~param:`Padding pad in
  let* dilation = pos ~op ~param:`Dilation dilation in
  return
    { Conv.Conv2d.kernel; stride; pad_before = pad; pad_after = pad; dilation }

(* [in_channels] is the weight's per-group input extent TIMES the group count,
   and both factors are model-supplied. Computed in [int] it could wrap before
   reaching [Dim.extent]'s assertion, which would then escape as an
   [Invalid_argument] from inside the [try] below -- reported as a
   [`Validation_failure] string rather than the typed row the other two
   definitions of this rule return. Bounded through the same helper
   [Window_axis] and [Native_interp] use, so all three agree. *)
let make_conv2d_params ~op w_shape sh sw ph pw dh dw groups =
  let* h =
    conv_axis_window ~op ~kernel:w_shape.(2) ~stride:sh ~pad:ph ~dilation:dh
  in
  let* w =
    conv_axis_window ~op ~kernel:w_shape.(3) ~stride:sw ~pad:pw ~dilation:dw
  in
  let* groups = pos ~op ~param:`Groups groups in
  let* in_channels =
    (* [Graph_shape.error] flat-includes [Shape_error.t], and
       [Graph_builder.error] flat-includes that, so the row crosses into this
       module's [`Build] seam unchanged rather than re-labelled. *)
    let of_shape r =
      Err.map_error
        (fun (e : Shape_error.t) -> `Build (e :> Graph_builder.error))
        r
    in
    let* c = of_shape (Window_axis.factor ~what:`In_channels w_shape.(1)) in
    let* g = of_shape (Window_axis.factor ~what:`In_channels (groups :> int)) in
    let product = Int64.mul c g in
    if product >= Window_axis.limit then
      of_shape
        (Err.fail
           (`Window_over_limit
              Shape_error.Window_over_limit.
                {
                  what = `In_channels;
                  value = product;
                  limit = Window_axis.limit;
                }))
    else return (Dim.extent (Int64.to_int product))
  in
  return { Conv.Conv2d.h; w; in_channels; groups }

let make_conv2d_padding_params ~op sh sw padding dh dw groups =
  (* NOT [padding_of_string], which [invalid_arg]s: this string is model data.
    Containing that in the arm's [try] turned it into a [`Validation_failure]
    string, where [Native_interp] returns the mode itself in a typed row.
    [Conv2d_padding.of_string] is the shared checked parser, so the accepted set
    cannot drift between the two importers. *)
  let* padding =
    match Conv.Conv2d_padding.of_string padding with
    | Ok p -> return p
    | Error s -> fail (`Unsupported_padding_mode s)
  in
  let* sh = pos ~op ~param:`Stride sh in
  let* sw = pos ~op ~param:`Stride sw in
  let* dh = pos ~op ~param:`Dilation dh in
  let* dw = pos ~op ~param:`Dilation dw in
  let* groups = pos ~op ~param:`Groups groups in
  return
    {
      Conv.Conv2d_padding.stride = { h = sh; w = sw };
      padding;
      dilation = { h = dh; w = dw };
      groups;
    }

let make_convolution_params ~op sh sw ph pw dh dw transposed oph opw groups =
  let* sh = pos ~op ~param:`Stride sh in
  let* sw = pos ~op ~param:`Stride sw in
  let* ph = nonneg ~op ~param:`Padding ph in
  let* pw = nonneg ~op ~param:`Padding pw in
  let* dh = pos ~op ~param:`Dilation dh in
  let* dw = pos ~op ~param:`Dilation dw in
  (* [`Output_padding], not [`Padding]. The comment this replaces claimed the op
     name made the two unambiguous -- but `convolution.default` carries BOTH
     arguments, so it never did, and the same malformed node produced a
     different payload here than through the serialized importer. *)
  let* oph = nonneg ~op ~param:`Output_padding oph in
  let* opw = nonneg ~op ~param:`Output_padding opw in
  let* groups = pos ~op ~param:`Groups groups in
  return
    {
      Conv.Convolution.stride = { h = sh; w = sw };
      padding = { h = ph; w = pw };
      dilation = { h = dh; w = dw };
      transposed;
      output_padding = { h = oph; w = opw };
      groups;
    }

let make_pool_params ~op ~ceil_mode kh kw sh sw ph pw =
  let* kh = extent ~op ~param:`Kernel_size kh in
  let* kw = extent ~op ~param:`Kernel_size kw in
  let* sh = pos ~op ~param:`Stride sh in
  let* sw = pos ~op ~param:`Stride sw in
  let* ph = nonneg ~op ~param:`Padding ph in
  let* pw = nonneg ~op ~param:`Padding pw in
  return
    {
      Pool.MaxPool2d.ceil_mode;
      kernel = { h = kh; w = kw };
      stride = { h = sh; w = sw };
      pad = { h = ph; w = pw };
    }

(* Pool stride defaults to kernel_size when absent (PyTorch convention). *)
let pool_stride kernel_size node =
  let* stride = ints_arg ~default:[] node "stride" in
  return (match stride with [] -> kernel_size | s -> s)

(* [Pool.MaxPool2d.params] has no field for [dilation], so a non-default value
   is refused rather than approximated -- the native params have nowhere to
   put it. [ceil_mode] IS represented (see [Pool.MaxPool2d.params.ceil_mode]),
   so it is decoded and returned rather than rejected. *)
let pool_dilation_and_ceil_mode node =
  let* dilation = ints_arg ~default:[ 1; 1 ] node "dilation" in
  (* Normalized through [hw2] first, so a wrong ARITY keeps the typed arity
     diagnostic every other H/W argument gets. Testing "does some element differ
     from 1" accepted [] and [1;1;1], which ATen itself refuses. *)
  let* h, w = hw2 "dilation" dilation in
  let* () =
    if h <> 1 || w <> 1 then
      fail
        (`Pool_unsupported
           {
             Pool_unsupported.op = node.Node.target;
             option = Pool_unsupported.Dilation dilation;
           })
    else return ()
  in
  bool_arg ~default:false node "ceil_mode"
