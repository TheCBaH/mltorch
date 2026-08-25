(* Native-side dispatch: given a graph node and the ATen environment (inputs),
   build a native Graph_ir.graph encoding the equivalent computation.

   Returns [None] if no native implementation exists for this op.  Returns
   [Some (Error e)] if the op is mapped but argument conversion or param
   validation fails.  Returns [Some (Ok (g, bindings))] where [g] is the native
   graph and [bindings] maps each graph input id to its converted native tensor.

   Ops requiring NCHW<->NHWC relayout (conv2d, max_pool2d, linear/addmm) produce
   a graph named "<op>_relayout" that wraps the core op in permute nodes.  All
   other ops produce a flat single-op graph.  Unmapped ops return None. *)

open Pytorch_types
module D = Interp_decode

type aten_env = Interp_decode.env
type invalid_hw_arg = { name : string; values : int list }

(* [cause] is the bridge's OWN row, not a rendering of it: [Tensor_bridge]
   used to hand back a string and this wrapped it, so a caller could read what
   went wrong but never branch on it. *)
type tensor_bridge_error = { arg_name : string; cause : Tensor_bridge.error }

(* Own modules with the payload named [t], per CLAUDE.md's record convention.
   The top-level records above predate it and are not a pattern to extend. *)

(* Shared by every arm that resolves a decoded dim through
   [Aten_shape.axis_of_dim]: [op] is what distinguishes one arm's fault from
   another's (unbind.int, mean.dim, permute.default, transpose.int, ...),
   [dim] is the ORIGINAL value as decoded (not normalized), and [rank] is the
   operand's rank. *)
module Invalid_dim = struct
  type t = { op : string; dim : int; rank : int }
end

(* [permute.default]/[transpose.int] build a full permutation from a decoded
   dims list; a list whose length disagrees with the operand's rank is a
   distinct fault from any single dim being out of range. *)
module Dims_count = struct
  type t = { op : string; rank : int; got : int }
end

(* [cat.default]/[stack.default]: ATen requires every tensor in the list to
   share one rank (there is no broadcasting between list entries), checked
   here rather than left to [Concat]'s own shape rule -- that rule compares
   FRAME axes, and two different ranks land their data on different frame
   axes under [Aten_shape.of_aten]'s right-alignment, so a rank disagreement
   would surface there as a confusing off-axis extent mismatch instead of
   the rank fault it actually is. *)
module Concat_rank_mismatch = struct
  type t = { op : string; first : int; other : int }
end

(* Carries ATen's OWN dtype, read off the source tensor before conversion,
   rather than a Native [Payload.fmt]. Two reasons: [Payload.fmt] is a
   three-parameter GADT and cannot be a bare field type (its existential is
   [Payload.packed_fmt]), and translating a dtype into a Native format purely to
   report it would be a conversion inside an error. *)
module Unsupported_input_dtype = struct
  type t = { arg_name : string; dtype : Aten_scalar_type.t }
end

(* The two pooling options [Pool.MaxPool2d.params] has no field for. Own module
   because the row carries the op alongside the value, and [op] would collide
   with the other records here. A closed pair rather than one string tag: the
   dilation has a value worth reporting and [ceil_mode] does not. *)
(* The two [normalized_shape] faults, shared by every normalisation. The arm
   read the LENGTH of [normalized_shape] and nothing else, so a shape naming
   extents the input does not have normalized over the wrong axes and returned a
   plausible wrong answer; and [trailing_axes] silently returns the whole axis
   list when [k > rank], so an over-long shape normalized over everything.

   [op] names which normalisation, because three targets now share these rows
   and the message used to say "rms_norm" for all of them. *)
module Normalized_rank = struct
  type t = { op : Norm.Target.t; rank : int; got : int }
end

module Normalized_shape = struct
  type t = { op : Norm.Target.t; expected : int list; got : int list }
end

(* An optional operand's declared RANK, which right-alignment into the six-axis
   frame erases before any shared shape rule can see it. *)
module Operand_rank = struct
  type t = { arg_name : string; expected : int; got : int }
end

module Pool_unsupported = struct
  type option = Dilation of int list | Ceil_mode
  type t = { op : string; option : option }
end

module Adaptive_pool_rank = struct
  type t = { got : int }
end

(* sdpa's typed rejection boundary is [Attention.Sdpa.Reject], shared with
   [Native_interp] (op8-impl.md commit 3) for [Op_config.Bad]'s reason: the
   two importers must reject the same values. *)

type error =
  [ `Decode of Interp_decode.error
  | `Tensor_bridge of tensor_bridge_error
  | `Build of Graph_builder.error
  | `Invalid_hw_arg of invalid_hw_arg
  | `Validation_failure of string
  | `Invalid_dim of Invalid_dim.t
  | `Dims_count of Dims_count.t
  | `Unsupported_input_dtype of Unsupported_input_dtype.t
  | `Addmm_invalid_weight_rank of int array
  | `Conv2d_invalid_weight_rank of int array
  | `Conv2d_padding_invalid_weight_rank of int array
  | `Convolution_invalid_weight_rank of int array
  | `Linear_invalid_weight_rank of int array
  | `Pool_unsupported of Pool_unsupported.t
  | `Adaptive_pool_rank of Adaptive_pool_rank.t
  | `Normalized_rank of Normalized_rank.t
  | `Normalized_shape of Normalized_shape.t
  | `Operand_rank of Operand_rank.t
  | `Bad_config of Op_config.Bad.t
  | `Unsupported_padding_mode of string
  | `Aten_shape of Aten_shape.error
  | `Bad_pad_list of Pad.Pad.Bad_pad_list.t
  | `Sdpa_reject of Attention.Sdpa.Reject.t
  | `Concat_no_tensors of string
  | `Concat_rank_mismatch of Concat_rank_mismatch.t ]

(* Deliberately not [Fmt.brackets], which boxes its content and so may
   line-wrap; the original bare "[%s]" (String.concat) never did, regardless
   of width. *)
let pp_int_list ppf xs =
  Fmt.pf ppf "[%a]" (Fmt.list ~sep:(Fmt.any ", ") Fmt.int) xs

let pp_int_array ppf xs = pp_int_list ppf (Array.to_list xs)

let pp_error ppf : [< error ] -> unit = function
  | `Decode e -> Interp_decode.pp_error ppf e
  | `Tensor_bridge { arg_name; cause } ->
      Fmt.pf ppf "%s: %a" arg_name Tensor_bridge.pp_error cause
  | `Build e -> Graph_builder.pp_error ppf e
  | `Invalid_hw_arg { name; values } ->
      Fmt.pf ppf "%s: expected [h; w] or [v], got %a" name pp_int_list values
  | `Validation_failure msg -> Fmt.string ppf msg
  | `Invalid_dim { Invalid_dim.op; dim; rank } ->
      Fmt.pf ppf "%s: invalid dimension %d for rank %d" op dim rank
  | `Dims_count { Dims_count.op; rank; got } ->
      Fmt.pf ppf "%s: expected %d dims, got %d" op rank got
  | `Bad_config e -> Op_config.Bad.pp ppf e
  | `Unsupported_padding_mode s ->
      Fmt.pf ppf "padding mode %S is neither \"valid\" nor \"same\"" s
  | `Operand_rank { Operand_rank.arg_name; expected; got } ->
      Fmt.pf ppf "%s must be rank-%d, got rank-%d" arg_name expected got
  | `Normalized_rank { Normalized_rank.op; rank; got } ->
      Fmt.pf ppf
        "%a: normalized_shape has %d entries, outside [1, %d] for this rank"
        Norm.Target.pp op got rank
  | `Normalized_shape { Normalized_shape.op; expected; got } ->
      Fmt.pf ppf
        "%a: normalized_shape %a does not match the input's trailing extents %a"
        Norm.Target.pp op pp_int_list got pp_int_list expected
  | `Pool_unsupported { Pool_unsupported.op; option } -> (
      match option with
      | Pool_unsupported.Dilation d ->
          Fmt.pf ppf "%s: dilation=%a is not supported (only 1)" op pp_int_list
            d
      | Pool_unsupported.Ceil_mode ->
          Fmt.pf ppf "%s: ceil_mode=true is not supported" op)
  | `Adaptive_pool_rank { Adaptive_pool_rank.got } ->
      Fmt.pf ppf
        "adaptive_avg_pool2d input must be rank-3 (CHW) or rank-4 (NCHW), got \
         rank-%d"
        got
  | `Unsupported_input_dtype { Unsupported_input_dtype.arg_name; dtype } ->
      Fmt.pf ppf "%s: the native engine computes in f32, got %s" arg_name
        (Aten_scalar_type.to_string dtype)
  | `Addmm_invalid_weight_rank shape ->
      Fmt.pf ppf "addmm: mat2 must be rank-2, got shape %a" pp_int_array shape
  | `Conv2d_invalid_weight_rank shape ->
      Fmt.pf ppf "conv2d: weight must be rank-4, got shape %a" pp_int_array
        shape
  | `Conv2d_padding_invalid_weight_rank shape ->
      Fmt.pf ppf "conv2d.padding: weight must be rank-4, got shape %a"
        pp_int_array shape
  | `Convolution_invalid_weight_rank shape ->
      Fmt.pf ppf "convolution: weight must be rank-4, got shape %a" pp_int_array
        shape
  | `Linear_invalid_weight_rank shape ->
      Fmt.pf ppf "linear: weight must be rank-2, got shape %a" pp_int_array
        shape
  | `Aten_shape e -> Aten_shape.pp_error ppf e
  | `Bad_pad_list e -> Pad.Pad.Bad_pad_list.pp ppf e
  | `Sdpa_reject e -> Attention.Sdpa.Reject.pp ppf e
  | `Concat_no_tensors op -> Fmt.pf ppf "%s: at least one tensor is required" op
  | `Concat_rank_mismatch { Concat_rank_mismatch.op; first; other } ->
      Fmt.pf ppf "%s: every tensor must have the same rank: %d vs %d" op first
        other

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

let make_pool_params ~op kh kw sh sw ph pw =
  let* kh = extent ~op ~param:`Kernel_size kh in
  let* kw = extent ~op ~param:`Kernel_size kw in
  let* sh = pos ~op ~param:`Stride sh in
  let* sw = pos ~op ~param:`Stride sw in
  let* ph = nonneg ~op ~param:`Padding ph in
  let* pw = nonneg ~op ~param:`Padding pw in
  return
    {
      Pool.MaxPool2d.kernel = { h = kh; w = kw };
      stride = { h = sh; w = sw };
      pad = { h = ph; w = pw };
    }

(* Pool stride defaults to kernel_size when absent (PyTorch convention). *)
let pool_stride kernel_size node =
  let* stride = ints_arg ~default:[] node "stride" in
  return (match stride with [] -> kernel_size | s -> s)

(* [Pool.MaxPool2d.params] carries neither, so BOTH pooling arms decoded
   kernel/stride/padding and left these two unread -- silently computing a
   different op under the right name for any export that set them. Refused here
   rather than approximated: the native params have nowhere to put them. *)
let reject_pool_extras node =
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
  let* ceil_mode = bool_arg ~default:false node "ceil_mode" in
  if ceil_mode then
    fail
      (`Pool_unsupported
         {
           Pool_unsupported.op = node.Node.target;
           option = Pool_unsupported.Ceil_mode;
         })
  else return ()

(* --- Op dispatch --- *)

let dispatch ~(aten_env : aten_env) (node : Node.t) :
    (Graph_ir.graph * (Graph_ir.Tensor_id.t * Tensor.packed) list, error) Err.t
    option =
  (* Arms in global alphabetical order by the dispatched op name. *)
  match node.target with
  | "torch.ops.aten._native_batch_norm_legit_no_training.default" ->
      Some
        ((* Inference batch norm: only out0 (the normalised activations) is
            represented. ATen also returns save_mean/save_invstd, but in eval mode
            those are recorded size-[0] in the exported graph and are dropped
            (the engine has no empty tensors). See
            .ai/native_multi_output_design.md. *)
         let* aten_x = tensor_arg aten_env node "input" in
         let* x = native_of_aten "input" aten_x in
         let* aten_rm = tensor_arg aten_env node "running_mean" in
         let* rm = native_of_aten "running_mean" aten_rm in
         let* aten_rv = tensor_arg aten_env node "running_var" in
         let* rv = native_of_aten "running_var" aten_rv in
         let* eps = float_arg node "eps" in
         let (Tensor.Tensor rm_r) = rm in
         (* weight/bias are optional (ATen `Tensor?`); materialise the identity
            (ones / zeros) [C] vector when absent, as [rms_norm] does. *)
         let* weight =
           if optional_tensor_present node "weight" then
             let* w = tensor_arg aten_env node "weight" in
             native_of_aten "weight" w
           else return (Tensor.materialize rm_r.shape (fun _ -> 1.))
         in
         let* bias =
           if optional_tensor_present node "bias" then
             let* b = tensor_arg aten_env node "bias" in
             native_of_aten "bias" b
           else return (Tensor.materialize rm_r.shape (fun _ -> 0.))
         in
         let params = { Norm.BatchNorm.channel = Axis.C; eps } in
         build_g ~name:"batch_norm_relayout" [ x; weight; bias; rm; rv ]
           (function
           | [ x_id; w_id; b_id; rm_id; rv_id ] ->
               let open Graph_builder in
               let* x' = permute perm_nchw_to_nhwc x_id in
               let* y' =
                 batch_norm params ~x:x' ~weight:w_id ~bias:b_id
                   ~running_mean:rm_id ~running_var:rv_id ()
               in
               let+ y = permute perm_nhwc_to_nchw y' in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.add.Tensor" | "torch.ops.aten.add_.Tensor" ->
      Some
        (let* () = reject_alpha node in
         let* a = native_tensor_arg aten_env node "self" in
         let* other = tensor_or_scalar aten_env node "other" in
         match other with
         | `Tensor b ->
             build_g ~name:"add" [ a; b ] (function
               | [ a_id; b_id ] ->
                   let open Graph_builder in
                   let+ y = add a_id b_id in
                   [ y ]
               | _ -> assert false)
         | `Scalar scalar ->
             build_g ~name:"add_scalar" [ a ] (function
               | [ a_id ] ->
                   let open Graph_builder in
                   let+ y = add_scalar scalar a_id in
                   [ y ]
               | _ -> assert false))
  | "torch.ops.aten.addmm.default" ->
      Some
        ((* addmm(bias, mat1, mat2) = bias + mat1 @ mat2; alpha=beta=1 assumed. *)
         let* aten_bias = tensor_arg aten_env node "self" in
         let* aten_x = tensor_arg aten_env node "mat1" in
         let* aten_w = tensor_arg aten_env node "mat2" in
         let w_shape = Aten_tensor.shape aten_w in
         if Array.length w_shape <> 2 then
           fail (`Addmm_invalid_weight_rank w_shape)
         else
           let* bias = native_of_aten "self" aten_bias in
           let* x = native_of_aten "mat1" aten_x in
           let* w = native_of_aten "mat2" aten_w in
           try
             let params =
               { Linear.Linear.in_features = Dim.extent w_shape.(0) }
             in
             build_g ~name:"addmm_relayout" [ bias; x; w ] (function
               | [ bias_id; x_id; w_id ] ->
                   let open Graph_builder in
                   let* w' = permute perm_addmm_weight w_id in
                   let+ y = linear params ~x:x_id ~weight:w' ~bias:bias_id () in
                   [ y ]
               | _ -> assert false)
           with Invalid_argument msg -> fail (`Validation_failure msg))
  | "torch.ops.aten.bmm.default" -> (
      match
        ( native_tensor_arg aten_env node "self",
          native_tensor_arg aten_env node "mat2" )
      with
      | Error e, _ | _, Error e -> Some (Error e)
      | Ok a, Ok b ->
          build_g ~name:"bmm" [ a; b ] (function
            | [ a_id; b_id ] ->
                let open Graph_builder in
                let+ y = bmm a_id b_id in
                [ y ]
            | _ -> assert false)
          |> some_graph)
  | "torch.ops.aten.clamp.default" | "torch.ops.aten.clamp_.default" ->
      Some
        (let* x = native_tensor_arg aten_env node "self" in
         let* min = scalar_opt_arg node "min" in
         let* max = scalar_opt_arg node "max" in
         (* The both-absent pair is rejected by [Graph_builder] via
            [Clamp.output_shape], the same place ATen's meta function rejects
            it, so no check is needed here. *)
         build_g ~name:"clamp" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = clamp { Pointwise.Clamp.min; max } x_id in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.clone.default" ->
      Some
        (let* x = native_tensor_arg aten_env node "self" in
         (* A requested memory_format is a layout change this op does not
            perform; rejecting beats silently ignoring it. *)
         match D.find_arg node "memory_format" with
         | Some (Argument.None _) | None ->
             build_g ~name:"clone" [ x ] (function
               | [ x_id ] ->
                   let open Graph_builder in
                   let+ y = clone x_id in
                   [ y ]
               | _ -> assert false)
         | Some _ ->
             fail (`Validation_failure "clone: memory_format is not supported"))
  | "torch.ops.aten.conv2d.default" ->
      Some
        (let* groups = int_arg ~default:1 node "groups" in
         let* dilation = ints_arg ~default:[ 1; 1 ] node "dilation" in
         let* dh, dw = hw2 "dilation" dilation in
         let* aten_x = tensor_arg aten_env node "input" in
         let* aten_w = tensor_arg aten_env node "weight" in
         let w_shape = Aten_tensor.shape aten_w in
         if Array.length w_shape <> 4 then
           fail (`Conv2d_invalid_weight_rank w_shape)
         else
           let* stride = ints_arg ~default:[ 1; 1 ] node "stride" in
           let* padding = ints_arg ~default:[ 0; 0 ] node "padding" in
           let* sh, sw = hw2 "stride" stride in
           let* ph, pw = hw2 "padding" padding in
           let* bias_opt =
             if optional_tensor_present node "bias" then
               let* bias = tensor_arg aten_env node "bias" in
               let* () = require_rank "bias" ~expected:1 bias in
               let* bias = native_of_aten "bias" bias in
               return (Some bias)
             else return None
           in
           let* x = native_of_aten "input" aten_x in
           let* w = native_of_aten "weight" aten_w in
           (* INSIDE the [try], with the rest of the parameter construction.
              Moving it out was justified by a mistake: a [try] does not
              interfere with the typed [Error] this returns, because that is a
              VALUE and [let*] short-circuits on it without raising. What the
              [try] catches is the other half -- [Dim.extent],
              [Op_config.Pos.of_int] and [Nonneg.of_int] assert their
              preconditions, and [Window_axis.factor] bounds magnitude only, so
              an ordinary [groups=0] or [stride=[0,1]] still reaches an
              assertion. Outside the boundary those escaped [dispatch] as an
              uncaught [Invalid_argument]. *)
           try
             let* params =
               make_conv2d_params ~op:node.Node.target w_shape sh sw ph pw dh dw
                 groups
             in
             let tensors = [ x; w ] @ Option.to_list bias_opt in
             build_g ~name:"conv2d_relayout" tensors (function
               | [ x_id; w_id ] ->
                   let open Graph_builder in
                   let* x' = permute perm_nchw_to_nhwc x_id in
                   let* w' = permute perm_oihw_to_conv_weight w_id in
                   let* y' = conv2d params ~x:x' ~weight:w' () in
                   let+ y = permute perm_nhwc_to_nchw y' in
                   [ y ]
               | [ x_id; w_id; b_id ] ->
                   let open Graph_builder in
                   let* x' = permute perm_nchw_to_nhwc x_id in
                   let* w' = permute perm_oihw_to_conv_weight w_id in
                   let* y' = conv2d params ~x:x' ~weight:w' ~bias:b_id () in
                   let+ y = permute perm_nhwc_to_nchw y' in
                   [ y ]
               | _ -> assert false)
           with Invalid_argument msg -> fail (`Validation_failure msg))
  | "torch.ops.aten.conv2d.padding" ->
      Some
        (let* groups = int_arg ~default:1 node "groups" in
         let* dilation = ints_arg ~default:[ 1; 1 ] node "dilation" in
         let* dh, dw = hw2 "dilation" dilation in
         let* aten_x = tensor_arg aten_env node "input" in
         let* aten_w = tensor_arg aten_env node "weight" in
         let w_shape = Aten_tensor.shape aten_w in
         if Array.length w_shape <> 4 then
           fail (`Conv2d_padding_invalid_weight_rank w_shape)
         else
           let* stride = ints_arg ~default:[ 1; 1 ] node "stride" in
           let* padding = string_arg ~default:"valid" node "padding" in
           let* sh, sw = hw2 "stride" stride in
           let* bias_opt =
             if optional_tensor_present node "bias" then
               let* bias = tensor_arg aten_env node "bias" in
               let* () = require_rank "bias" ~expected:1 bias in
               let* bias = native_of_aten "bias" bias in
               return (Some bias)
             else return None
           in
           let* x = native_of_aten "input" aten_x in
           let* w = native_of_aten "weight" aten_w in
           try
             let* params =
               make_conv2d_padding_params ~op:node.Node.target sh sw padding dh
                 dw groups
             in
             let tensors = [ x; w ] @ Option.to_list bias_opt in
             build_g ~name:"conv2d_padding_relayout" tensors (function
               | [ x_id; w_id ] ->
                   let open Graph_builder in
                   let* x' = permute perm_nchw_to_nhwc x_id in
                   let* w' = permute perm_oihw_to_conv_weight w_id in
                   let* y' = conv2d_padding params ~x:x' ~weight:w' () in
                   let+ y = permute perm_nhwc_to_nchw y' in
                   [ y ]
               | [ x_id; w_id; b_id ] ->
                   let open Graph_builder in
                   let* x' = permute perm_nchw_to_nhwc x_id in
                   let* w' = permute perm_oihw_to_conv_weight w_id in
                   let* y' =
                     conv2d_padding params ~x:x' ~weight:w' ~bias:b_id ()
                   in
                   let+ y = permute perm_nhwc_to_nchw y' in
                   [ y ]
               | _ -> assert false)
           with Invalid_argument msg -> fail (`Validation_failure msg))
  | "torch.ops.aten.convolution.default" ->
      Some
        (let* transposed = bool_arg node "transposed" in
         let* groups = int_arg ~default:1 node "groups" in
         let* dilation = ints_arg ~default:[ 1; 1 ] node "dilation" in
         let* dh, dw = hw2 "dilation" dilation in
         let* aten_x = tensor_arg aten_env node "input" in
         let* aten_w = tensor_arg aten_env node "weight" in
         let w_shape = Aten_tensor.shape aten_w in
         if Array.length w_shape <> 4 then
           fail (`Convolution_invalid_weight_rank w_shape)
         else
           let* stride = ints_arg ~default:[ 1; 1 ] node "stride" in
           let* padding = ints_arg ~default:[ 0; 0 ] node "padding" in
           let* output_padding =
             ints_arg ~default:[ 0; 0 ] node "output_padding"
           in
           let* sh, sw = hw2 "stride" stride in
           let* ph, pw = hw2 "padding" padding in
           let* oph, opw = hw2 "output_padding" output_padding in
           let* bias_opt =
             if optional_tensor_present node "bias" then
               let* bias = tensor_arg aten_env node "bias" in
               let* () = require_rank "bias" ~expected:1 bias in
               let* bias = native_of_aten "bias" bias in
               return (Some bias)
             else return None
           in
           let* x = native_of_aten "input" aten_x in
           let* w = native_of_aten "weight" aten_w in
           try
             let* params =
               make_convolution_params ~op:node.Node.target sh sw ph pw dh dw
                 transposed oph opw groups
             in
             let tensors = [ x; w ] @ Option.to_list bias_opt in
             build_g ~name:"convolution_relayout" tensors (function
               | [ x_id; w_id ] ->
                   let open Graph_builder in
                   let* x' = permute perm_nchw_to_nhwc x_id in
                   let* w' = permute perm_oihw_to_conv_weight w_id in
                   let* y' = convolution params ~x:x' ~weight:w' () in
                   let+ y = permute perm_nhwc_to_nchw y' in
                   [ y ]
               | [ x_id; w_id; b_id ] ->
                   let open Graph_builder in
                   let* x' = permute perm_nchw_to_nhwc x_id in
                   let* w' = permute perm_oihw_to_conv_weight w_id in
                   let* y' =
                     convolution params ~x:x' ~weight:w' ~bias:b_id ()
                   in
                   let+ y = permute perm_nhwc_to_nchw y' in
                   [ y ]
               | _ -> assert false)
           with Invalid_argument msg -> fail (`Validation_failure msg))
  | "torch.ops.aten.div.Tensor" | "torch.ops.aten.div_.Tensor" ->
      Some
        (let* a = native_tensor_arg aten_env node "self" in
         let* other = tensor_or_scalar aten_env node "other" in
         match other with
         | `Tensor b ->
             build_g ~name:"div" [ a; b ] (function
               | [ a_id; b_id ] ->
                   let open Graph_builder in
                   let+ y = div a_id b_id in
                   [ y ]
               | _ -> assert false)
         | `Scalar scalar ->
             build_g ~name:"div_scalar" [ a ] (function
               | [ a_id ] ->
                   let open Graph_builder in
                   let+ y = div_scalar scalar a_id in
                   [ y ]
               | _ -> assert false))
  | "torch.ops.aten.gelu.default" ->
      Some
        (let* x = native_tensor_arg aten_env node "self" in
         let* approximate = string_arg ~default:"none" node "approximate" in
         let* approximate =
           match approximate with
           | "none" -> return Pointwise.Gelu.Exact
           | "tanh" -> return Pointwise.Gelu.Tanh
           | _ ->
               fail
                 (`Validation_failure
                    (Printf.sprintf
                       "gelu approximate=%s is not supported (only \"none\" or \
                        \"tanh\")"
                       approximate))
         in
         build_g ~name:"gelu" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = gelu approximate x_id in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.hardsigmoid.default" | "torch.ops.aten.hardsigmoid_.default"
    ->
      Some
        (let* x = native_tensor_arg aten_env node "self" in
         build_g ~name:"hardsigmoid" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = hardsigmoid x_id in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.hardswish.default" | "torch.ops.aten.hardswish_.default" ->
      Some
        (let* x = native_tensor_arg aten_env node "self" in
         build_g ~name:"hardswish" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = hardswish x_id in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.hardtanh.default" | "torch.ops.aten.hardtanh_.default" ->
      Some
        (let* x = native_tensor_arg aten_env node "self" in
         (* Schema `Scalar min_val=-1, Scalar max_val=1`: both bounds may arrive
            as Int or Float, and either may be absent. *)
         let* min_val =
           scalar_arg ~default:(Aten_scalar.Float (-1.)) node "min_val"
         in
         let* max_val =
           scalar_arg ~default:(Aten_scalar.Float 1.) node "max_val"
         in
         build_g ~name:"hardtanh" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = hardtanh { Pointwise.Hardtanh.min_val; max_val } x_id in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.linear.default" ->
      Some
        (let* aten_x = tensor_arg aten_env node "input" in
         let* aten_w = tensor_arg aten_env node "weight" in
         let w_shape = Aten_tensor.shape aten_w in
         if Array.length w_shape <> 2 then
           fail (`Linear_invalid_weight_rank w_shape)
         else
           let* bias_opt =
             if optional_tensor_present node "bias" then
               let* bias = tensor_arg aten_env node "bias" in
               let* () = require_rank "bias" ~expected:1 bias in
               let* bias = native_of_aten "bias" bias in
               return (Some bias)
             else return None
           in
           let* x = native_of_aten "input" aten_x in
           let* w = native_of_aten "weight" aten_w in
           try
             let params =
               { Linear.Linear.in_features = Dim.extent w_shape.(1) }
             in
             let tensors = [ x; w ] @ Option.to_list bias_opt in
             build_g ~name:"linear_relayout" tensors (function
               | [ x_id; w_id ] ->
                   let open Graph_builder in
                   let* w' = permute perm_linear_weight w_id in
                   let+ y = linear params ~x:x_id ~weight:w' () in
                   [ y ]
               | [ x_id; w_id; b_id ] ->
                   let open Graph_builder in
                   let* w' = permute perm_linear_weight w_id in
                   let+ y = linear params ~x:x_id ~weight:w' ~bias:b_id () in
                   [ y ]
               | _ -> assert false)
           with Invalid_argument msg -> fail (`Validation_failure msg))
  | "torch.ops.aten.max_pool2d.default" ->
      Some
        (let* aten_x = tensor_arg aten_env node "self" in
         let* () = reject_pool_extras node in
         let* kernel_size = ints_arg node "kernel_size" in
         let* stride = pool_stride kernel_size node in
         let* padding = ints_arg ~default:[ 0; 0 ] node "padding" in
         let* kh, kw = hw2 "kernel_size" kernel_size in
         let* sh, sw = hw2 "stride" stride in
         let* ph, pw = hw2 "padding" padding in
         let* x = native_of_aten "self" aten_x in
         let* params =
           make_pool_params ~op:node.Node.target kh kw sh sw ph pw
         in
         try
           build_g ~name:"max_pool2d_relayout" [ x ] (function
             | [ x_id ] ->
                 let open Graph_builder in
                 let* x' = permute perm_nchw_to_nhwc x_id in
                 let* y' = max_pool2d params x' in
                 let+ y = permute perm_nhwc_to_nchw y' in
                 [ y ]
             | _ -> assert false)
         with Invalid_argument msg -> fail (`Validation_failure msg))
  | "torch.ops.aten.adaptive_avg_pool2d.default" ->
      Some
        (let* aten_x = tensor_arg aten_env node "self" in
         let got = aten_rank aten_x in
         let* () =
           if got = 3 || got = 4 then return ()
           else fail (`Adaptive_pool_rank { Adaptive_pool_rank.got })
         in
         let* output_size = ints_arg node "output_size" in
         let* out_h, out_w =
           match output_size with
           | [ h; w ] -> return (h, w)
           | values -> fail (`Invalid_hw_arg { name = "output_size"; values })
         in
         let* h = pos ~op:node.Node.target ~param:`Output_size out_h in
         let* w = pos ~op:node.Node.target ~param:`Output_size out_w in
         let* x = native_of_aten "self" aten_x in
         let params = { Pool.AdaptiveAvgPool2d.output_size = { h; w } } in
         build_g ~name:"adaptive_avg_pool2d_relayout" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let* x' = permute perm_nchw_to_nhwc x_id in
               let* y' = adaptive_avg_pool2d params x' in
               let+ y = permute perm_nhwc_to_nchw y' in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.max_pool2d_with_indices.default" ->
      Some
        ((* Two ATen outputs (values, indices). We materialise both, relayout the
            value back to NCHW as the graph output, and route the dead indices
            edge into a Discard sink (see .ai/native_multi_output_design.md). *)
         let* aten_x = tensor_arg aten_env node "self" in
         let* () = reject_pool_extras node in
         let* kernel_size = ints_arg node "kernel_size" in
         let* stride = pool_stride kernel_size node in
         let* padding = ints_arg ~default:[ 0; 0 ] node "padding" in
         let* kh, kw = hw2 "kernel_size" kernel_size in
         let* sh, sw = hw2 "stride" stride in
         let* ph, pw = hw2 "padding" padding in
         let* x = native_of_aten "self" aten_x in
         let* params =
           make_pool_params ~op:node.Node.target kh kw sh sw ph pw
         in
         try
           build_g ~name:"max_pool2d_with_indices_relayout" [ x ] (function
             | [ x_id ] ->
                 let open Graph_builder in
                 let* x' = permute perm_nchw_to_nhwc x_id in
                 let* values, indices = max_pool2d_with_indices params x' in
                 let* () = discard indices in
                 let+ y = permute perm_nhwc_to_nchw values in
                 [ y ]
             | _ -> assert false)
         with Invalid_argument msg -> fail (`Validation_failure msg))
  | "torch.ops.aten.mean.dim" ->
      Some
        (let* t = tensor_arg aten_env node "self" in
         let rank = aten_rank t in
         let* dims = dims_arg node ~op:"mean.dim" ~rank "dim" in
         let* keepdim = bool_arg node "keepdim" in
         let* x = native_of_aten "self" t in
         let params = { Reduce.Mean.dims; keepdim } in
         build_g ~name:"mean" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = mean params x_id in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.mul.Tensor" | "torch.ops.aten.mul_.Tensor" ->
      Some
        (let* a = native_tensor_arg aten_env node "self" in
         let* other = tensor_or_scalar aten_env node "other" in
         match other with
         | `Tensor b ->
             build_g ~name:"mul" [ a; b ] (function
               | [ a_id; b_id ] ->
                   let open Graph_builder in
                   let+ y = mul a_id b_id in
                   [ y ]
               | _ -> assert false)
         | `Scalar scalar ->
             build_g ~name:"mul_scalar" [ a ] (function
               | [ a_id ] ->
                   let open Graph_builder in
                   let+ y = mul_scalar scalar a_id in
                   [ y ]
               | _ -> assert false))
  | "torch.ops.aten.mul.Scalar" ->
      Some
        (let* a = native_tensor_arg aten_env node "self" in
         let* s = decode_result (D.scalar_arg_result node "other") in
         let* scalar = float_of_aten_scalar "other" s in
         build_g ~name:"mul_scalar" [ a ] (function
           | [ a_id ] ->
               let open Graph_builder in
               let+ y = mul_scalar scalar a_id in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.permute.default" ->
      Some
        (let* t = tensor_arg aten_env node "self" in
         let rank = aten_rank t in
         let* dims = ints_arg node "dims" in
         let* perm = native_perm_of_aten ~op:"permute.default" ~rank dims in
         let* x = native_of_aten "self" t in
         build_g ~name:"permute" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = permute perm x_id in
               [ y ]
           | _ -> assert false))
  (* Reuses [native_perm_of_aten], the same permute machinery
     [permute.default] builds on, rather than a new builder -- outer padding
     axes stay identity because that helper already does that. Equal dims are
     a real identity transpose, not special-cased away: [List.init] produces
     the identity list, and it lowers like any other permutation. *)
  | "torch.ops.aten.transpose.int" ->
      Some
        (let* t = tensor_arg aten_env node "self" in
         let rank = aten_rank t in
         let* dim0 = int_arg node "dim0" in
         let* dim1 = int_arg node "dim1" in
         let* d0 = norm_dim ~op:"transpose.int" ~rank dim0 in
         let* d1 = norm_dim ~op:"transpose.int" ~rank dim1 in
         let dims =
           List.init rank (fun i ->
               if i = d0 then d1 else if i = d1 then d0 else i)
         in
         let* perm = native_perm_of_aten ~op:"transpose.int" ~rank dims in
         let* x = native_of_aten "self" t in
         build_g ~name:"transpose" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = permute perm x_id in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.relu.default" | "torch.ops.aten.relu_.default" -> (
      match native_tensor_arg aten_env node "self" with
      | Error e -> Some (Error e)
      | Ok x ->
          build_g ~name:"relu" [ x ] (function
            | [ x_id ] ->
                let open Graph_builder in
                let+ y = relu x_id in
                [ y ]
            | _ -> assert false)
          |> some_graph)
  (* [input] is right-aligned NCHW like [batch_norm]'s, so the same
     [perm_nchw_to_nhwc]/[perm_nhwc_to_nchw] pair relays it around the op --
     [Group_norm.Compute] reads its window on native C, which is where the
     permute puts ATen's channel axis. Unlike [batch_norm]'s arm, absent
     [weight]/[bias] stay [None] rather than being materialised as ones/zeros
     tensors here, for the reason the [layer_norm]/[native_layer_norm] arm
     below states at length: [Graph_ir] carries them as options and [Eval_op]
     fills them, so materialising upstream would build a structurally
     different graph from [Native_interp]'s.

     NOT bound in [bin/aten_ops_gen.ml]'s [selection] (deliberately, unlike
     every other op this bridge lowers): `at::native::group_norm`'s CPU
     kernel is outside the hand-curated source closure [lib/aten/
     build_archive.sh] compiles (see its own header comment on why that
     closure is small on purpose) -- pulling it in undefined-symbol'd every
     binary linking [aten] at [RegisterCompositeImplicitAutograd_0.cpp].
     This arm needs no ATen C binding to build a Native graph, only
     [Interp_decode]'s node-argument helpers, so real-model import and the
     payload-free sweep both work; only real-ATen verification
     ([Interp_verify]/[Interp_dispatch]) is unavailable for this op until
     that closure grows. *)
  | "torch.ops.aten.group_norm.default" ->
      Some
        (let* aten_x = tensor_arg aten_env node "input" in
         let* num_groups = int_arg node "num_groups" in
         let* eps = float_arg ~default:1e-05 node "eps" in
         (* Decoded, not ignored, then discarded -- the cuDNN implementation
            hint, the same treatment [layer_norm.default]'s [cudnn_enable]
            gets below and for the same reason: a non-boolean here is a
            malformed node, and this is what says so. *)
         let* (_ : bool) = bool_arg ~default:true node "cudnn_enabled" in
         let* groups = pos ~op:"group_norm.default" ~param:`Groups num_groups in
         let* x = native_of_aten "input" aten_x in
         let* affine =
           Err.List.map
             (fun name ->
               if optional_tensor_present node name then
                 let* t = tensor_arg aten_env node name in
                 let* () =
                   require_rank
                     (Printf.sprintf "group_norm %s" name)
                     ~expected:1 t
                 in
                 let* t = native_of_aten name t in
                 return (Some t)
               else return None)
             [ "weight"; "bias" ]
         in
         let weight_opt, bias_opt =
           match affine with [ w; b ] -> (w, b) | _ -> assert false
         in
         let params = { Norm.GroupNorm.channel = Axis.C; groups; eps } in
         build_g ~name:"group_norm_relayout"
           (([ x ] @ Option.to_list weight_opt) @ Option.to_list bias_opt)
           (fun ids ->
             let open Graph_builder in
             (* All FOUR states spelled out, matched against the options the
                operand list was built from -- the same discipline
                [layer_norm]'s arm below uses and for the same reason: "bias
                but no weight" is a state a paired encoding would get wrong. *)
             let x_id, weight, bias =
               match (ids, weight_opt, bias_opt) with
               | [ x_id; w_id; b_id ], Some _, Some _ ->
                   (x_id, Some w_id, Some b_id)
               | [ x_id; w_id ], Some _, None -> (x_id, Some w_id, None)
               | [ x_id; b_id ], None, Some _ -> (x_id, None, Some b_id)
               | [ x_id ], None, None -> (x_id, None, None)
               | _ -> assert false
             in
             let* x' = permute perm_nchw_to_nhwc x_id in
             let* y' = group_norm params ~x:x' ?weight ?bias () in
             let+ y = permute perm_nhwc_to_nchw y' in
             [ y ]))
  (* The FUNCTIONAL layer norm and its DECOMPOSED twin, in one body. They differ
     in three things and in nothing else: [native_layer_norm]'s [eps] is
     required with no schema default, it has no [cudnn_enable], and it returns
     a 3-tuple. The arithmetic, the axis derivation, the normalized_shape
     validation and the affine handling are identical, so they are written once.

     THE 3-TUPLE NEEDS NOTHING HERE. [Verify.requires_exact_outputs] is true
     only for a dynamic [Argument.Tensors] return; a fixed tuple falls under the
     leading-outputs rule, so exposing one output is legitimate and is verified
     against the first ATen result alone. The liveness question -- whether a
     graph READS [mean] or [rstd] -- is a whole-graph property this single-node
     bridge cannot see, and [Native_interp] is where it is answered
     ([`Live_layer_norm_stats]). *)
  | ( "torch.ops.aten.layer_norm.default"
    | "torch.ops.aten.native_layer_norm.default" ) as target ->
      Some
        (let functional = target = "torch.ops.aten.layer_norm.default" in
         let op =
           if functional then Norm.Target.Layer_norm
           else Norm.Target.Native_layer_norm
         in
         let* t = tensor_arg aten_env node "input" in
         let* normalized_shape = ints_arg node "normalized_shape" in
         let* dims =
           normalized_dims ~op ~x_shape:(Aten_tensor.shape t) ~normalized_shape
         in
         (* [layer_norm]'s eps is a REQUIRED float with a schema default of
            1e-5, so [float_arg ~default], not [eps_arg]: rms_norm's eps is a
            [float?] whose absence means "ATen picks", and reading one with the
            other's decoder is how a null epsilon comes to be read as zero.
            [native_layer_norm]'s has no default at all -- a third spelling of
            the same argument -- so its absence is a malformed node. *)
         let* eps =
           if functional then float_arg ~default:1e-05 node "eps"
           else float_arg node "eps"
         in
         (* Decoded, not ignored, and then deliberately DISCARDED -- which is
            the one argument in this repository where that is the faithful
            reading rather than the [alpha]-shaped bug. ATen's own composite is
            [layer_norm_symint (..., bool /* cudnn_enable, deprecated */)]: it
            names the parameter in a comment and drops it, computing
            native_layer_norm either way. Accepting only the value some corpus
            happens to show would reject the schema's own default of true and
            with it almost every real node. Decoding it still matters: a
            non-boolean there is a malformed node, and this is what says so.
            The decomposed form does not carry the argument at all -- none
            survives export -- so it is read only for the functional one. *)
         let* () =
           if functional then
             let* (_ : bool) = bool_arg ~default:true node "cudnn_enable" in
             return ()
           else return ()
         in
         let params = { Norm.LayerNorm.dims; eps } in
         let* x = native_of_aten "input" t in
         (* NO ones/zeros tensors for absent affine operands, for the reason the
            rms_norm arm below states at length: [Graph_ir] carries them as
            options, [Eval_op] fills them, and materialising here would build a
            structurally different graph from [Native_interp]'s. *)
         let k = List.length normalized_shape in
         let* affine =
           Err.List.map
             (fun name ->
               if optional_tensor_present node name then
                 let* t = tensor_arg aten_env node name in
                 (* Rank [k], the length of normalized_shape: ATen indexes both
                    affine operands by the whole normalized shape. *)
                 let* () =
                   require_rank
                     (Fmt.str "%a %s" Norm.Target.pp op name)
                     ~expected:k t
                 in
                 let* t = native_of_aten name t in
                 return (Some t)
               else return None)
             [ "weight"; "bias" ]
         in
         let weight_opt, bias_opt =
           match affine with [ w; b ] -> (w, b) | _ -> assert false
         in
         build_g ~name:"layer_norm"
           (([ x ] @ Option.to_list weight_opt) @ Option.to_list bias_opt)
           (fun ids ->
             let open Graph_builder in
             (* All FOUR states spelled out, matched against the options the
                operand list was built from -- so the id positions and the
                options cannot disagree. "bias but no weight" is the state no
                model produces and the one a paired encoding would get wrong,
                which is why it is written rather than folded away. *)
             let x_id, weight, bias =
               match (ids, weight_opt, bias_opt) with
               | [ x ], None, None -> (x, None, None)
               | [ x; w ], Some _, None -> (x, Some w, None)
               | [ x; b ], None, Some _ -> (x, None, Some b)
               | [ x; w; b ], Some _, Some _ -> (x, Some w, Some b)
               | _ -> assert false
             in
             let+ y = layer_norm params ~x:x_id ?weight ?bias () in
             [ y ]))
  | "torch.ops.aten.rms_norm.default" ->
      Some
        (let* t = tensor_arg aten_env node "input" in
         let* normalized_shape = ints_arg node "normalized_shape" in
         let* dims =
           normalized_dims ~op:Norm.Target.Rms_norm
             ~x_shape:(Aten_tensor.shape t) ~normalized_shape
         in
         let* eps = eps_arg node "eps" in
         let params = { Norm.RmsNorm.dims; eps } in
         let* x = native_of_aten "input" t in
         (* NO ones tensor for an absent weight. [Graph_ir]'s [Rms_norm] carries
            [weight : Tensor_ref.t option] and Native4D reads the option
            (lower.ml:293-299); materializing a constant made this path build a
            structurally different graph from [Native_interp]'s for the same
            node, and left that arm unreachable from here. The numeric result is
            unchanged -- multiplying by ones is what the option's absence
            means. *)
         let* weight_opt =
           if optional_tensor_present node "weight" then
             let* weight = tensor_arg aten_env node "weight" in
             (* Rank [k], the length of normalized_shape: ATen indexes the
                weight by the whole normalized shape. *)
             let* () =
               require_rank "rms_norm weight"
                 ~expected:(List.length normalized_shape)
                 weight
             in
             let* weight = native_of_aten "weight" weight in
             return (Some weight)
           else return None
         in
         build_g ~name:"rms_norm"
           ([ x ] @ Option.to_list weight_opt)
           (function
             | [ x_id ] ->
                 let open Graph_builder in
                 let+ y = rms_norm params ~x:x_id () in
                 [ y ]
             | [ x_id; w_id ] ->
                 let open Graph_builder in
                 let+ y = rms_norm params ~x:x_id ~weight:w_id () in
                 [ y ]
             | _ -> assert false))
  (* op8-impl.md commit 3. Both arms (this one and Native_interp's) implement
     the SAME contract and neither calls ATen: [dropout_p]/[is_causal]/
     [enable_gqa] are typed rejections rather than fields [Attention.Sdpa.t]
     carries (op8-impl.md's [params] has only [scale]); a boolean mask is
     rejected rather than reinterpreted as f32 (F10); Q/K/V/mask rank is
     checked on the RAW ATen tensor, before [native_of_aten] erases it
     (F13) -- the only place it can be, since [Tensor_sig.t] keeps no rank
     field for [Attention.Sdpa.output_shape] to check downstream.
     [value.C <> query.C] (Ev <> E) and an inadmissible mask broadcast shape
     are NOT re-checked here: both are flash-oracle boundaries (F4), not
     rank facts, and [Attention.Sdpa.output_shape] already rejects them (via
     [Graph_builder.build]'s [`Build]) with the same [Shape_error.Sdpa] rows
     [Native_interp] reaches through the same op. *)
  | "torch.ops.aten.scaled_dot_product_attention.default" ->
      Some
        (let* query = tensor_arg aten_env node "query" in
         let* key = tensor_arg aten_env node "key" in
         let* value = tensor_arg aten_env node "value" in
         let* () = require_rank "sdpa query" ~expected:4 query in
         let* () = require_rank "sdpa key" ~expected:4 key in
         let* () = require_rank "sdpa value" ~expected:4 value in
         let* () = require_f32 "sdpa query" query in
         let* () = require_f32 "sdpa key" key in
         let* () = require_f32 "sdpa value" value in
         let* dropout_p = float_arg ~default:0.0 node "dropout_p" in
         let* () =
           if Float.equal dropout_p 0.0 then return ()
           else fail (`Sdpa_reject (Attention.Sdpa.Reject.Dropout dropout_p))
         in
         let* is_causal = bool_arg ~default:false node "is_causal" in
         let* () =
           if is_causal then fail (`Sdpa_reject Attention.Sdpa.Reject.Causal)
           else return ()
         in
         let* enable_gqa = bool_arg ~default:false node "enable_gqa" in
         let* () =
           if enable_gqa then fail (`Sdpa_reject Attention.Sdpa.Reject.Gqa)
           else return ()
         in
         let* scale_opt = float_opt_arg node "scale" in
         let* scale =
           match scale_opt with
           | None -> return Attention.Sdpa.Scale.Default
           | Some s ->
               let* () =
                 if Float.is_finite s then return ()
                 else
                   fail
                     (`Sdpa_reject (Attention.Sdpa.Reject.Non_finite_scale s))
               in
               let* () =
                 if Float.compare s 0.0 < 0 then
                   fail (`Sdpa_reject (Attention.Sdpa.Reject.Negative_scale s))
                 else return ()
               in
               return (Attention.Sdpa.Scale.Explicit s)
         in
         let* mask_opt =
           if optional_tensor_present node "attn_mask" then
             let* m = tensor_arg aten_env node "attn_mask" in
             let* () =
               match Aten_tensor.scalar_type m with
               | Aten_scalar_type.Bool ->
                   fail (`Sdpa_reject Attention.Sdpa.Reject.Boolean_mask)
               | _ -> return ()
             in
             let* () = require_f32 "sdpa attn_mask" m in
             let got = aten_rank m in
             let* () =
               if got = 2 || got = 4 then return ()
               else
                 fail
                   (`Sdpa_reject
                      (Attention.Sdpa.Reject.Rank
                         {
                           arg_name = "sdpa attn_mask";
                           expected = [ 2; 4 ];
                           got;
                         }))
             in
             let* mask = native_of_aten "attn_mask" m in
             return (Some mask)
           else return None
         in
         let* q = native_of_aten "query" query in
         let* k = native_of_aten "key" key in
         let* v = native_of_aten "value" value in
         let params = { Attention.Sdpa.scale } in
         build_g ~name:"sdpa"
           ([ q; k; v ] @ Option.to_list mask_opt)
           (function
             | [ q_id; k_id; v_id ] ->
                 let open Graph_builder in
                 let+ y = sdpa params ~query:q_id ~key:k_id ~value:v_id () in
                 [ y ]
             | [ q_id; k_id; v_id; m_id ] ->
                 let open Graph_builder in
                 let+ y =
                   sdpa params ~query:q_id ~key:k_id ~value:v_id ~mask:m_id ()
                 in
                 [ y ]
             | _ -> assert false))
  | "torch.ops.aten.sigmoid.default" ->
      Some
        (let* x = native_tensor_arg aten_env node "self" in
         build_g ~name:"sigmoid" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = sigmoid x_id in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.silu.default" | "torch.ops.aten.silu_.default" ->
      Some
        (let* x = native_tensor_arg aten_env node "self" in
         build_g ~name:"silu" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = silu x_id in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.sqrt.default" | "torch.ops.aten.sqrt_.default" -> (
      match native_tensor_arg aten_env node "self" with
      | Error e -> Some (Error e)
      | Ok x ->
          build_g ~name:"sqrt" [ x ] (function
            | [ x_id ] ->
                let open Graph_builder in
                let+ y = sqrt x_id in
                [ y ]
            | _ -> assert false)
          |> some_graph)
  (* [x - s] legalizes to [x + (-s)]: IEEE negation is exact and the builder
     narrows to f32 on both spellings either way, so the two are bit-identical
     (op3-impl.md F7). No [sub_scalar] builder exists and none should: this
     negation is the whole legalization. *)
  | "torch.ops.aten.sub.Tensor" | "torch.ops.aten.sub_.Tensor" ->
      Some
        (let* () = reject_alpha node in
         let* a = native_tensor_arg aten_env node "self" in
         let* other = tensor_or_scalar aten_env node "other" in
         match other with
         | `Tensor b ->
             build_g ~name:"sub" [ a; b ] (function
               | [ a_id; b_id ] ->
                   let open Graph_builder in
                   let+ y = sub a_id b_id in
                   [ y ]
               | _ -> assert false)
         | `Scalar scalar ->
             build_g ~name:"sub_scalar" [ a ] (function
               | [ a_id ] ->
                   let open Graph_builder in
                   let+ y = add_scalar (-.scalar) a_id in
                   [ y ]
               | _ -> assert false))
  (* The only arm returning a variable number of outputs, and the only one whose
     count is fixed by the OPERAND rather than the op. Every slice is exposed:
     unlike a fixed tuple's dead output, there is nothing here to drop, and
     [Verify.verify_node] requires exact cardinality for a dynamic list for
     precisely that reason.

     This must NOT call ATen's own unbind. ATen is the oracle
     [Interp_verify]/[Verify.verify_node] runs; the native side has to execute
     [Graph_ir.Unbind] through [Eval_direct] or the comparison is vacuous. *)
  | "torch.ops.aten.pad.default" ->
      Some
        (let* aten_x = tensor_arg aten_env node "self" in
         let* () = require_f32 "self" aten_x in
         (* Rank from the ORIGINAL ATen tensor, before [of_aten] right-aligns it
            into the frame and the rank stops being recoverable -- the same
            ordering [unbind.int] uses, and here it is load-bearing twice over:
            the pad list is indexed FROM the innermost dimension, so a wrong rank
            silently pads the wrong axes. *)
         let rank = aten_rank aten_x in
         let* pad = ints_arg node "pad" in
         let* mode = string_arg ~default:"constant" node "mode" in
         let* value = float_opt_arg node "value" in
         (* No [Err.map_error]: [params_of_aten] already fails with this
            module's own [`Bad_pad_list] row, so the detection origin is the
            check that found the fault rather than this call site. *)
         let* params = Pad.Pad.params_of_aten ~rank ~pad ~mode ~value in
         let* x = native_of_aten "self" aten_x in
         build_g ~name:"pad" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = pad params x_id in
               [ y ]
           | _ -> assert false))
  (* The bounds come from the LIVE tensor's extent, which is the whole reason
     this arm and the importer's differ at all: everything after
     [Aten_shape.resolve_slice] is shared, and the extent is the one thing the
     two paths learn differently. Rank before [of_aten], as [pad.default] and
     [unbind.int] do -- the frame does not carry it. *)
  | "torch.ops.aten.slice.Tensor" ->
      Some
        (let* aten_x = tensor_arg aten_env node "self" in
         let* () = require_f32 "self" aten_x in
         let rank = aten_rank aten_x in
         let* dim = int_arg ~default:0 node "dim" in
         let* start = int_opt_arg node "start" in
         let* stop = int_opt_arg node "end" in
         let* step = int_arg ~default:1 node "step" in
         let* x = native_of_aten "self" aten_x in
         let* axis = dim_axis ~op:"slice.Tensor" ~rank dim in
         let extent = Vec6.get (packed_shape x) axis in
         let* bounds =
           Err.map_error
             (fun e -> `Aten_shape e)
             (Aten_shape.resolve_slice ~extent ~start ~stop ~step)
         in
         build_g ~name:"slice" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y =
                 slice
                   {
                     Split.Slice.axis;
                     start = bounds.Aten_shape.Slice_bounds.start;
                     stop = bounds.Aten_shape.Slice_bounds.stop;
                     step = bounds.Aten_shape.Slice_bounds.step;
                   }
                   x_id
               in
               [ y ]
           | _ -> assert false))
  (* The variadic Native [Concat] op, direct: every operand keeps its rank,
     [dim] names an existing axis. ATen requires every tensor to share one
     rank (no broadcasting across the list), checked explicitly rather than
     left to [Concat.output_shape]'s own axis-agreement rule — see
     [Concat_rank_mismatch]'s doc comment for why that rule would report the
     wrong fault. *)
  | "torch.ops.aten.cat.default" ->
      Some
        (let* aten_xs = tensors_arg aten_env node "tensors" in
         let* () = Err.List.iter (require_f32 "tensors") aten_xs in
         let* dim = int_arg ~default:0 node "dim" in
         match aten_xs with
         | [] -> fail (`Concat_no_tensors "cat.default")
         | aten_x0 :: _ ->
             let rank = aten_rank aten_x0 in
             let* () =
               Err.List.iter
                 (fun t ->
                   let got = aten_rank t in
                   if got = rank then return ()
                   else
                     fail
                       (`Concat_rank_mismatch
                          {
                            Concat_rank_mismatch.op = "cat.default";
                            first = rank;
                            other = got;
                          }))
                 aten_xs
             in
             let* d = norm_dim ~op:"cat.default" ~rank dim in
             let axis = Aten_shape.axis_of_dim ~rank d in
             let* xs = Err.List.map (native_of_aten "tensors") aten_xs in
             build_g ~name:"cat" xs (fun ids ->
                 let open Graph_builder in
                 let+ y = concat { Concat.Concat.axis } ids in
                 [ y ]))
  (* One [Stack] node: inserts a size-1 axis per operand at [axis], then joins
     them — ATen's own definition is exactly
     `cat([t.unsqueeze(dim) for t in tensors], dim)`, which [Stack]'s shape
     rule/[Compute] reuse from [Concat]/[Split.Select]'s implementations
     rather than materializing as separate [Reshape] nodes. [dim] is judged
     against rank+1 valid positions, the OUTPUT rank, same reasoning as
     [norm_unsqueeze_dim]. *)
  | "torch.ops.aten.stack.default" ->
      Some
        (let* aten_xs = tensors_arg aten_env node "tensors" in
         let* () = Err.List.iter (require_f32 "tensors") aten_xs in
         let* dim = int_arg ~default:0 node "dim" in
         match aten_xs with
         | [] -> fail (`Concat_no_tensors "stack.default")
         | aten_x0 :: _ ->
             let rank = aten_rank aten_x0 in
             let* () =
               Err.List.iter
                 (fun t ->
                   let got = aten_rank t in
                   if got = rank then return ()
                   else
                     fail
                       (`Concat_rank_mismatch
                          {
                            Concat_rank_mismatch.op = "stack.default";
                            first = rank;
                            other = got;
                          }))
                 aten_xs
             in
             let* d = norm_unsqueeze_dim ~op:"stack.default" ~rank dim in
             let* xs = Err.List.map (native_of_aten "tensors") aten_xs in
             let axis = Aten_shape.axis_of_dim ~rank:(rank + 1) d in
             build_g ~name:"stack" xs (fun ids ->
                 let open Graph_builder in
                 let+ y = stack { Concat.Stack.axis } ids in
                 [ y ]))
  (* One [Select] node: picks index [idx] along [axis] and drops it. Unlike
     [slice.Tensor], ATen REJECTS an out-of-range [index] rather than clamping
     it -- [Aten_shape.resolve_index], not [resolve_slice]. *)
  | "torch.ops.aten.select.int" ->
      Some
        (let* aten_x = tensor_arg aten_env node "self" in
         let* () = require_f32 "self" aten_x in
         let rank = aten_rank aten_x in
         let* dim = int_arg node "dim" in
         let* index = int_arg node "index" in
         let* x = native_of_aten "self" aten_x in
         let* d = norm_dim ~op:"select.int" ~rank dim in
         let axis = Aten_shape.axis_of_dim ~rank d in
         let extent = Vec6.get (packed_shape x) axis in
         let* idx =
           Err.map_error
             (fun e -> `Aten_shape e)
             (Aten_shape.resolve_index ~extent ~index)
         in
         build_g ~name:"select" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = select { Split.Select.axis; index = idx } x_id in
               [ y ]
           | _ -> assert false))
  (* Legalized to [Reshape] alone: inserting a size-1 axis never changes the
     linearized data order, so no [Slice] is needed, unlike [select.int]'s
     axis removal. The target is [Aten_shape.to_aten]'s list with a bare [1]
     spliced in at the normalized position -- the same round-trip
     [select.int]/[view.default] use. *)
  | "torch.ops.aten.unsqueeze.default" ->
      Some
        (let* aten_x = tensor_arg aten_env node "self" in
         let* () = require_f32 "self" aten_x in
         let rank = aten_rank aten_x in
         let* dim = int_arg node "dim" in
         let* x = native_of_aten "self" aten_x in
         let* d = norm_unsqueeze_dim ~op:"unsqueeze.default" ~rank dim in
         let aten_list =
           Array.to_list (Aten_shape.to_aten ~rank (packed_shape x))
         in
         let front = List.filteri (fun i _ -> i < d) aten_list in
         let back = List.filteri (fun i _ -> i >= d) aten_list in
         let out_shape = Array.of_list (front @ [ 1 ] @ back) in
         let* target =
           Err.map_error (fun e -> `Aten_shape e) (Aten_shape.of_aten out_shape)
         in
         build_g ~name:"unsqueeze" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = reshape { Reshape.Reshape.shape = target } x_id in
               [ y ]
           | _ -> assert false))
  | "torch.ops.aten.unbind.int" ->
      Some
        (let* aten_x = tensor_arg aten_env node "self" in
         (* Rank comes from the ORIGINAL ATen tensor: [of_aten] right-aligns
            into the six-axis frame, after which the rank is not recoverable. *)
         let rank = aten_rank aten_x in
         let* dim = int_arg ~default:0 node "dim" in
         (* Convert BEFORE judging the dim, so a rank Native cannot represent is
            reported as the rank fault it is rather than as a bad dimension. *)
         let* x = native_of_aten "self" aten_x in
         let* axis = dim_axis ~op:"unbind.int" ~rank dim in
         build_g ~name:"unbind" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               unbind { Split.Unbind.axis } x_id
           | _ -> assert false))
  (* One [Split_with_sizes] node: divides [axis] into contiguous windows of
     [split_sizes], KEEPING the axis in every output, unlike [unbind.int]
     which drops it. [split_sizes] is a required arg (no schema default), the
     same as [view.default]'s [size]; [Split.Split_with_sizes.output_shapes]
     is what checks the sizes are positive and sum to the axis's extent, not
     this arm -- the same division of labor [cat.default]'s rank check and
     [Concat.output_shape]'s axis-agreement check follow. Preserves the
     input's own dtype like [unbind.int], not [require_f32]'d like
     [cat.default]/[stack.default]. *)
  | "torch.ops.aten.split_with_sizes.default" ->
      Some
        (let* aten_x = tensor_arg aten_env node "self" in
         let rank = aten_rank aten_x in
         let* dim = int_arg ~default:0 node "dim" in
         let* sizes = ints_arg node "split_sizes" in
         let* x = native_of_aten "self" aten_x in
         let* axis = dim_axis ~op:"split_with_sizes.default" ~rank dim in
         build_g ~name:"split_with_sizes" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               split_with_sizes { Split.Split_with_sizes.axis; sizes } x_id
           | _ -> assert false))
  (* Both overloads share this body: after commit 1's checked resolver the body
     is three calls, and two copies of it is exactly the drift risk this repo
     keeps warning about. Still exact-target dispatch in op3.md's sense -- both
     targets are named, there is no fallthrough, and [native_of_aten]/every
     diagnostic below reads [node.target], so a failure still says which
     overload it was (op3-impl.md Part IV #2). [Identical] AFTER
     materialization, not alias-identical: ATen may return a view over the
     same storage, native graph edges are values. See
     .ai/native_aten_bridge_layout.md. *)
  | "torch.ops.aten.view.default" | "torch.ops.aten._unsafe_view.default" ->
      Some
        ((* Contiguous reshape. [of_aten] inputs are already ATen-row-major, so
            the native reshape needs no surrounding permutes; the target native
            shape is [size] right-aligned (a single -1 resolved against numel). *)
         let* aten_x = tensor_arg aten_env node "self" in
         let* size = ints_arg node "size" in
         let* x = native_of_aten "self" aten_x in
         let (Tensor.Tensor r) = x in
         (* [x]'s shape already cleared [Tensor_bridge.of_aten]'s numel
            preflight, so this count is < [Kernel.Limits.Hard.numel] and the
            [int64] conversion below cannot itself be the overflow this design
            guards against. *)
         let numel = Int64.of_int (Dim.to_int (Vec6.numel r.shape)) in
         let* resolved =
           Err.map_error
             (fun e -> `Aten_shape e)
             (Aten_shape.resolve_view_size ~numel size)
         in
         let* target =
           Err.map_error
             (fun e -> `Aten_shape e)
             (Aten_shape.of_aten (Array.of_list resolved))
         in
         let params = { Reshape.Reshape.shape = target } in
         build_g ~name:"view" [ x ] (function
           | [ x_id ] ->
               let open Graph_builder in
               let+ y = reshape params x_id in
               [ y ]
           | _ -> assert false))
  | _ -> None
