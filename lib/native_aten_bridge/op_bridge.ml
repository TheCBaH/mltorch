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
(* rms_norm's two [normalized_shape] faults. The arm read the LENGTH of
   [normalized_shape] and nothing else, so a shape naming extents the input does
   not have normalized over the wrong axes and returned a plausible wrong
   answer; and [trailing_axes] silently returns the whole axis list when
   [k > rank], so an over-long shape normalized over everything. *)
module Normalized_rank = struct
  type t = { rank : int; got : int }
end

module Normalized_shape = struct
  type t = { expected : int list; got : int list }
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
  | `Normalized_rank of Normalized_rank.t
  | `Normalized_shape of Normalized_shape.t
  | `Operand_rank of Operand_rank.t
  | `Bad_config of Op_config.Bad.t
  | `Unsupported_padding_mode of string
  | `Aten_shape of Aten_shape.error ]

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
  | `Normalized_rank { Normalized_rank.rank; got } ->
      Fmt.pf ppf
        "rms_norm: normalized_shape has %d entries, outside [1, %d] for this \
         rank"
        got rank
  | `Normalized_shape { Normalized_shape.expected; got } ->
      Fmt.pf ppf
        "rms_norm: normalized_shape %a does not match the input's trailing \
         extents %a"
        pp_int_list got pp_int_list expected
  | `Pool_unsupported { Pool_unsupported.op; option } -> (
      match option with
      | Pool_unsupported.Dilation d ->
          Fmt.pf ppf "%s: dilation=%a is not supported (only 1)" op pp_int_list
            d
      | Pool_unsupported.Ceil_mode ->
          Fmt.pf ppf "%s: ceil_mode=true is not supported" op)
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

let ( let* ) = Err.Syntax.( let* )
let return = Err.return
let fail = Err.fail
let decode_result r = Err.map_error (fun e -> `Decode e) r
let tensor_arg env node name = decode_result (D.tensor_arg_result env node name)

let int_arg ?default node name =
  decode_result (D.int_arg_result ?default node name)

let ints_arg ?(default = []) node name =
  decode_result (D.ints_arg_result ~default node name)

let bool_arg ?(default = false) node name =
  decode_result (D.bool_arg_result ~default node name)

let string_arg ~default node name =
  decode_result (D.string_arg_result ~default node name)

let packed_shape (Tensor.Tensor r) = r.shape

(* Monadically allocate one input edge per packed tensor, left to right.
   The resulting ids match graph.inputs in insertion order. *)
let rec alloc_inputs = function
  | [] -> Graph_builder.return []
  | t :: rest ->
      let open Graph_builder in
      let* id = input ~shape:(packed_shape t) () in
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
let dim_axis ~op ~rank dim =
  let d = if dim < 0 then dim + rank else dim in
  if rank < 1 || d < 0 || d >= rank then
    fail (`Invalid_dim { Invalid_dim.op; dim; rank })
  else return (Aten_shape.axis_of_dim ~rank dim)

let dims_arg node ~op ~rank name =
  match D.find_arg node name with
  | Some (Argument.Ints []) | Some (Argument.None _) | None ->
      return (Aten_shape.used_axes ~rank)
  | Some (Argument.Ints xs) -> Err.List.map (dim_axis ~op ~rank) xs
  | _ -> return (Aten_shape.used_axes ~rank)

(* The engine's compute domain is f32 and [Graph_builder] gives every op output
   [Payload.F32], so an i64 operand would silently become an f32 result that
   then fails the verifier's dtype pairing. [Tensor_bridge.of_aten] itself
   accepts i64, so the refusal belongs to the arm that knows what it will build.
   Read from the ATen tensor before conversion, so the error names the source
   dtype. *)
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
let normalized_dims ~(x_shape : int array) ~normalized_shape =
  let rank = Array.length x_shape in
  let k = List.length normalized_shape in
  let* () =
    if k < 1 || k > rank then
      fail (`Normalized_rank { Normalized_rank.rank; got = k })
    else return ()
  in
  let expected = Array.to_list (Array.sub x_shape (rank - k) k) in
  let* () =
    if expected <> normalized_shape then
      fail
        (`Normalized_shape { Normalized_shape.expected; got = normalized_shape })
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
  | "torch.ops.aten.mul.Tensor" | "torch.ops.aten.mul_.Tensor" -> (
      match
        ( native_tensor_arg aten_env node "self",
          native_tensor_arg aten_env node "other" )
      with
      | Error e, _ | _, Error e -> Some (Error e)
      | Ok a, Ok b ->
          build_g ~name:"mul" [ a; b ] (function
            | [ a_id; b_id ] ->
                let open Graph_builder in
                let+ y = mul a_id b_id in
                [ y ]
            | _ -> assert false)
          |> some_graph)
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
  | "torch.ops.aten.rms_norm.default" ->
      Some
        (let* t = tensor_arg aten_env node "input" in
         let* normalized_shape = ints_arg node "normalized_shape" in
         let* dims =
           normalized_dims ~x_shape:(Aten_tensor.shape t) ~normalized_shape
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
  | "torch.ops.aten.sub.Tensor" | "torch.ops.aten.sub_.Tensor" -> (
      match
        ( (let* () = reject_alpha node in
           native_tensor_arg aten_env node "self"),
          native_tensor_arg aten_env node "other" )
      with
      | Error e, _ | _, Error e -> Some (Error e)
      | Ok a, Ok b ->
          build_g ~name:"sub" [ a; b ] (function
            | [ a_id; b_id ] ->
                let open Graph_builder in
                let+ y = sub a_id b_id in
                [ y ]
            | _ -> assert false)
          |> some_graph)
  (* The only arm returning a variable number of outputs, and the only one whose
     count is fixed by the OPERAND rather than the op. Every slice is exposed:
     unlike a fixed tuple's dead output, there is nothing here to drop, and
     [Verify.verify_node] requires exact cardinality for a dynamic list for
     precisely that reason.

     This must NOT call ATen's own unbind. ATen is the oracle
     [Interp_verify]/[Verify.verify_node] runs; the native side has to execute
     [Graph_ir.Unbind] through [Eval_direct] or the comparison is vacuous. *)
  | "torch.ops.aten.unbind.int" ->
      Some
        (let* aten_x = tensor_arg aten_env node "self" in
         let* () = require_f32 "self" aten_x in
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
  | "torch.ops.aten.view.default" ->
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
