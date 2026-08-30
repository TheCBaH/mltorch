(* Argument decoding, shape/rank derivation and op-configuration helpers for
   [Native_interp.lower]. Split from native_interp.ml. Entirely
   internal: none of this is exposed through native_interp.mli, and [lower]
   (still in native_interp.ml) is the sole caller. Every helper takes its
   [Err.Escape.t] token explicitly, exactly as before the split, so moving
   this code changes nothing about control flow -- only its file. *)

open Pytorch_types
open Schema_runtime
open Native_interp_error
module Tensor_id = Graph_ir.Tensor_id

(* Internal control flow only — the .mli exposes [error] and nothing else. The
   lowering walk is deeply recursive and threading a result through every arm
   would rewrite it, so it exits through [Err.Escape] instead: one token per
   [with_escape] call, threaded to every helper that can detect a fault.

   It replaces a private [exception Lower_error of error] carrying the BARE row.
   That lost the [Err.Error.t] across this module's own boundary — the catch
   rebuilt the wrapper with [Err.fail], so every malformed graph was reported as
   detected at the catch site rather than where the fault was found — and it let
   [tensor_of_pt2]'s re-labelled row escape uncaught past an [Err.t]
   signature. [throw] records [Detect] where the fault is, and [with_escape]
   catches by construction. *)
let malformed esc (e : malformed) = Err.Escape.throw esc (e :> error)

let shape_of_sizes esc name sizes =
  let dims =
    List.map
      (function
        | SymInt.Int i when i >= 1 -> i
        | SymInt.Int 0 ->
            malformed esc (`Bad_dimension { tensor = name; fault = `Zero })
        | SymInt.Int i ->
            malformed esc
              (`Bad_dimension { tensor = name; fault = `Negative i })
        | SymInt.Expr _ ->
            malformed esc (`Bad_dimension { tensor = name; fault = `Symbolic }))
      sizes
  in
  match List.rev dims with
  | [ c; w; h; d; t; n ] -> Vec6.shape ~n ~t ~d ~h ~w ~c
  | [ c; w; h; d; t ] -> Vec6.shape ~n:1 ~t ~d ~h ~w ~c
  | [ c; w; h; d ] -> Vec6.shape ~n:1 ~t:1 ~d ~h ~w ~c
  | [ c; w; h ] -> Vec6.shape ~n:1 ~t:1 ~d:1 ~h ~w ~c
  | [ c; w ] -> Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w ~c
  | c :: [] -> Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c
  | [] -> Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1
  | _ ->
      malformed esc (`Bad_dimension { tensor = name; fault = `Rank_over_six })

let tensor_shape esc (graph : Pytorch_types.Graph.t) name =
  match String_map.find_opt name graph.tensor_values with
  | Some meta -> shape_of_sizes esc name meta.TensorMeta.sizes
  | None -> malformed esc (`Missing_metadata { ssa = name; role = `Tensor })

let find_arg esc (node : Pytorch_types.Node.t) name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | Some a -> a.arg
  | None -> malformed esc (`Missing_arg { op = node.target; arg = name })

let tensor_name esc (node : Pytorch_types.Node.t) name =
  match find_arg esc node name with
  | Argument.Tensor t -> t.TensorArgument.name
  | _ ->
      malformed esc
        (`Wrong_arg_kind { op = node.target; arg = name; expected = `Tensor })

(* [cat.default]/[stack.default]'s [tensors] argument, the first Tensor[]-typed
   ARGUMENT this module decodes (every earlier [Argument.Tensors] use is on
   the OUTPUT side, e.g. [output_names]). *)
let tensor_names_arg esc (node : Pytorch_types.Node.t) name =
  match find_arg esc node name with
  | Argument.Tensors ts -> List.map (fun (t : TensorArgument.t) -> t.name) ts
  | _ ->
      malformed esc
        (`Wrong_arg_kind
           { op = node.target; arg = name; expected = `Tensor_list })

(* [~absent_ok] distinguishes an argument that is PRESENT and None from one not
   in the node's input list at all. The schema default for every optional tensor
   here is None, and [Op_bridge] already reads omission that way
   ([optional_tensor_present]), so an exact target that refused it would
   disagree with the other importer about the same node — which is the property
   the two paths exist to cross-check.

   Defaulted to [false] so the arms that predate this keep the behaviour their
   goldens pin; they are fed only by real exports, which serialise every
   argument explicitly, and each can revisit it in its own row. *)
let optional_tensor_name ?(absent_ok = false) esc (node : Pytorch_types.Node.t)
    name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | None when absent_ok -> None
  | _ -> (
      match find_arg esc node name with
      | Argument.Tensor t -> Some t.TensorArgument.name
      | Argument.None _ -> None
      | Argument.Optional_tensor (OptionalTensorArgument.Tensor t) ->
          Some t.TensorArgument.name
      | Argument.Optional_tensor (OptionalTensorArgument.None _) -> None
      | _ ->
          malformed esc
            (`Wrong_arg_kind
               { op = node.target; arg = name; expected = `Optional_tensor }))

let sym_int_value esc (node : Pytorch_types.Node.t) name = function
  | SymIntArgument.Int i -> i
  | SymIntArgument.Name symbol ->
      malformed esc
        (`Unresolved_sym_arg
           { Unresolved_sym_arg.op = node.target; arg = name; symbol })

let ints_arg esc ?(default = []) (node : Pytorch_types.Node.t) name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | None -> default
  | Some { arg = Argument.Ints xs; _ } -> xs
  | Some { arg = Argument.Sym_ints xs; _ } ->
      List.map (sym_int_value esc node name) xs
  | Some { arg = Argument.None _; _ } -> default
  | Some _ ->
      malformed esc
        (`Wrong_arg_kind { op = node.target; arg = name; expected = `Int_list })

(* [float[]?]: no [Sym_floats] resolution the way [ints_arg] resolves
   [Sym_ints] -- a schema [float[]] is never symbolic. Absent/[None] both
   collapse to [default], the same convention [ints_arg] uses for e.g.
   `upsample_bilinear2d.vec`'s [scale_factors] when [output_size] is given
   instead. *)
let floats_arg esc ?(default = []) (node : Pytorch_types.Node.t) name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | None -> default
  | Some { arg = Argument.Floats xs; _ } -> xs
  | Some { arg = Argument.None _; _ } -> default
  | Some _ ->
      malformed esc
        (`Wrong_arg_kind
           { op = node.target; arg = name; expected = `Float_list })

(* A resolved [SymInt] is accepted and a NAMED one is refused as an unresolved
   symbol -- the same rule [Interp_decode.sym_int_value] applies on the ATen
   path, and the same one [Bad_dimension]'s [`Symbolic] fault already applied to
   tensor METADATA here. Before [slice.Tensor] no bound op had a [SymInt]
   argument, so an [Argument.Sym_int] reached [`Wrong_arg_kind] whatever it
   carried: a resolved bound was refused for the wrong reason and an unresolved
   one for a reason that did not name the symbol. *)
let int_arg esc ?(default = 0) (node : Pytorch_types.Node.t) name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | None -> default
  | Some { arg = Argument.Int i; _ } -> i
  | Some { arg = Argument.Sym_int sv; _ } -> sym_int_value esc node name sv
  | Some _ ->
      malformed esc
        (`Wrong_arg_kind { op = node.target; arg = name; expected = `Int })

(* An [int?] whose ABSENCE is a distinguishable answer, as [float_opt_arg_opt]
   is for [pad]'s fill: [slice]'s [start]/[end] default to the whole axis, and
   [Aten_shape.resolve_slice] is what knows that. An explicit [Argument.None] is
   the same as an absent argument, which is the schema's own default
   ([SymInt? start=None]) and not a guess. *)
let int_opt_arg_opt esc (node : Pytorch_types.Node.t) name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | None -> None
  | Some { arg = Argument.Int i; _ } -> Some i
  | Some { arg = Argument.Sym_int sv; _ } ->
      Some (sym_int_value esc node name sv)
  | Some { arg = Argument.None _; _ } -> None
  | Some _ ->
      malformed esc
        (`Wrong_arg_kind { op = node.target; arg = name; expected = `Int_opt })

let string_arg esc ~default (node : Pytorch_types.Node.t) name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | None -> default
  | Some { arg = Argument.String s; _ } -> s
  | Some _ ->
      malformed esc
        (`Wrong_arg_kind { op = node.target; arg = name; expected = `String })

let bool_arg esc ?(default = false) (node : Pytorch_types.Node.t) name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | None -> default
  | Some { arg = Argument.Bool b; _ } -> b
  | Some _ ->
      malformed esc
        (`Wrong_arg_kind { op = node.target; arg = name; expected = `Bool })

(* A REQUIRED [float]: no default, so omission is [`Missing_arg] and an explicit
   none is [`Wrong_arg_kind]. Carrying a [?(default = 0.)] here was the same
   mistake in a quieter form than the explicit-none one -- it made "required"
   mean "defaults to zero", so a batch-norm node that simply omitted [eps]
   computed with an epsilon of zero, while [Op_bridge]'s decoder reported the
   argument missing. Anything with a real schema default passes it explicitly. *)
let float_arg esc ?default (node : Pytorch_types.Node.t) name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | None -> (
      match default with
      | Some d -> d
      | None -> malformed esc (`Missing_arg { op = node.target; arg = name }))
  | Some { arg = Argument.Float f; _ } -> f
  | Some _ ->
      malformed esc
        (`Wrong_arg_kind { op = node.target; arg = name; expected = `Float })

(* A [float?]. Omission and an explicit none are the SAME REQUEST -- both are
   serialized, since the generated op-spec path writes [Float_opt None] out as
   [Argument.None] -- and anything else is still refused.

   Separate from [float_arg] rather than an arm added to it. Accepting an
   explicit none for every caller made
   [_native_batch_norm_legit_no_training.default], whose schema has a REQUIRED
   [float eps], silently read a null epsilon as 0. -- a different op, computed
   under the right name. The optionality belongs to the argument, so it belongs
   at the call site. *)
let float_opt_arg esc ~default (node : Pytorch_types.Node.t) name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | None -> default
  | Some { arg = Argument.Float f; _ } -> f
  | Some { arg = Argument.None _; _ } -> default
  | Some _ ->
      malformed esc
        (`Wrong_arg_kind { op = node.target; arg = name; expected = `Float })

(* A [float?] whose ABSENCE is a distinguishable answer rather than a default:
   [aten.pad]'s [value] means 0.0 in constant mode and must be absent (or zero)
   in reflect, so collapsing the two here would erase the distinction the mode
   check needs. Contrast [float_opt_arg] above, which supplies a default because
   its callers have one. *)
let float_opt_arg_opt esc (node : Pytorch_types.Node.t) name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | None -> None
  | Some { arg = Argument.Float f; _ } -> Some f
  | Some { arg = Argument.Int i; _ } -> Some (float_of_int i)
  | Some { arg = Argument.None _; _ } -> None
  | Some _ ->
      malformed esc
        (`Wrong_arg_kind { op = node.target; arg = name; expected = `Float })

(* A schema [Scalar] argument crosses as either an Int or a Float — clamp's
   bounds arrive as `as_int` in MobileNet-v3 and hardtanh's as `as_float` in v2,
   for the same kind of parameter. Mirrors [Interp_decode.scalar_arg] /
   [scalar_opt_arg], which the ATen path decodes with. *)
let scalar_arg esc ~default (node : Pytorch_types.Node.t) name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | None -> default
  | Some { arg = Argument.Int i; _ } -> float_of_int i
  | Some { arg = Argument.Float f; _ } -> f
  | Some { arg = Argument.None _; _ } -> default
  | Some _ ->
      malformed esc
        (`Wrong_arg_kind { op = node.target; arg = name; expected = `Scalar })

let scalar_opt_arg esc (node : Pytorch_types.Node.t) name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | None -> None
  | Some { arg = Argument.Int i; _ } -> Some (float_of_int i)
  | Some { arg = Argument.Float f; _ } -> Some f
  | Some { arg = Argument.None _; _ } -> None
  | Some _ ->
      malformed esc
        (`Wrong_arg_kind
           { op = node.target; arg = name; expected = `Optional_scalar })

(* A REQUIRED schema [Scalar]: no default, so omission is [`Missing_arg] and an
   explicit none (or any other kind) is [`Wrong_arg_kind]. [mul.Scalar]'s
   [other] has no schema default, so reusing [scalar_arg]'s [~default] -- which
   treats both omission and an explicit none as the default -- would silently
   read a missing multiplier as the caller's placeholder value; the same
   mistake [float_arg]'s comment documents for batch-norm's [eps]. *)
let required_scalar_arg esc (node : Pytorch_types.Node.t) name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | None -> malformed esc (`Missing_arg { op = node.target; arg = name })
  | Some { arg = Argument.Int i; _ } -> float_of_int i
  | Some { arg = Argument.Float f; _ } -> f
  | Some _ ->
      malformed esc
        (`Wrong_arg_kind { op = node.target; arg = name; expected = `Scalar })

(* [add.Tensor]/[sub.Tensor] carry `*, Scalar alpha=1` and compute
   [self + alpha * other]. Nothing in this model zoo serialises a non-default
   alpha, so it is not implemented — but it must not be silently dropped either,
   since that would quietly compute the wrong thing. Reject instead. *)
let reject_alpha esc (node : Pytorch_types.Node.t) =
  match scalar_opt_arg esc node "alpha" with
  | None -> ()
  | Some a when Float.equal a 1. -> ()
  | Some a ->
      malformed esc
        (`Unsupported_option { op = node.target; option = `Alpha a })

(* [clone]'s [memory_format]. Native's engine has exactly one physical layout
   per shape -- there is no channels-last stride concept anywhere in the
   six-axis frame -- so a request to make the result CONTIGUOUS, or to
   PRESERVE whatever format the input already has, is always already true:
   every native tensor is already in that one dense layout. (Confirmed
   against the corpus: every `clone.default` occurrence across the 100-model
   sweep requests either no format or exactly `ContiguousFormat` -- see
   `.ai/pt2_model_support.md`.) A request for an actual different physical
   arrangement (channels-last) asks for something this engine cannot
   represent, so it is refused rather than silently treated as a no-op. *)
let check_clone_memory_format esc (node : Pytorch_types.Node.t) =
  match
    List.find_opt
      (fun (a : NamedArgument.t) -> a.name = "memory_format")
      node.Node.inputs
  with
  | None | Some { arg = Argument.None _; _ } -> ()
  | Some { arg = Argument.Memory_format mf; _ } -> (
      match mf with
      | MemoryFormat.ContiguousFormat | MemoryFormat.PreserveFormat -> ()
      | MemoryFormat.ChannelsLast ->
          malformed esc
            (`Unsupported_option
               { op = node.target; option = `Memory_format `Channels_last })
      | MemoryFormat.ChannelsLast3d ->
          malformed esc
            (`Unsupported_option
               { op = node.target; option = `Memory_format `Channels_last_3d })
      | MemoryFormat.Unknown ->
          malformed esc
            (`Unsupported_option
               { op = node.target; option = `Memory_format `Unknown }))
  | Some _ ->
      malformed esc
        (`Wrong_arg_kind
           {
             op = node.target;
             arg = "memory_format";
             expected = `Memory_format_opt;
           })

(* [linalg_vector_norm.default]'s [dtype] casts before reducing; the native IR
   has no dtype-conversion op, so honouring the request is impossible and
   ignoring it would misreport what was computed -- same reasoning as
   [reject_memory_format] above, for a different argument. *)
let reject_dtype esc (node : Pytorch_types.Node.t) =
  match
    List.find_opt
      (fun (a : NamedArgument.t) -> a.name = "dtype")
      node.Node.inputs
  with
  | None | Some { arg = Argument.None _; _ } -> ()
  | Some _ ->
      malformed esc (`Unsupported_option { op = node.target; option = `Dtype })

(* A `Tensor[]` return is ONE output of kind [Argument.Tensors] holding every
   result name in order — a different shape from a fixed tuple, whose elements
   are separate [Argument.Tensor] entries. Both flatten to a name list here.

   THE CEILING LIVES IN THIS FUNCTION, not in the operator arm that wants it.
   [lower_node] calls [materialized_output_names] before it calls [lower_op], so
   an arm-local preflight would run after the first [List.map] had already built
   a list sized by model data. [take_bounded] therefore counts as it walks and
   stops AT the limit, never learning the real length — which is exactly why
   [Shape_error.Output_count] distinguishes [At_least] from [Exact].

   The rule is [>= limit], matching [Kernel.Limits.create] and
   [Split.Unbind.output_shapes]: 4095 names are accepted, 4096 are not. *)
let output_limit = Kernel.Limits.Hard.outputs

(* The rule is [>= output_limit], so the allowance is one less than the limit:
   4095 names are accepted, the 4096th is refused. Carried as a remaining
   budget rather than a running total so the traversal can stop without ever
   holding a length. *)
let output_allowance = output_limit - 1

let over_limit esc =
  Err.Escape.throw esc
    (`Output_count_over_limit
       {
         Shape_error.Output_count.limit = output_limit;
         observed = Shape_error.Output_count.At_least output_limit;
       })

(* [split.Tensor(self, split_size, dim)]'s chunk-size list -- the equal-
   chunk-size sibling of [split_with_sizes.default], which legalizes onto the
   *existing* [Split.Split_with_sizes] node the same way ([chunk_sizes] in
   [Op_bridge_decode] derives the identical sizes list). Unlike that ATen-
   linked path, [extent] here is a METADATA-only declared size with no real
   tensor backing it, so a small [extent]/[split_size] pair can name an
   arbitrarily large chunk count for free -- the same resource-preflight
   reasoning [output_allowance]'s own header gives for
   [materialized_output_names]: the count must be bounded BEFORE the list is
   built, not after. *)
let split_tensor_sizes esc ~extent ~split_size =
  let full = extent / split_size in
  let remainder = extent - (full * split_size) in
  let count = full + if remainder > 0 then 1 else 0 in
  if count >= output_limit then
    Err.Escape.throw esc
      (`Output_count_over_limit
         {
           Shape_error.Output_count.limit = output_limit;
           observed = Shape_error.Output_count.Exact count;
         })
  else
    let sizes = List.init full (fun _ -> split_size) in
    if remainder > 0 then sizes @ [ remainder ] else sizes

(* Prepend [xs]'s names to [acc] while [budget] lasts; throws on the element
   that would reach the limit. Counting rather than [List.length]-then-check is
   the point: the list may be arbitrarily long, and this never walks past the
   ceiling. *)
let rec take_bounded esc ~budget acc = function
  | [] -> (acc, budget)
  | (t : TensorArgument.t) :: rest ->
      if budget <= 0 then over_limit esc
      else
        take_bounded esc ~budget:(budget - 1)
          (t.TensorArgument.name :: acc)
          rest

(* Flatten one argument's tensor names, threading the remaining budget so that
   SEVERAL list-valued arguments are bounded in aggregate rather than each on
   its own — several individually legal lists can exceed the ceiling together. *)
let flatten_output esc ~on_bad_kind ~budget acc (a : Argument.t) =
  match a with
  | Argument.Tensor t ->
      if budget <= 0 then over_limit esc
      else (t.TensorArgument.name :: acc, budget - 1)
  | Argument.Tensors ts -> take_bounded esc ~budget acc ts
  | _ -> on_bad_kind ()

let flatten_outputs esc ~on_bad_kind args =
  let names, _ =
    List.fold_left
      (fun (acc, budget) a -> flatten_output esc ~on_bad_kind ~budget acc a)
      ([], output_allowance) args
  in
  List.rev names

let output_names esc (node : Pytorch_types.Node.t) =
  flatten_outputs esc
    ~on_bad_kind:(fun () -> malformed esc (`Non_tensor_node_output node.target))
    node.outputs

let is_nontrivial_node (node : Pytorch_types.Node.t) =
  match node.target with
  | "torch.ops.aten.conv2d.default" | "torch.ops.aten.conv2d.padding"
  | "torch.ops.aten.convolution.default" | "torch.ops.aten.linear.default"
  | "torch.ops.aten._native_batch_norm_legit_no_training.default"
  | "torch.ops.aten.max_pool2d.default" | "torch.ops.aten.avg_pool2d.default"
  | "torch.ops.aten.adaptive_avg_pool2d.default"
  | "torch.ops.aten.max_pool2d_with_indices.default"
  | "torch.ops.aten.rms_norm.default" | "torch.ops.aten.layer_norm.default"
  | "torch.ops.aten.native_layer_norm.default" | "torch.ops.aten.addmm.default"
  | "torch.ops.aten.scaled_dot_product_attention.default"
  | "torch.ops.aten.upsample_bilinear2d.vec"
  | "torch.ops.aten.upsample_nearest2d.vec" ->
      true
  | _ -> false

let materialized_output_names esc (node : Pytorch_types.Node.t) =
  match node.target with
  | "torch.ops.aten._native_batch_norm_legit_no_training.default"
  | "torch.ops.aten.max_pool2d_with_indices.default"
  (* Third entry, and the first whose dropped outputs are NOT empty: they are
     real f32 tensors that happen to be dead in every occurrence the corpus
     contains. Dropping them here is what makes the [`Live_layer_norm_stats]
     check below load-bearing rather than decorative. *)
  | "torch.ops.aten.native_layer_norm.default" ->
      [ List.hd (output_names esc node) ]
  | _ -> output_names esc node

let hw2 esc param = function
  | [ h; w ] -> (h, w)
  | [ x ] -> (x, x)
  | xs -> malformed esc (`Bad_arity { param; got = List.length xs })

(* [Op_config.Pos]/[Nonneg]/[Dim.extent] assert a TRUSTED precondition and
   raise [Invalid_argument] when it fails. Every value below is decoded from the
   model, so none of them may reach those constructors unguarded: the raise
   crosses the [Err.Escape] frame and leaves [lower] as an exception, which is
   what [malformed_test.ml]'s three config witnesses pinned.

   These are the ONLY approved route from a decoded argument to a guarded
   config type in this module. [Dim.extent_checked] already existed for exactly
   this ([dim.mli]: "the validated form for an untrusted size"); the other two
   have no checked form, so the test is written out here. *)
let pos esc ~op ~param n =
  match Op_config.Bad.pos ~op ~param n with
  | Error e -> malformed esc (`Bad_config e)
  | Ok v -> v

let nonneg esc ~op ~param n =
  match Op_config.Bad.nonneg ~op ~param n with
  | Error e -> malformed esc (`Bad_config e)
  | Ok v -> v

let extent esc ~op ~param n =
  match Dim.extent_checked n with
  | Error _ ->
      malformed esc
        (`Bad_config { Op_config.Bad.op; param; fault = `Not_positive n })
  | Ok e -> e

(* The same asserting constructor reached from tensor METADATA rather than from
   an op-configuration field, so it gets the row that already describes that:
   [Bad_dimension]'s [`Zero] and [`Negative] faults, which the module documents
   as distinct because only [`Zero] arrives from real models. *)
let dim_extent esc ~tensor n =
  match Dim.extent_checked n with
  | Error _ ->
      malformed esc
        (`Bad_dimension
           { tensor; fault = (if n = 0 then `Zero else `Negative n) })
  | Ok e -> e

let pos_hw esc ~op ~param (h, w) =
  { Op_config.Hw.h = pos esc ~op ~param h; w = pos esc ~op ~param w }

let nonneg_hw esc ~op ~param (h, w) =
  { Op_config.Hw.h = nonneg esc ~op ~param h; w = nonneg esc ~op ~param w }

let env_find esc env name =
  match String_map.find_opt name env with
  | Some x -> x
  | None -> malformed esc (`Undefined_ssa name)

let add_env env names ids =
  (* An INVARIANT of this module, not a fact about the model: [names] comes
     from [materialized_output_names] and [ids] from the op call just made, so a
     mismatch is a defect here. It must not reach the caller dressed as a
     malformed graph. Same treatment as [Graph_ir.Index.assert_matches]. *)
  if List.compare_lengths names ids <> 0 then
    invalid_arg
      "Native_interp.add_env: output arity does not match the ids produced";
  List.fold_left2 (fun e name id -> String_map.add name id e) env names ids

let perm_nchw_to_nhwc =
  let open Axis in
  [ (N, N); (T, T); (D, D); (H, W); (W, C); (C, H) ]

let perm_nhwc_to_nchw =
  let open Axis in
  [ (N, N); (T, T); (D, D); (H, C); (W, H); (C, W) ]

let perm_oihw_to_conv_weight =
  let open Axis in
  [ (N, D); (T, T); (D, N); (H, W); (W, C); (C, H) ]

(* Rank-2 addmm weight [In,Out] (W=In, C=Out) -> native [N=Out, C=In]. *)
let perm_addmm_weight =
  let open Axis in
  [ (N, C); (T, T); (D, D); (H, H); (W, N); (C, W) ]

(* Rank-2 linear weight [Out,In] (W=Out, C=In) -> native [N=Out, C=In]. NOT the
   permutation above, and not a rename of it: `addmm`'s [mat2] is the transpose
   of `linear`'s [weight], so an arm that reused one for the other would build a
   weight whose output and input axes are swapped. Both spellings exist in
   [Op_bridge] (op_bridge.ml:221,227) for the same reason. *)
let perm_linear_weight =
  let open Axis in
  [ (N, W); (T, T); (D, D); (H, H); (W, N); (C, C) ]

(* The [tensor_values] lookup, open-coded at five sites with the same three
   steps and a different role label each. Three functions rather than one
   because the sites want different depths: [mean.dim], [permute.default] and
   [unbind.int] need only the RANK, which a symbolic dimension does not
   prevent, while a conv weight needs the extents themselves.

   [role] stays a parameter so each caller keeps its own diagnostic. Sharing one
   role across two arms would make the row ambiguous about which one failed,
   which is the property that made these worth typing in the first place. *)
let tensor_meta esc (graph : Pytorch_types.Graph.t) ~ssa ~role =
  match String_map.find_opt ssa graph.tensor_values with
  | Some x -> x
  | None -> malformed esc (`Missing_metadata { ssa; role })

let meta_rank (meta : TensorMeta.t) = List.length meta.TensorMeta.sizes

(* [shape_of_sizes] RIGHT-ALIGNS a declared size list into the six-axis frame,
   so [C] and [1,C] land on exactly the same extents. [Graph_shape]'s operand
   check compares those frames and therefore cannot tell the two apart -- but
   ATen can, and refuses a bias that is not 1-D. The declared RANK exists only
   on this side of the conversion, so no shared native rule can cover it and
   each importer has to check its own. *)
let require_rank esc (graph : Pytorch_types.Graph.t) ~ssa ~role ~expected =
  let got = meta_rank (tensor_meta esc graph ~ssa ~role) in
  if got <> expected then
    malformed esc
      (`Bad_dimension { tensor = ssa; fault = `Expected_rank { expected; got } })

let static_sizes esc ~tensor (meta : TensorMeta.t) =
  List.map
    (function
      | SymInt.Int i -> i
      | SymInt.Expr _ ->
          malformed esc (`Bad_dimension { tensor; fault = `Symbolic }))
    meta.TensorMeta.sizes

let sizes_rank_4 esc ~tensor = function
  | [ a; b; c; d ] -> (a, b, c, d)
  | sizes ->
      malformed esc
        (`Bad_dimension
           {
             tensor;
             fault = `Expected_rank { expected = 4; got = List.length sizes };
           })

let sizes_rank_2 esc ~tensor = function
  | [ a; b ] -> (a, b)
  | sizes ->
      malformed esc
        (`Bad_dimension
           {
             tensor;
             fault = `Expected_rank { expected = 2; got = List.length sizes };
           })

(* [Conv2d.params.in_channels] is the ACTIVATION's channel count: the weight's
   per-group input extent times the group count. Same rule as
   [Op_bridge.make_conv2d_params] and [Conv.Conv2d_padding.to_conv2d_params],
   restated here only because the importer reads serialized metadata where those
   read a live tensor.

   COMPUTED IN int64 AND BOUNDED BEFORE NARROWING, which is the whole point.
   js_of_ocaml's [int] is 32 bits and [Kernel.Limits.Hard.extent] is
   0x8000_0000, so two individually plausible factors can multiply past the
   representable range and WRAP to a small positive number — a silently wrong
   graph rather than a rejected one. Bounding the factors would not catch it:
   the product is its own quantity. [test/native_interp] runs under node
   ([modes best js]), so this is reachable and not a theoretical concern. *)
let conv_in_channels esc ~tensor ~cin ~groups =
  (* Each FACTOR is bounded before the multiplication, not only the product.
     Widening to [int64] does not by itself make the multiplication safe: these
     are raw decoded ints, and on a 63-bit-[int] backend two factors near
     2^62 overflow [Int64.mul] silently and land back inside the range the
     check below accepts. The bound is unobservable under js_of_ocaml, where a
     factor that large is not representable in a 32-bit [int] at all -- which is
     exactly why it cannot be left to the product check. *)
  let over n = Int64.of_int n >= Kernel.Limits.Hard.extent in
  if over cin || over groups then
    malformed esc
      (`Bad_dimension
         {
           tensor;
           fault =
             `Over_max_extent
               (if over cin then Int64.of_int cin else Int64.of_int groups);
         });
  let product = Int64.mul (Int64.of_int cin) (Int64.of_int groups) in
  if product >= Kernel.Limits.Hard.extent then
    malformed esc (`Bad_dimension { tensor; fault = `Over_max_extent product })
  else if product < 1L then
    malformed esc
      (`Bad_dimension
         { tensor; fault = (if product = 0L then `Zero else `Negative cin) })
  else Dim.extent (Int64.to_int product)

(* The exact `conv2d.default` overload, whose [params] record is NOT
   [Convolution]'s: per-axis windows carrying their own kernel extent, an
   activation channel count, and no transposed/output_padding fields. Sharing
   the metadata and H/W decoding with [conv_params] while keeping the two
   records apart is what lets the exact IR node survive to Native4D, which reads
   [Conv2d] directly. *)
let conv2d_params esc (graph : Pytorch_types.Graph.t)
    (node : Pytorch_types.Node.t) =
  let op = node.Node.target in
  let weight_name = tensor_name esc node "weight" in
  let sizes =
    static_sizes esc ~tensor:weight_name
      (tensor_meta esc graph ~ssa:weight_name ~role:`Conv2d_weight)
  in
  let _cout, cin, kh, kw = sizes_rank_4 esc ~tensor:weight_name sizes in
  let sh, sw = hw2 esc `Stride (ints_arg esc ~default:[ 1; 1 ] node "stride") in
  let ph, pw =
    hw2 esc `Padding (ints_arg esc ~default:[ 0; 0 ] node "padding")
  in
  let dh, dw =
    hw2 esc `Dilation (ints_arg esc ~default:[ 1; 1 ] node "dilation")
  in
  let groups = int_arg esc ~default:1 node "groups" in
  (* Serialized integer padding is SYMMETRIC — ATen pads both sides of an axis
     equally — so both fields take the one value. The asymmetric form exists for
     [Conv2d_padding]'s "same", which splits an odd total unevenly. *)
  let axis ~kernel ~stride ~pad ~dilation : Conv.Conv2d.axis_window =
    {
      kernel = extent esc ~op ~param:`Kernel_size kernel;
      stride = pos esc ~op ~param:`Stride stride;
      pad_before = nonneg esc ~op ~param:`Padding pad;
      pad_after = nonneg esc ~op ~param:`Padding pad;
      dilation = pos esc ~op ~param:`Dilation dilation;
    }
  in
  {
    Conv.Conv2d.h = axis ~kernel:kh ~stride:sh ~pad:ph ~dilation:dh;
    w = axis ~kernel:kw ~stride:sw ~pad:pw ~dilation:dw;
    in_channels = conv_in_channels esc ~tensor:weight_name ~cin ~groups;
    groups = pos esc ~op ~param:`Groups groups;
  }

(* The overload whose padding is a MODE rather than a number. Its [params] carry
   neither a kernel extent nor a channel count: [Conv2d_padding.to_conv2d_params]
   derives both from the weight's shape, and that one definition is already
   shared by shape inference, [Compute] and Native4D. Resolving the mode here
   would make a fourth. *)
let conv2d_padding_params esc (graph : Pytorch_types.Graph.t)
    (node : Pytorch_types.Node.t) =
  let op = node.Node.target in
  let weight_name = tensor_name esc node "weight" in
  (* Read for its RANK alone -- the extents are shape inference's business here.
     Checked all the same, so the two conv2d arms accept the same weights: a
     rank-3 weight would otherwise be right-aligned into the six-axis frame and
     relayouted as though its leading axis were the output channel. *)
  let sizes =
    static_sizes esc ~tensor:weight_name
      (tensor_meta esc graph ~ssa:weight_name ~role:`Conv2d_padding_weight)
  in
  let _ = sizes_rank_4 esc ~tensor:weight_name sizes in
  let padding =
    (* NOT [Conv.Conv2d_padding.padding_of_string], which [invalid_arg]s on
       anything else (conv.ml:394). This string is model data, so it gets a
       typed row -- the same rule the guarded config constructors follow. *)
    match
      Conv.Conv2d_padding.of_string
        (string_arg esc ~default:"valid" node "padding")
    with
    | Error s -> malformed esc (`Unsupported_padding_mode s)
    | Ok p -> p
  in
  let stride = hw2 esc `Stride (ints_arg esc ~default:[ 1; 1 ] node "stride") in
  let dilation =
    hw2 esc `Dilation (ints_arg esc ~default:[ 1; 1 ] node "dilation")
  in
  {
    Conv.Conv2d_padding.stride = pos_hw esc ~op ~param:`Stride stride;
    padding;
    dilation = pos_hw esc ~op ~param:`Dilation dilation;
    groups = pos esc ~op ~param:`Groups (int_arg esc ~default:1 node "groups");
  }

let conv_params esc (graph : Pytorch_types.Graph.t)
    (node : Pytorch_types.Node.t) =
  let weight_name = tensor_name esc node "weight" in
  let sizes =
    static_sizes esc ~tensor:weight_name
      (tensor_meta esc graph ~ssa:weight_name ~role:`Convolution_weight)
  in
  let cout, cin, kh, kw = sizes_rank_4 esc ~tensor:weight_name sizes in
  let _ = cout in
  let op = node.Node.target in
  let stride = hw2 esc `Stride (ints_arg esc ~default:[ 1; 1 ] node "stride") in
  let padding =
    hw2 esc `Padding (ints_arg esc ~default:[ 0; 0 ] node "padding")
  in
  let dilation =
    hw2 esc `Dilation (ints_arg esc ~default:[ 1; 1 ] node "dilation")
  in
  let groups = int_arg esc ~default:1 node "groups" in
  (* [output_padding] IS an argument of this overload -- the schema is
     convolution(..., bool transposed, SymInt[] output_padding, int groups) --
     and forcing it to zero was wrong in both directions. A transposed
     convolution's output extent includes it, so a nonzero value built a smaller
     op than the model asked for; and a NON-transposed one with a nonzero value
     is invalid, which [Convolution.output_shape] rejects (conv.ml:833) and
     discarding the argument let through. [Op_bridge] has always decoded it, so
     the two importers built different ops from one node. *)
  let output_padding =
    hw2 esc `Output_padding
      (ints_arg esc ~default:[ 0; 0 ] node "output_padding")
  in
  ( {
      Conv.Convolution.stride = pos_hw esc ~op ~param:`Stride stride;
      padding = nonneg_hw esc ~op ~param:`Padding padding;
      dilation = pos_hw esc ~op ~param:`Dilation dilation;
      transposed = bool_arg esc node "transposed";
      output_padding = nonneg_hw esc ~op ~param:`Output_padding output_padding;
      groups = pos esc ~op ~param:`Groups groups;
    },
    cin,
    kh,
    kw )

(* Shared by [pool_params] and [avg_pool_params]: both max- and avg-pool's
   [params] carry the same kernel/stride/pad shape. *)
let pool_window_fields esc (node : Pytorch_types.Node.t) =
  let op = node.Node.target in
  let kh, kw = hw2 esc `Kernel_size (ints_arg esc node "kernel_size") in
  (* Validated BEFORE the stride is defaulted from it. The default makes the two
     the same value, so a kernel of 0 reached the stride's check first and was
     reported as a bad stride -- a diagnostic naming an argument the model never
     supplied. *)
  let kernel =
    {
      Op_config.Hw.h = extent esc ~op ~param:`Kernel_size kh;
      w = extent esc ~op ~param:`Kernel_size kw;
    }
  in
  (* An EMPTY stride list means "same as the kernel" and is a different
     spelling from the argument being absent, so both normalize here. Same rule
     as [Op_bridge.pool_stride] (op_bridge.ml:371). *)
  let stride =
    match ints_arg esc ~default:[ kh; kw ] node "stride" with
    | [] -> [ kh; kw ]
    | s -> s
  in
  let stride = hw2 esc `Stride stride in
  let padding =
    hw2 esc `Padding (ints_arg esc ~default:[ 0; 0 ] node "padding")
  in
  ( kernel,
    pos_hw esc ~op ~param:`Stride stride,
    nonneg_hw esc ~op ~param:`Padding padding )

(* [max_pool2d.default]/[max_pool2d_with_indices.default]'s own [dilation]:
   REJECTED, not carried -- [Pool.MaxPool2d.params] has no field for it, so a
   non-default value would compute a different op under the right name.
   Extending the native IR is warranted only on a measured need, and no model
   this repository can download serialises either pooling target with a
   non-default dilation, so there is nothing to measure and a rejection is the
   honest answer. [avg_pool2d.default] has no [dilation] argument at all (see
   [avg_pool_params]), so it needs no analogous check. *)
let pool_params esc (node : Pytorch_types.Node.t) =
  let op = node.Node.target in
  let kernel, stride, pad = pool_window_fields esc node in
  (* NORMALIZED FIRST, then compared against the only value the params can hold.
     Testing "does some element differ from 1" accepted [] and [1;1;1] as well
     as [1;1] -- and ATen refuses both ("dilation must be either a single int,
     or a tuple of two ints"), so the arity check every other H/W argument gets
     was the one thing standing between a refused node and a silent drop. *)
  let dilation = ints_arg esc ~default:[ 1; 1 ] node "dilation" in
  (match hw2 esc `Dilation dilation with
  | 1, 1 -> ()
  | _ -> malformed esc (`Unsupported_option { op; option = `Dilation dilation }));
  let ceil_mode = bool_arg esc ~default:false node "ceil_mode" in
  { Pool.MaxPool2d.ceil_mode; kernel; stride; pad }

(* [avg_pool2d.default] carries no [dilation] argument at all (unlike the
   max-pool overloads) but does carry [count_include_pad] (represented, see
   [Pool.AvgPool2d.params.count_include_pad]) and [divisor_override] (has no
   field to hold a non-default value, so a present one is refused). *)
let avg_pool_params esc (node : Pytorch_types.Node.t) =
  let op = node.Node.target in
  let kernel, stride, pad = pool_window_fields esc node in
  (match int_opt_arg_opt esc node "divisor_override" with
  | None -> ()
  | Some d ->
      malformed esc (`Unsupported_option { op; option = `Divisor_override d }));
  let ceil_mode = bool_arg esc ~default:false node "ceil_mode" in
  let count_include_pad = bool_arg esc ~default:true node "count_include_pad" in
  { Pool.AvgPool2d.ceil_mode; count_include_pad; kernel; stride; pad }

(* [used] is the innermost [rank] frame axes, so it has SIX entries once rank
   exceeds six — and then [d >= rank] admits d = 6 and [List.nth] raises
   [Failure "nth"]. The rank comes from a node's [tensor_values] metadata, which
   is untrusted model data and is NOT covered by [shape_of_sizes]'s own
   rank check: that one runs over graph inputs and captured tensors, not over an
   edge some node produced. Guarding here covers every caller
   (mean.dim, permute.default, unbind.int) rather than each arm separately, and
   reports the same row [shape_of_sizes] would for the same condition. *)
let used_axes_for esc ~tensor rank =
  if rank > 6 then
    malformed esc (`Bad_dimension { tensor; fault = `Rank_over_six })
  else List.filteri (fun i _ -> i >= 6 - rank) Axis.all

let axes_for_rank esc ~tensor rank dims =
  let used = used_axes_for esc ~tensor rank in
  List.map
    (fun d ->
      let d = if d < 0 then d + rank else d in
      if d < 0 || d >= rank then
        malformed esc (`Axis_out_of_range { axis = d; rank })
      else List.nth used d)
    dims

let native_perm esc ~tensor ~rank dims =
  let used = used_axes_for esc ~tensor rank in
  let outer = List.filter (fun a -> not (List.mem a used)) Axis.all in
  List.map (fun a -> (a, a)) outer
  @ List.mapi
      (fun i d ->
        let d = if d < 0 then d + rank else d in
        if d < 0 || d >= rank then
          malformed esc (`Axis_out_of_range { axis = d; rank });
        (List.nth used i, List.nth used d))
      dims

(* Shares [Aten_shape.resolve_view_size] with [Op_bridge] rather than
   re-deriving the [-1] convention: op3-impl.md F1 found this resolver
   accepted an invalid target silently (two [-1]s, a numel mismatch, a
   non-divisible inference) and F8 found its diagnostic named a tensor called
   "view" that never existed. Composed through [Err.Escape.or_throw], which
   exists precisely so a recursive walk can call an ordinary result-returning
   function without threading results through its own arms
   ([conv_in_channels] above is the same pattern: a bounded [int64] count
   inside the escape walk, reported as a typed row). *)
let resolve_view esc ~tensor shape size =
  let bad_view fault : error = `Bad_view { Bad_view.size; fault } in
  let numel =
    Err.Escape.or_throw esc
      (Err.map_error bad_view
         (Vec6.numel_bounded ~limit:Kernel.Limits.Hard.numel shape))
  in
  let resolved =
    Err.Escape.or_throw esc
      (Err.map_error
         (fun e -> bad_view (`Aten_shape e))
         (Aten_shape.resolve_view_size ~numel size))
  in
  shape_of_sizes esc tensor (List.map (fun x -> SymInt.Int x) resolved)

(* Shared by [upsample_bilinear2d.vec]/[upsample_nearest2d.vec]'s arms: both
   schemas are `(Tensor input, SymInt[]? output_size, ..., float[]?
   scale_factors)`, and ATen's own `compute_output_size` accepts exactly one
   of the two, never both, never neither. Same resolution [Op_bridge]'s
   [resolve_upsample_size] performs, restated here only because this importer
   reads serialized metadata where that one reads a live tensor. [op] is
   [node.target] (the FULL name), matching [Bad_upsample_size]'s own field. *)
let resolve_upsample_size esc ~op ~in_h ~in_w output_size scale_factors =
  match (output_size, scale_factors) with
  | [ h; w ], [] -> (h, w)
  | [], [ sh; sw ] ->
      ( int_of_float (float_of_int in_h *. sh),
        int_of_float (float_of_int in_w *. sw) )
  | [], [] ->
      malformed esc
        (`Bad_upsample_size
           { Bad_upsample_size.op; fault = Bad_upsample_size.Neither })
  | (_ :: _ :: _ | [ _ ]), [] ->
      malformed esc
        (`Bad_arity
           { Bad_arity.param = `Output_size; got = List.length output_size })
  | [], _ ->
      malformed esc
        (`Bad_upsample_size
           {
             Bad_upsample_size.op;
             fault =
               Bad_upsample_size.Bad_scale_arity (List.length scale_factors);
           })
  | _ :: _, _ :: _ ->
      malformed esc
        (`Bad_upsample_size
           { Bad_upsample_size.op; fault = Bad_upsample_size.Both })

(* The shared resolver, wrapped in this module's own row. Beside [resolve_view]
   and for its reason: the arm that calls it runs inside the builder monad,
   where the ambient error type is [Graph_builder.error], so the widening has to
   happen out here where [error] is what a row can be. *)
let resolve_slice_arg esc ~extent ~start ~stop ~step =
  Err.Escape.or_throw esc
    (Err.map_error
       (fun e : error ->
         `Bad_slice { Bad_slice.start; stop; step; fault = `Aten_shape e })
       (Aten_shape.resolve_slice ~extent ~start ~stop ~step))

(* [aten.select.int]'s index, shared with [Op_bridge] the same way
   [resolve_slice_arg] shares [Aten_shape.resolve_slice]: ATen REJECTS an
   out-of-range index rather than clamping it, so this cannot reuse
   [resolve_slice_arg]'s bound. *)
let resolve_select_index esc ~extent ~index =
  Err.Escape.or_throw esc
    (Err.map_error
       (fun e : error ->
         `Bad_select { Bad_select.index; fault = `Aten_shape e })
       (Aten_shape.resolve_index ~extent ~index))
