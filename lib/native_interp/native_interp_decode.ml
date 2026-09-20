(* Argument decoding and op-configuration helpers for [Native_interp.lower].
   Split from native_interp.ml. Entirely internal: none of this is exposed
   through native_interp.mli, and [lower] (still in native_interp.ml) is the
   sole caller. Every helper takes its [Err.Escape.t] token explicitly,
   exactly as before the split, so moving this code changes nothing about
   control flow -- only its file.

   Permutation constants and shape/rank-resolution helpers (tensor_meta,
   resolve_repeat, resolve_tile, ...) live in native_interp_decode_shape.ml,
   split out once this file crossed the tracked 1000-line ceiling
   (scripts/check-file-size.sh) -- the same split conv2d/pool's own
   parameter builders already got in native_interp_decode_conv.ml. *)

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

(* A captured [Parameter]/[Buffer]/[Tensor_constant]'s declared element
   format, read from its own SSA metadata rather than defaulted to the
   graph's F32 compute format: an archive constant genuinely materializes as
   LONG (e.g. an [index.Tensor] gather index kept off a wrapping
   [clone.default]'s always-F32 edge, `.ai/index_tensor_design.md` round 6),
   and [Rewrite.origin]'s payload check would otherwise reject every such
   constant's real int64 bytes against a wrongly-declared F32 signature.
   Every other dtype (including one an op-specific check downstream, e.g.
   [Index_list.Wrong_dtype], rejects with a more specific message) keeps the
   engine's F32 default -- this is a signature fix for the one dtype the
   engine actually models, not a general dtype validator. BOOL is now modelled
   too: a captured Bool constant keeps its Bool format (its bytes are read as
   logical values and stored canonical, see [tensor_of_pt2]). *)
let tensor_fmt (graph : Pytorch_types.Graph.t) name =
  match String_map.find_opt name graph.tensor_values with
  | Some { TensorMeta.dtype = ScalarType.BOOL; _ } -> Payload.Fmt Payload.Bool
  | Some { TensorMeta.dtype = ScalarType.LONG; _ } -> Payload.Fmt Payload.I64
  | Some _ | None -> Payload.Fmt Payload.F32

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

(* [index.Tensor]'s [indices : Tensor?[]] -- a list mixing live-entry names
   with explicit [None]s, kept raw (not resolved to native ids) since the
   caller must inspect each live entry's own name before binding, to trace
   past a wrapping [clone.default] (see [Native_interp_lower_compute]).

   A [Tensor?[]]-typed argument with EVERY entry live is exported as a plain
   [Argument.Tensors] rather than [Argument.Optional_tensors] --
   `mvitv2_tiny`/`maxxvitv2_nano_rw_256`'s own `indices`
   (`.ai/index_tensor_design.md`) use exactly this encoding. Accepted here
   the same as the mixed form, each name wrapped [Some]. *)
let optional_tensor_names_arg esc (node : Pytorch_types.Node.t) name =
  match find_arg esc node name with
  | Argument.Optional_tensors ts ->
      List.map
        (function
          | OptionalTensorArgument.Tensor (t : TensorArgument.t) -> Some t.name
          | OptionalTensorArgument.None _ -> None)
        ts
  | Argument.Tensors ts ->
      List.map (fun (t : TensorArgument.t) -> Some t.name) ts
  | _ ->
      malformed esc
        (`Wrong_arg_kind
           { op = node.target; arg = name; expected = `Optional_tensor_list })

(* [index.Tensor]'s trace-past-Clone rule (`.ai/index_tensor_design.md` round
   6): an ATen [clone.default] with no format change is a value- AND
   dtype-preserving identity, so if the live [indices] entry is exactly a
   plain clone of a captured/constant tensor, bind the ORIGINAL pre-clone
   name instead of Clone's own output. Never a stage-walk and never applied
   to a computed (non-constant) input -- only a directly-bound captured
   constant can hold a non-F32 dtype in this engine at all (every computed
   intermediate is F32), so tracing past anything else would have nothing to
   gain. This is what keeps [Index_tensor]'s [index] operand off Clone's own
   native edge, whose signature [Graph_builder.op1] defaults to F32
   regardless of the source dtype -- a pre-existing, unrelated characteristic
   of every [clone.default] occurrence in this codebase, not something this
   op fixes generally. *)
let node_produces (node : Pytorch_types.Node.t) name =
  List.exists
    (function
      | Argument.Tensor (t : TensorArgument.t) -> String.equal t.name name
      | Argument.Tensors ts ->
          List.exists
            (fun (t : TensorArgument.t) -> String.equal t.name name)
            ts
      | _ -> false)
    node.Node.outputs

let clone_no_format_change (node : Pytorch_types.Node.t) =
  match
    List.find_opt
      (fun (a : NamedArgument.t) -> a.name = "memory_format")
      node.Node.inputs
  with
  | None -> true
  | Some { NamedArgument.arg = Argument.None _; _ } -> true
  | Some { NamedArgument.arg = Argument.Memory_format mf; _ } -> (
      match mf with
      | MemoryFormat.ContiguousFormat | MemoryFormat.PreserveFormat -> true
      | MemoryFormat.ChannelsLast | MemoryFormat.ChannelsLast3d
      | MemoryFormat.Unknown ->
          false)
  | Some _ -> false

let resolve_index_source (graph : Pytorch_types.Graph.t) ~constant_names
    (live_name : string) =
  match
    List.find_opt (fun n -> node_produces n live_name) graph.Graph.nodes
  with
  | Some producer
    when String.equal producer.Node.target "torch.ops.aten.clone.default"
         && clone_no_format_change producer -> (
      match
        List.find_opt
          (fun (a : NamedArgument.t) -> a.name = "self")
          producer.Node.inputs
      with
      | Some { NamedArgument.arg = Argument.Tensor (t : TensorArgument.t); _ }
        when String_map.mem t.name constant_names ->
          t.name
      | _ -> live_name)
  | _ -> live_name

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

(* A REQUIRED [int]: no default, so omission is [`Missing_arg] -- the same
   fix [float_arg]'s own comment gives for smuggling a default into a field
   the schema declares with none. *)
let required_int_arg esc (node : Pytorch_types.Node.t) name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | None -> malformed esc (`Missing_arg { op = node.target; arg = name })
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

(* A REQUIRED [bool]: no default, so omission is [`Missing_arg] -- the same
   fix [float_arg]'s own comment gives for smuggling a default into a field
   the schema declares with none. *)
let required_bool_arg esc (node : Pytorch_types.Node.t) name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | None -> malformed esc (`Missing_arg { op = node.target; arg = name })
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
  | "torch.ops.aten.conv1d.default" | "torch.ops.aten.conv2d.default"
  | "torch.ops.aten.conv2d.padding" | "torch.ops.aten.conv3d.default"
  | "torch.ops.aten.convolution.default" | "torch.ops.aten.linear.default"
  | "torch.ops.aten._native_batch_norm_legit_no_training.default"
  | "torch.ops.aten.max_pool2d.default" | "torch.ops.aten.avg_pool2d.default"
  | "torch.ops.aten.adaptive_avg_pool2d.default"
  | "torch.ops.aten.adaptive_max_pool2d.default" | "torch.ops.aten.max.dim"
  | "torch.ops.aten.max_pool2d_with_indices.default"
  | "torch.ops.aten.rms_norm.default" | "torch.ops.aten.layer_norm.default"
  | "torch.ops.aten.native_layer_norm.default" | "torch.ops.aten.addmm.default"
  | "torch.ops.aten.lstm.input"
  | "torch.ops.aten.scaled_dot_product_attention.default"
  | "torch.ops.aten.upsample_bicubic2d.vec"
  | "torch.ops.aten.upsample_bilinear2d.vec"
  | "torch.ops.aten.upsample_nearest2d.vec" ->
      true
  | _ -> false

let materialized_output_names esc (node : Pytorch_types.Node.t) =
  match node.target with
  | "torch.ops.aten._native_batch_norm_legit_no_training.default"
  | "torch.ops.aten.adaptive_max_pool2d.default"
  | "torch.ops.aten.max_pool2d_with_indices.default"
  (* Third entry, and the first two whose dropped outputs are NOT empty: they
     are real f32 tensors that happen to be dead in every occurrence the
     corpus contains. Dropping them here is what makes the
     [`Live_layer_norm_stats] check below load-bearing rather than
     decorative for [native_layer_norm.default]; [adaptive_max_pool2d.default]
     and [max_pool2d_with_indices.default] have no analogous liveness check of
     their own (unlike layer_norm's stats, their indices output is routed to
     a [Discard] sink by the arm above rather than silently assumed dead).

     [torch.ops.aten.max.dim] does NOT join this bucket: its index is a
     plausible live output elsewhere (`values, indices = x.max(dim)`), so
     both its serialized names stay tracked here and its own lowering arm
     (native_interp_lower_reduce.ml) decides retain-vs-discard per output
     from [ctx.reads], the way [torch.ops.aten.lstm.input]'s three outputs
     already do. *)
  | "torch.ops.aten.native_layer_norm.default" ->
      [ List.hd (output_names esc node) ]
  | _ -> output_names esc node

let hw2 esc param = function
  | [ h; w ] -> (h, w)
  | [ x ] -> (x, x)
  | xs -> malformed esc (`Bad_arity { param; got = List.length xs })

(* [aten.conv1d.default]'s single-axis twin of [hw2]: no broadcast, since a
   1-D op has no second axis to fill in from a scalar list. *)
let w1 esc param = function
  | [ x ] -> x
  | xs -> malformed esc (`Bad_w_arity { param; got = List.length xs })

(* [aten.conv3d.default]'s three-axis twin of [hw2]: 3 values (ATen's own
   D/H/W order), or a single value broadcast to all three. *)
let dhw3 esc param = function
  | [ d; h; w ] -> (d, h, w)
  | [ x ] -> (x, x, x)
  | xs -> malformed esc (`Bad_dhw_arity { param; got = List.length xs })

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
