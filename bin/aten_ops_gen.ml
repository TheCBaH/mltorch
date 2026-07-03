(* Driver for the ATen operation-binding generator. Emits ONE of the three
   build artifacts (atg_ops.h, atg_ops.cpp, aten_operation_description.ml) to stdout from a
   curated selection of ops, so lib/aten/dune can produce each with a
   [with-stdout-to] rule.

   Usage: aten_ops_gen <native_functions.yaml> <atg_ops.h|atg_ops.cpp|aten_operation_description.ml>

   The op set is curated on purpose: the static-dispatch + --gc-sections build
   trims unreached at:: kernels, so binding every supported op would force-link
   ~1000 kernels and defeat that trimming. Two sources feed the selection:

     - [Allow]: ops pulled verbatim from native_functions.yaml by name. The
       generator (Aten_gen) must be able to emit them as-is.
     - [Override]: a hand-written schema signature used INSTEAD of the yaml
       entry, for ops the generator cannot yet emit unmodified. e.g. the
       conversions (clone/contiguous/cpu/to.dtype) drop MemoryFormat args. *)

type selection =
  | Allow of { base : string; overload : string option }
  | Override of { signature : string; style : [ `Function | `Method ] }

(* [op "name"] / [op "name" ~overload:"X"] selects an op from the yaml by base
   name (+ optional overload); [custom sig] overrides with a hand-written
   signature for ops the generator cannot emit unmodified. [~style:`Method] emits
   a [self->op(...)] call, for method-only ops with no at:: free function (or no
   schema entry at all, e.g. cpu, which torchgen synthesizes from to(kCPU)). *)
let op ?overload base = Allow { base; overload }
let custom ?(style = `Function) signature = Override { signature; style }

let selection =
  [
    op "add" ~overload:"Tensor";
    op "add_" ~overload:"Tensor";
    op "mul" ~overload:"Tensor";
    op "mul" ~overload:"Scalar";
    op "div" ~overload:"Tensor";
    op "clamp";
    (* masking ops used by the ViT models (attention-mask construction) *)
    op "eq" ~overload:"Scalar";
    op "logical_not";
    op "any" ~overload:"dim";
    op "where" ~overload:"self";
    op "full_like";
    op "relu";
    op "relu_";
    op "sigmoid";
    op "hardtanh";
    op "hardtanh_";
    op "silu_";
    op "reshape";
    op "flatten" ~overload:"using_ints";
    op "permute";
    op "convolution";
    op "addmm";
    op "bmm";
    op "_softmax";
    op "gelu";
    op "cat";
    (* tuple returns: output 0 is the return, extra outputs via out-params *)
    op "_native_batch_norm_legit_no_training";
    op "native_layer_norm";
    op "rms_norm";
    op "max_pool2d_with_indices";
    op "max_pool2d";
    op "adaptive_avg_pool2d";
    op "linear";
    op "batch_norm";
    op "conv2d";
    op "conv2d" ~overload:"padding";
    op "dropout";
    op "dropout_";
    op "avg_pool2d";
    op "view";
    (* shape/view ops used by the ViT models *)
    op "expand";
    op "select" ~overload:"int";
    op "squeeze" ~overload:"dims";
    op "unsqueeze";
    op "mean" ~overload:"dim";
    op "argmax";
    op "topk";
    op "clone";
    op "contiguous";
    op "to" ~overload:"dtype";
    (* cpu: no native_functions.yaml entry (synthesized from to(kCPU)); the
       Tensor::cpu() method is nullary, so this signature is exact, not trimmed. *)
    custom ~style:`Method "cpu(Tensor self) -> Tensor";
  ]

let die fmt =
  Printf.ksprintf
    (fun s ->
      prerr_endline s;
      exit 1)
    fmt

(* Parse a schema signature and run it through the generator, failing loudly if
   it cannot be parsed or the generator skips it (the selection is curated, so a
   skip is a configuration error, not an expected outcome). *)
let generate_sig ~origin ~style sg =
  match Aten_func_schema.parse sg with
  | Error e -> die "%s: parse error: %s" origin e
  | Ok op -> (
      match Aten_gen.generate ~style op with
      | Skipped r -> die "%s: generator skipped (%s): %s" origin r sg
      | Generated g -> g)

(* Find the native_functions.yaml entry matching [base]/[overload]. The call
   style follows the schema's [variants]: ops marked method-only have no at::
   free function, so they must be emitted as a Tensor method call. *)
let find_entry entries ~base ~overload =
  let matches (e : Aten_raw.t) =
    match Aten_func_schema.parse e.func with
    | Error _ -> false
    | Ok op -> op.name.base = base && op.name.overload = overload
  in
  match List.find_opt matches entries with
  | Some e -> e
  | None ->
      let ov = match overload with None -> "" | Some s -> "." ^ s in
      die "no native_functions.yaml entry for op %s%s" base ov

(* torchgen defaults an absent [variants] to "function", so emit a free-function
   call unless the op is explicitly method-only (a non-empty list without
   Function, e.g. in-place add_). *)
let style_of (e : Aten_raw.t) =
  match e.variants with
  | [] -> `Function
  | vs -> if List.mem Aten_raw.Variant.Function vs then `Function else `Method

(* Parse every binding-selected op for the schema-driven interpreter dispatch
   generator (Aten_decode_gen) -- the same curated [selection] that drives the
   bindings, so the two never drift. Ops whose argument/return types the
   generator cannot decode simply produce no arm (Aten_decode_gen.dispatch_arm
   returns None) and fall through to the interpreter's failwith. *)
let dispatch_ops entries =
  List.filter_map
    (fun sel ->
      let sg =
        match sel with
        | Allow { base; overload } -> (find_entry entries ~base ~overload).func
        | Override { signature; _ } -> signature
      in
      match Aten_func_schema.parse sg with
      | Ok op -> Some op
      | Error e -> die "dispatch: parse error: %s" e)
    selection

let resolve entries =
  List.map
    (fun sel ->
      match sel with
      | Allow { base; overload } ->
          let e = find_entry entries ~base ~overload in
          generate_sig ~origin:"schema" ~style:(style_of e) e.func
      | Override { signature; style } ->
          generate_sig ~origin:"override" ~style signature)
    selection

let () =
  if Array.length Sys.argv <> 3 then
    die
      "Usage: aten_ops_gen <native_functions.yaml> \
       <atg_ops.h|atg_ops.cpp|aten_operation_description.ml>";
  let yaml = In_channel.with_open_bin Sys.argv.(1) In_channel.input_all in
  let entries =
    match Aten_raw.of_yaml_string yaml with
    | Error (`Msg e) -> die "YAML parse error: %s" e
    | Ok entries -> entries
  in
  let ops = resolve entries in
  let out =
    match Sys.argv.(2) with
    | "atg_ops.h" -> Aten_emit.header ops
    | "atg_ops.cpp" -> Aten_emit.source ops
    | "aten_operation_description.ml" -> Aten_emit.ocaml ops
    | "interp_dispatch.ml" -> Aten_decode_gen.file (dispatch_ops entries)
    | "aten_op_config.ml" -> Aten_config_gen.file (dispatch_ops entries)
    | "aten_op_spec.ml" -> Aten_spec_gen.file (dispatch_ops entries)
    | "aten_op_walk.ml" -> Aten_walk_gen.file (dispatch_ops entries)
    | other -> die "unknown output target: %s" other
  in
  print_string out
