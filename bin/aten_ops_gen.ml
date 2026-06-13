(* Driver for the ATen operation-binding generator. Emits ONE of the three
   build artifacts (atg_ops.h, atg_ops.cpp, operation_description.ml) to stdout from a
   curated selection of ops, so lib/aten/dune can produce each with a
   [with-stdout-to] rule.

   Usage: aten_ops_gen <native_functions.yaml> <atg_ops.h|atg_ops.cpp|operation_description.ml>

   The op set is curated on purpose: the static-dispatch + --gc-sections build
   trims unreached at:: kernels, so binding every supported op would force-link
   ~1000 kernels and defeat that trimming. Two sources feed the selection:

     - [Allow]: ops pulled verbatim from native_functions.yaml by name. The
       generator (Aten_gen.Gen) must be able to emit them as-is.
     - [Override]: a hand-written schema signature used INSTEAD of the yaml
       entry, for ops the generator cannot yet emit unmodified. e.g. avg_pool2d
       carries an [int? divisor_override] (unsupported) plus several defaulted
       args; the override drops them to the (self, kernel_size) frontend
       overload that at::avg_pool2d still accepts via C++ defaults. *)

type selection =
  | Allow of { base : string; overload : string option }
  | Override of string

(* [op "name"] / [op "name" ~overload:"X"] selects an op from the yaml by base
   name (+ optional overload); [custom sig] overrides with a hand-written
   signature for ops the generator cannot emit unmodified. *)
let op ?overload base = Allow { base; overload }
let custom signature = Override signature

let selection =
  [
    op "add" ~overload:"Tensor";
    op "add_" ~overload:"Tensor";
    op "mul" ~overload:"Tensor";
    op "relu";
    op "relu_";
    op "sigmoid";
    op "hardtanh_";
    op "silu_";
    op "reshape";
    op "flatten" ~overload:"using_ints";
    op "max_pool2d";
    op "adaptive_avg_pool2d";
    op "linear";
    op "batch_norm";
    op "conv2d";
    op "dropout";
    op "dropout_";
    custom "avg_pool2d(Tensor self, int[2] kernel_size) -> Tensor";
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
  match Func_schema.parse sg with
  | Error e -> die "%s: parse error: %s" origin e
  | Ok op -> (
      match Aten_gen.Gen.generate ~style op with
      | Skipped r -> die "%s: generator skipped (%s): %s" origin r sg
      | Generated g -> g)

(* Find the native_functions.yaml entry matching [base]/[overload]. The call
   style follows the schema's [variants]: ops marked method-only have no at::
   free function, so they must be emitted as a Tensor method call. *)
let find_entry entries ~base ~overload =
  let matches (e : Raw.t) =
    match Func_schema.parse e.func with
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
let style_of (e : Raw.t) =
  match e.variants with
  | [] -> `Function
  | vs -> if List.mem Raw.Variant.Function vs then `Function else `Method

let resolve entries =
  List.map
    (fun sel ->
      match sel with
      | Allow { base; overload } ->
          let e = find_entry entries ~base ~overload in
          generate_sig ~origin:"schema" ~style:(style_of e) e.func
      | Override sg -> generate_sig ~origin:"override" ~style:`Function sg)
    selection

let () =
  if Array.length Sys.argv <> 3 then
    die
      "Usage: aten_ops_gen <native_functions.yaml> \
       <atg_ops.h|atg_ops.cpp|operation_description.ml>";
  let yaml = In_channel.with_open_bin Sys.argv.(1) In_channel.input_all in
  let entries =
    match Raw.of_yaml_string yaml with
    | Error (`Msg e) -> die "YAML parse error: %s" e
    | Ok entries -> entries
  in
  let ops = resolve entries in
  let out =
    match Sys.argv.(2) with
    | "atg_ops.h" -> Aten_gen.Emit.header ops
    | "atg_ops.cpp" -> Aten_gen.Emit.source ops
    | "operation_description.ml" -> Aten_gen.Emit.ocaml ops
    | other -> die "unknown output target: %s" other
  in
  print_string out
