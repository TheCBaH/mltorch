(* For every invocation of mobilenetv2_050, the text the inspector shows equals
   the kernel the whole-model builder used. Each builder is run with a [kernel]
   hook that calls the producer the builder would have called by default (the
   Loop emitter, or structured SSA) and records what it returned; the hook's
   result is compared with [Lowering_view.of_invocation] for the same
   invocation. *)

open Loop_ir
module A = Lowering_view.Artifact
module K = Lowering_key
module Lv = Lowering_view

let bundle () =
  let bytes = In_channel.with_open_bin "model.json" In_channel.input_all in
  let program =
    match
      Jsont_bytesrw.decode_string Pytorch_types.ExportedProgram.jsont bytes
    with
    | Ok p -> p
    | Error e -> failwith e
  in
  let lowered =
    Err.or_raise ~pp_error:Native_interp.pp_error (Native_interp.lower program)
  in
  Err.or_raise ~pp_error:Loop_bundle.pp_error
    (Loop_bundle.build ~config:Loop_bundle_c.default_config
       lowered.Pt2_native_graph.graph)

let nobody _ = false

(* The inspector's text for the invocation, or why it has none. *)
let shown inv variant =
  match (Lv.of_invocation ~granted:nobody inv variant).A.body with
  | A.Text { text; _ } -> Ok text
  | A.Refused r -> Error (Fmt.str "%a" A.Refusal.pp r)

(* A hook records (invocation, what the builder used) in call order. *)
let recorder () =
  let seen = ref [] in
  ((fun inv used -> seen := (inv, used) :: !seen), fun () -> List.rev !seen)

type tally = { mutable equal : int; mutable other : string list }

(* Compare every recorded kernel's printed form with the inspector's. *)
let check label variant ~print records =
  let t = { equal = 0; other = [] } in
  List.iteri
    (fun i (inv, used) ->
      match (shown inv variant, print inv used) with
      | Ok a, Some b when String.equal a b -> t.equal <- t.equal + 1
      | Ok _, Some _ -> t.other <- Fmt.str "#%d differs" i :: t.other
      | Error r, _ -> t.other <- Fmt.str "#%d refused: %s" i r :: t.other
      | Ok _, None -> t.other <- Fmt.str "#%d not comparable" i :: t.other)
    records;
  Fmt.pr "%s: %d invocations, %d equal%s@." label (List.length records) t.equal
    (match List.rev t.other with
    | [] -> ""
    | l ->
        ", NOT equal: " ^ String.concat "; " (List.filteri (fun i _ -> i < 5) l))

(* The builder names C kernels with a placeholder; the view uses [kernel]. *)
let c_text (k : Loop_c_kernel.t) =
  let rename s =
    let p = "static int KERNEL(" in
    if String.starts_with ~prefix:p s then
      "static int " ^ Lv.kernel_name ^ "("
      ^ String.sub s (String.length p) (String.length s - String.length p)
    else s
  in
  let src = rename k.Loop_c_kernel.source in
  if k.Loop_c_kernel.prelude = "" then src
  else k.Loop_c_kernel.prelude ^ "\n" ^ src

let c_case b label variant ~numerics ~vector ~produce =
  let record, get = recorder () in
  ignore
    (Err.or_raise ~pp_error:Loop_bundle_c.pp_error
       (Loop_bundle_c.build ~numerics ?vector
          ~kernel:(fun ~name inv ->
            match produce ~name inv with
            | Ok k as r ->
                record inv k;
                r
            | e -> e)
          b));
  check label variant ~print:(fun _ k -> Some (c_text k)) (get ())

let js_case b label variant ~produce =
  let record, get = recorder () in
  ignore
    (Err.or_raise ~pp_error:Loop_bundle_js.pp_error
       (Loop_bundle_js.build
          ~kernel:(fun inv ->
            match produce inv with
            | Ok a as r ->
                record inv a;
                r
            | e -> e)
          b));
  check label variant ~print:(fun _ a -> Some (Js_print.script a)) (get ())

(* A Wasm kernel is linked into a module the way a view's is. It is comparable
   exactly when it holds no constant table: the whole-model allocator hands
   each kernel its own addresses, the view's module starts its own. *)
let wat_case b label variant ~numerics ~vector ~produce =
  let record, get = recorder () in
  ignore
    (Err.or_raise ~pp_error:Loop_bundle_wasm.pp_error
       (Loop_bundle_wasm.build ~numerics ?vector
          ~kernel:(fun ~table_alloc inv ->
            match produce ~table_alloc inv with
            | Ok k as r ->
                record inv k;
                r
            | e -> e)
          b));
  let records = get () in
  let tabled =
    List.length (List.filter (fun (_, k) -> k.Loop_wasm.data <> []) records)
  in
  check (Fmt.str "%s (%d with constant tables, rebased: skipped)" label tabled)
    variant records ~print:(fun _ (k : Loop_wasm.kernel) ->
      if k.Loop_wasm.data <> [] then None
      else
        match
          Err.payload
            (Loop_wasm_module.lower_with ~numerics
               (fun ~mark_base:_ ~table_alloc:_ -> Err.return k))
        with
        | Ok m -> Some (Wasm_wat.to_string m.Loop_wasm_module.module_)
        | Error (_ : unit) -> None)

let loop_with numerics =
  K.Route.Loop { K.Loop_spec.default with K.Loop_spec.numerics = Some numerics }

let ssa_route = K.Route.Ssa K.Ssa_spec.default

let ssa_planned numerics target =
  Ssa_backends.Pipeline.Planned { numerics; target }

let run () =
  let b = bundle () in
  let ref64 = Loop_numerics.Reference_f64 in
  let simd = Loop_numerics.Simd_fp32_relaxed in
  c_case b "C loop reference"
    (K.C (loop_with ref64))
    ~numerics:ref64 ~vector:None
    ~produce:(fun ~name inv ->
      Result.map_error
        (fun _ -> "refused")
        (Err.payload
           (Loop_c.kernel ~numerics:ref64 ~name inv.Loop_bundle.program)));
  c_case b "C loop simd_fp32_relaxed" (K.default K.C) ~numerics:simd
    ~vector:(Some Loop_target.neon128) ~produce:(fun ~name inv ->
      Result.map_error
        (fun _ -> "refused")
        (Err.payload
           (Loop_c.kernel ~vector:Loop_target.neon128 ~numerics:simd ~name
              inv.Loop_bundle.program)));
  c_case b "C ssa planned" (K.C ssa_route) ~numerics:simd ~vector:None
    ~produce:(fun ~name inv ->
      Ssa_backends.c
        ~pipeline:
          (ssa_planned Ssa_ir.Ssa_numerics.Simd_fp32_relaxed
             Ssa_ir.Ssa_target.neon128)
        ~name inv);
  js_case b "JS loop" (K.default K.Js) ~produce:(fun inv ->
      Ok (Loop_js.to_ast inv.Loop_bundle.program));
  js_case b "JS ssa planned" (K.Js ssa_route) ~produce:(fun inv ->
      Ssa_backends.js
        ~pipeline:
          (ssa_planned Ssa_ir.Ssa_numerics.Reference_f64
             Ssa_ir.Ssa_target.scalar)
        inv);
  wat_case b "WAT loop reference"
    (K.Wat (loop_with ref64))
    ~numerics:ref64 ~vector:None
    ~produce:(fun ~table_alloc inv ->
      Result.map_error
        (fun _ -> "refused")
        (Err.payload
           (Loop_wasm.kernel ~numerics:ref64 ~table_alloc
              inv.Loop_bundle.program)));
  wat_case b "WAT loop simd_fp32_relaxed" (K.default K.Wat) ~numerics:simd
    ~vector:(Some Loop_target.wasm128) ~produce:(fun ~table_alloc inv ->
      Result.map_error
        (fun _ -> "refused")
        (Err.payload
           (Loop_wasm.kernel ~vector:Loop_target.wasm128 ~numerics:simd
              ~table_alloc inv.Loop_bundle.program)));
  wat_case b "WAT ssa planned" (K.Wat ssa_route) ~numerics:simd ~vector:None
    ~produce:(fun ~table_alloc inv ->
      Ssa_backends.wasm
        ~pipeline:
          (ssa_planned Ssa_ir.Ssa_numerics.Simd_fp32_relaxed
             Ssa_ir.Ssa_target.wasm128)
        ~table_alloc inv)
