open Js_of_ocaml
open Graph_ir
open Loop_ir
module P = C_payload_layout
module B = Loop_bundle_wasm

type error =
  [ `Generate of B.error
  | `Wasm_unavailable
  | `Wasm_compile of string
  | `Wasm_instantiate of string
  | `Missing_constant of Tensor_id.t
  | `Missing_input of Tensor_id.t
  | `Binding_mismatch of Kernel_eval.Binding_mismatch.t
  | `Disposed
  | `Inference_failed of int * Loop_interp.error
  | `Bad_failure_record of string
  | `Js_exception of string ]

let pp_error ppf : [< error ] -> unit = function
  | `Generate e -> B.pp_error ppf e
  | `Wasm_unavailable -> Fmt.string ppf "this host has no WebAssembly"
  | `Wasm_compile m -> Fmt.pf ppf "the engine rejected the module: %s" m
  | `Wasm_instantiate m -> Fmt.pf ppf "instantiation failed: %s" m
  | `Missing_constant id ->
      Fmt.pf ppf "no value for constant %a" Tensor_id.pp id
  | `Missing_input id -> Fmt.pf ppf "no value for input %a" Tensor_id.pp id
  | `Binding_mismatch m -> Kernel_eval.Binding_mismatch.pp ppf m
  | `Disposed -> Fmt.string ppf "the model was disposed"
  | `Inference_failed (i, e) ->
      Fmt.pf ppf "inference failed in invocation %d: %a" i Loop_interp.pp_error
        e
  | `Bad_failure_record m -> Fmt.pf ppf "unreadable failure record: %s" m
  | `Js_exception m -> Fmt.pf ppf "model_run threw: %s" m

type live = { exports : Js.Unsafe.any; memory : Js.Unsafe.any }

type t = {
  w : B.t;
  bundle : Loop_bundle.t;
  mutable live : live option;
  mutable bytes : int;
}

let global = Js.Unsafe.global
let get = Js.Unsafe.get
let inject = Js.Unsafe.inject
let num n = inject (float_of_int n)
let ( let* ) = Result.bind

let webassembly () =
  let w = get global (Js.string "WebAssembly") in
  if Js.Optdef.test w then Some w else None

let uint8 ?(offset = 0) ?length buffer =
  let ctor = get global (Js.string "Uint8Array") in
  match length with
  | None -> Js.Unsafe.new_obj ctor [| buffer; num offset |]
  | Some n -> Js.Unsafe.new_obj ctor [| buffer; num offset; num n |]

let bytes_of_string s =
  let a =
    Js.Unsafe.new_obj
      (get global (Js.string "Uint8Array"))
      [| num (String.length s) |]
  in
  String.iteri (fun i c -> Js.Unsafe.set a i (Char.code c)) s;
  a

let imports () =
  let math = get global (Js.string "Math") in
  let f name = (Js.string name, get math (Js.string name)) in
  Js.Unsafe.obj
    [|
      ( "math",
        inject
          (Js.Unsafe.obj
             (Array.map
                (fun (k, v) -> (Js.to_string k, v))
                [| f "exp"; f "log"; f "sin"; f "cos" |])) );
    |]

let message e =
  match Js.Js_error.of_exn e with
  | Some err -> Js.Js_error.to_string err
  | None -> Printexc.to_string e

(* A tensor's bytes, as a [Uint8Array] over its own storage: an int64 tensor's is
   the [Int32Array] of [lo, hi] pairs, whose bytes are little-endian words. *)
let tensor_bytes tensor =
  let s = Loop_js_exec.storage tensor in
  uint8
    ~offset:(Js.Unsafe.get s (Js.string "byteOffset"))
    ~length:(Js.Unsafe.get s (Js.string "byteLength"))
    (Js.Unsafe.get s (Js.string "buffer"))

let little_endian = Loop_js_exec.little_endian

let checked (layout : P.t) ~lookup ~missing =
  List.fold_left
    (fun acc (e : P.Entry.t) ->
      let* acc = acc in
      match lookup e.P.Entry.id with
      | None -> Error (missing e.P.Entry.id)
      | Some t -> (
          match
            Err.payload (Kernel_eval.check_binding e.P.Entry.id e.P.Entry.sg t)
          with
          | Error (`Binding_mismatch m) -> Error (`Binding_mismatch m)
          | Ok () -> Ok (t :: acc)))
    (Ok []) layout.P.entries
  |> Result.map List.rev

let memory_view live = uint8 (Js.Unsafe.get live.memory (Js.string "buffer"))

let place live ~at (layout : P.t) tensors =
  let mem = memory_view live in
  List.iter2
    (fun (e : P.Entry.t) tensor ->
      ignore
        (Js.Unsafe.meth_call mem "set"
           [|
             inject (tensor_bytes tensor);
             num (at + Int64.to_int e.P.Entry.offset);
           |]))
    layout.P.entries tensors

let finish w bundle ~constants instance =
  let exports = get instance (Js.string "exports") in
  let live = { exports; memory = get exports (Js.string "memory") } in
  let pl = w.B.placement in
  let* tensors =
    checked w.B.weights ~lookup:constants ~missing:(fun id ->
        `Missing_constant id)
  in
  place live ~at:pl.B.Placement.weights w.B.weights tensors;
  Ok
    {
      w;
      bundle;
      live = Some live;
      bytes =
        Js.Unsafe.get
          (Js.Unsafe.get live.memory (Js.string "buffer"))
          (Js.string "byteLength");
    }

let generate b =
  if not little_endian then
    Error (`Js_exception "a big-endian host is not supported")
  else Result.map_error (fun e -> `Generate e) (Err.payload (B.build b))

let prepare_r b ~constants =
  let* w = generate b in
  match webassembly () with
  | None -> Error `Wasm_unavailable
  | Some wa ->
      let bytes = bytes_of_string w.B.bytes in
      let* modul =
        match
          Js.Unsafe.new_obj (get wa (Js.string "Module")) [| inject bytes |]
        with
        | m -> Ok m
        | exception e -> Error (`Wasm_compile (message e))
      in
      let* instance =
        match
          Js.Unsafe.new_obj
            (get wa (Js.string "Instance"))
            [| inject modul; inject (imports ()) |]
        with
        | i -> Ok i
        | exception e -> Error (`Wasm_instantiate (message e))
      in
      finish w b ~constants instance

let lift r = Err.import Fun.id r
let prepare b ~constants = lift (prepare_r b ~constants)

let prepare_async b ~constants k =
  match generate b with
  | Error e -> k (lift (Error e))
  | Ok w -> (
      match webassembly () with
      | None -> k (lift (Error `Wasm_unavailable))
      | Some wa ->
          let bytes = bytes_of_string w.B.bytes in
          let promise =
            Js.Unsafe.meth_call wa "instantiate"
              [| inject bytes; inject (imports ()) |]
          in
          let ok =
            Js.wrap_callback (fun result ->
                let instance = get result (Js.string "instance") in
                k (lift (finish w b ~constants instance)))
          in
          let err =
            Js.wrap_callback (fun e ->
                let m =
                  Js.to_string
                    (Js.Unsafe.meth_call e "toString" [||] : Js.js_string Js.t)
                in
                k (lift (Error (`Wasm_compile m))))
          in
          ignore
            (Js.Unsafe.meth_call promise "then" [| inject ok; inject err |]))

let words live =
  let dv =
    Js.Unsafe.new_obj
      (get global (Js.string "DataView"))
      [| Js.Unsafe.get live.memory (Js.string "buffer") |]
  in
  let i32 off : int =
    Js.Unsafe.meth_call dv "getInt32" [| num off; inject Js._true |]
  in
  let i64 off =
    let lo = Int64.logand (Int64.of_int (i32 off)) 0xFFFF_FFFFL in
    let hi = Int64.of_int (i32 (off + 4)) in
    Int64.logor lo (Int64.shift_left hi 32)
  in
  ( i32 4,
    i32 0,
    Array.init Loop_wasm_failure.error_words (fun k -> i64 (8 + (8 * k))) )

let decode_failure t live =
  let invocation, kind, v = words live in
  let invocations = t.bundle.Loop_bundle.invocations in
  if invocation < 0 || invocation >= List.length invocations then
    Error (`Bad_failure_record "invocation out of range")
  else
    let inv = List.nth invocations invocation in
    let sites = Loop_js_failure.sites inv.Loop_bundle.program in
    match Loop_c_failure.decode ~sites ~kind ~v with
    | Ok row -> Error (`Inference_failed (invocation, row))
    | Error m -> Error (`Bad_failure_record m)

let run_r t ~bind =
  match t.live with
  | None -> Error `Disposed
  | Some live ->
      let w = t.w in
      let pl = w.B.placement in
      let* tensors =
        checked w.B.inputs ~lookup:bind ~missing:(fun id -> `Missing_input id)
      in
      place live ~at:pl.B.Placement.inputs w.B.inputs tensors;
      let rc =
        match
          Js.Unsafe.meth_call live.exports "model_run"
            [|
              num pl.B.Placement.weights;
              num pl.B.Placement.inputs;
              num pl.B.Placement.workspace;
              num pl.B.Placement.outputs;
            |]
        with
        | (r : int) -> Ok r
        | exception e -> Error (`Js_exception (message e))
      in
      let* rc = rc in
      if rc <> 0 then decode_failure t live
      else
        let mem = memory_view live in
        let* outs =
          List.fold_left
            (fun acc (e : P.Entry.t) ->
              let* acc = acc in
              match Err.payload (Tensor.create_of_sig e.P.Entry.sg) with
              | Error (`Quant_missing _) ->
                  Error (`Js_exception "a quantized graph output")
              | Ok tensor ->
                  let n = Int64.to_int e.P.Entry.bytes in
                  (if n > 0 then
                     let src =
                       Js.Unsafe.meth_call mem "subarray"
                         [|
                           num
                             (pl.B.Placement.outputs
                             + Int64.to_int e.P.Entry.offset);
                           num
                             (pl.B.Placement.outputs
                             + Int64.to_int e.P.Entry.offset
                             + n);
                         |]
                     in
                     ignore
                       (Js.Unsafe.meth_call (tensor_bytes tensor) "set"
                          [| src |]));
                  Ok (tensor :: acc))
            (Ok []) w.B.outputs.P.entries
        in
        Ok (List.rev outs)

let run t ~bind = lift (run_r t ~bind)

let dispose t =
  t.live <- None;
  t.bytes <- 0

let bundle_wasm t = t.w
let memory_bytes t = t.bytes
