open Graph_ir
open Loop_ir
module P = C_payload_layout
module W = C_workspace_plan
module Proc = Loop_c_exec.Proc
module Io = Loop_c_exec.Payload_io

type toolchain = { clang : string list; builtins : string }

type error =
  [ `Generate_c of Loop_bundle_c.error
  | `Toolchain_missing of string
  | `Compile_failed of Proc.status * string
  | `Missing_constant of Tensor_id.t
  | `Missing_input of Tensor_id.t
  | `Binding_mismatch of Kernel_eval.Binding_mismatch.t
  | `Io of string
  | `Node_unavailable of string
  | `Run_failed of Proc.status * string
  | `Inference_failed of int * Loop_interp.error
  | `Bad_failure_record of string
  | `Bad_output of string
  | `Memory_over_policy of int64 ]

let pp_error ppf : [< error ] -> unit = function
  | `Generate_c e -> Loop_bundle_c.pp_error ppf e
  | `Toolchain_missing m -> Fmt.pf ppf "wasm toolchain missing: %s" m
  | `Compile_failed (st, log) ->
      Fmt.pf ppf "compiling C to Wasm failed (%a):@.%s" Proc.pp_status st log
  | `Missing_constant id ->
      Fmt.pf ppf "no value for constant %a" Tensor_id.pp id
  | `Missing_input id -> Fmt.pf ppf "no value for input %a" Tensor_id.pp id
  | `Binding_mismatch m -> Kernel_eval.Binding_mismatch.pp ppf m
  | `Io m -> Fmt.pf ppf "I/O: %s" m
  | `Node_unavailable m -> Fmt.pf ppf "node unavailable: %s" m
  | `Run_failed (st, log) ->
      Fmt.pf ppf "the Wasm host failed (%a):@.%s" Proc.pp_status st log
  | `Inference_failed (i, e) ->
      Fmt.pf ppf "inference failed in invocation %d: %a" i Loop_interp.pp_error
        e
  | `Bad_failure_record m -> Fmt.pf ppf "unreadable failure record: %s" m
  | `Bad_output m -> Fmt.pf ppf "output file rejected: %s" m
  | `Memory_over_policy n ->
      Fmt.pf ppf "%Ld bytes exceed the 2 GiB memory policy" n

type sizes = {
  source_bytes : int;
  module_bytes : int;
  memory_bytes : int;
  heap_base : int;
  vector_instructions : int;
}

type prepared = {
  dir : string;
  module_file : string;
  weights : string;
  template : string;
  bundle : Loop_bundle.t;
  c : Loop_bundle_c.t;
  placement : Wasm_node.placement;
  sizes : sizes;
  mutable runs : int;
  mutable timings : (string * float) list;
}

let ( let* ) = Result.bind

let scalar_flags =
  [
    "-O2";
    "-ffp-contract=off";
    "-fno-strict-aliasing";
    "-fno-vectorize";
    "-fno-slp-vectorize";
    "-mno-simd128";
  ]

let memory_cap = 0x8000_0000L

let toolchain_from_env () =
  match Sys.getenv_opt "MLTORCH_WASI_SYSROOT" with
  | None | Some "" ->
      Error
        (`Toolchain_missing
           "MLTORCH_WASI_SYSROOT is unset (run \
            scripts/wasi-sysroot-userland.py and pass its SYSROOT)")
  | Some sysroot ->
      let builtins =
        Filename.concat sysroot
          "lib/llvm-19/lib/clang/19/lib/wasi/libclang_rt.builtins-wasm32.a"
      in
      let clang =
        Option.value (Sys.getenv_opt "MLTORCH_WASM_CLANG") ~default:"clang"
      in
      if
        not
          (Sys.file_exists
             (Filename.concat sysroot "include/wasm32-wasi/math.h"))
      then
        Error
          (`Toolchain_missing
             (Printf.sprintf "no wasm32 libc headers under %s" sysroot))
      else if not (Sys.file_exists builtins) then
        Error (`Toolchain_missing ("no compiler runtime at " ^ builtins))
      else
        Ok
          {
            clang = [ clang; "--target=wasm32-wasi"; "--sysroot=" ^ sysroot ];
            builtins;
          }

(* The C entry points a host can call: [run] over the inference unit's
   [model_run], the failure record's address, and where the module's static data
   ends. *)
let wrapper =
  Printf.sprintf
    {|#include <stdint.h>
struct model_error { int32_t kind; int32_t invocation; int64_t v[%d]; };
extern int model_run(const void *weights, const void *inputs, void *workspace,
                     void *outputs, struct model_error *error);
static struct model_error g_error;
extern unsigned char __heap_base;
__attribute__((export_name("run"))) int run(const void *w, const void *in, void *ws, void *out) {
  return model_run(w, in, ws, out, &g_error);
}
__attribute__((export_name("error_ptr"))) const void *error_ptr(void) { return &g_error; }
__attribute__((export_name("heap_base"))) unsigned heap_base(void) { return (unsigned)(uintptr_t)&__heap_base; }
|}
    Loop_c_runtime.error_words

let run_cmd argv =
  match Proc.run argv with
  | Error m -> Error (`Toolchain_missing m)
  | Ok (Proc.Exited 0, log) -> Ok log
  | Ok (st, log) -> Error (`Compile_failed (st, log))

let count_vector_instructions asm =
  let n = ref 0 in
  List.iter
    (fun line ->
      let t = String.trim line in
      if
        String.length t > 5
        && (String.starts_with ~prefix:"v128." t
           || String.contains t '.'
              && List.exists
                   (fun p -> String.starts_with ~prefix:p t)
                   [
                     "i8x16."; "i16x8."; "i32x4."; "i64x2."; "f32x4."; "f64x2.";
                   ])
      then incr n)
    (String.split_on_char '\n' asm);
  !n

let align n a = Int64.mul (Int64.div (Int64.add n (Int64.sub a 1L)) a) a

let prepare_r ?(flags = scalar_flags) ~toolchain ~dir (b : Loop_bundle.t)
    ~constants =
  let* c =
    Result.map_error
      (fun e -> `Generate_c e)
      (Err.payload (Loop_bundle_c.build b))
  in
  let* () = Io.unix_io (fun () -> Io.mkdir_p dir) in
  let file name = Filename.concat dir name in
  let* tensors =
    Io.bound_tensors c.Loop_bundle_c.weights ~lookup:constants
      ~missing:(fun id -> `Missing_constant id)
  in
  let* () =
    Io.io (fun () ->
        Proc.write_file (file "model_infer.c") c.Loop_bundle_c.source;
        Proc.write_file (file "model_wasm.c") wrapper)
  in
  let* () =
    Io.write_payload c.Loop_bundle_c.weights ~identity:c.Loop_bundle_c.identity
      ~path:(file "weights.bin") tensors
  in
  let* () =
    Io.unix_io (fun () ->
        let fd =
          Unix.openfile (file "outputs.template")
            [ Unix.O_RDWR; Unix.O_CREAT; Unix.O_TRUNC ]
            0o644
        in
        Fun.protect
          ~finally:(fun () -> Unix.close fd)
          (fun () ->
            let outputs = c.Loop_bundle_c.outputs in
            Unix.ftruncate fd (Int64.to_int outputs.P.length);
            let h = P.header outputs ~identity:c.Loop_bundle_c.identity in
            ignore (Unix.write_substring fd h 0 (String.length h))))
  in
  let cc = toolchain.clang @ flags in
  let obj name = file (name ^ ".o") in
  let* _ =
    run_cmd (cc @ [ "-c"; file "model_infer.c"; "-o"; obj "model_infer" ])
  in
  let* _ =
    run_cmd (cc @ [ "-c"; file "model_wasm.c"; "-o"; obj "model_wasm" ])
  in
  let* asm =
    let* _ =
      run_cmd (cc @ [ "-S"; file "model_infer.c"; "-o"; file "model_infer.s" ])
    in
    Result.map_error
      (fun m -> `Io m)
      (try Ok (Proc.read_file (file "model_infer.s"))
       with Sys_error m -> Error m)
  in
  let link ~pages =
    let out = file "model.wasm" in
    let* _ =
      run_cmd
        (toolchain.clang
        @ [
            "-nostartfiles";
            "-Wl,--no-entry";
            "-Wl,-z,stack-size=1048576";
            Printf.sprintf "-Wl,--initial-memory=%d" (pages * 65536);
            "-o";
            out;
            obj "model_infer";
            obj "model_wasm";
            "-nodefaultlibs";
            "-lc";
            "-lm";
            toolchain.builtins;
          ])
    in
    Ok out
  in
  let* module_file = link ~pages:64 in
  (* The static data ends at [heap_base], known once linked. *)
  let* heap_base =
    let script = file "heap_base.js" in
    Proc.write_file script
      "const fs=require('fs');const i=new WebAssembly.Instance(new \
       WebAssembly.Module(fs.readFileSync(process.argv[2])),{math:{},wasi_snapshot_preview1:new \
       Proxy({}, {get:()=>()=>0})});console.log(i.exports.heap_base());";
    match Proc.run (!Wasm_node.node @ [ script; module_file ]) with
    | Ok (Proc.Exited 0, log) -> (
        match int_of_string_opt (String.trim log) with
        | Some n -> Ok n
        | None -> Error (`Io ("heap_base: " ^ log)))
    | Ok (st, log) -> Error (`Run_failed (st, log))
    | Error m -> Error (`Node_unavailable m)
  in
  let ws = c.Loop_bundle_c.workspace in
  let static_end = align (Int64.of_int heap_base) 64L in
  let w_at = static_end in
  let i_at = align (Int64.add w_at c.Loop_bundle_c.weights.P.length) 64L in
  let ws_at =
    align
      (Int64.add i_at c.Loop_bundle_c.inputs.P.length)
      (Int64.max 64L (W.alignment ws))
  in
  let o_at = align (Int64.add ws_at (W.bytes ws)) 64L in
  let total = Int64.add o_at c.Loop_bundle_c.outputs.P.length in
  let* () =
    if Int64.compare total memory_cap > 0 then Error (`Memory_over_policy total)
    else Ok ()
  in
  let pages = Int64.to_int (Int64.div (Int64.add total 65535L) 65536L) in
  let* module_file = link ~pages in
  let placement =
    {
      Wasm_node.weights = Int64.to_int w_at;
      inputs = Int64.to_int i_at;
      workspace = Int64.to_int ws_at;
      outputs = Int64.to_int o_at;
      workspace_bytes = W.bytes ws;
      outputs_bytes = c.Loop_bundle_c.outputs.P.length;
    }
  in
  Ok
    {
      dir;
      module_file;
      weights = file "weights.bin";
      template = file "outputs.template";
      bundle = b;
      c;
      placement;
      sizes =
        {
          source_bytes = String.length c.Loop_bundle_c.source;
          module_bytes = String.length (Proc.read_file module_file);
          memory_bytes = Int64.to_int total;
          heap_base;
          vector_instructions = count_vector_instructions asm;
        };
      runs = 0;
      timings = [];
    }

let run_r ?(poison = false) ?(repeat = 0) p ~bind =
  p.runs <- p.runs + 1;
  let inputs = Filename.concat p.dir (Printf.sprintf "inputs.%d.bin" p.runs) in
  let outputs =
    Filename.concat p.dir (Printf.sprintf "outputs.%d.bin" p.runs)
  in
  let remove f = try Sys.remove f with Sys_error _ -> () in
  Fun.protect
    ~finally:(fun () ->
      remove inputs;
      remove outputs)
    (fun () ->
      remove outputs;
      let* tensors =
        Io.bound_tensors p.c.Loop_bundle_c.inputs ~lookup:bind
          ~missing:(fun id -> `Missing_input id)
      in
      let* () =
        Io.write_payload p.c.Loop_bundle_c.inputs
          ~identity:p.c.Loop_bundle_c.identity ~path:inputs tensors
      in
      match
        Wasm_node.execute ~dir:p.dir ~module_file:p.module_file
          ~weights:p.weights ~inputs ~template:p.template ~outputs p.placement
          ~poison ~repeat
      with
      | Error m -> Error (`Node_unavailable m)
      | Ok (Proc.Exited 0, log) ->
          p.timings <- Wasm_node.timing_of log;
          Io.read_payload p.c.Loop_bundle_c.outputs
            ~identity:p.c.Loop_bundle_c.identity ~path:outputs
      | Ok (Proc.Exited n, log) when n = Wasm_node.exit_inference ->
          Wasm_node.parse_failure p.bundle log
      | Ok (st, log) -> Error (`Run_failed (st, log)))

let lift r = Err.import Fun.id r

let prepare ?flags ~toolchain ~dir b ~constants =
  lift (prepare_r ?flags ~toolchain ~dir b ~constants)

let run ?poison ?repeat p ~bind = lift (run_r ?poison ?repeat p ~bind)
let timings p = p.timings
let sizes p = p.sizes
let bundle_c p = p.c
let directory p = p.dir
let module_path p = p.module_file
let weights_path p = p.weights
