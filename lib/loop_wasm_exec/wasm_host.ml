open Graph_ir
open Loop_ir
module P = C_payload_layout
module Proc = Loop_c_exec.Proc
module Io = Loop_c_exec.Payload_io

type error =
  [ `Generate of Loop_bundle_wasm.error
  | `Missing_constant of Tensor_id.t
  | `Missing_input of Tensor_id.t
  | `Binding_mismatch of Kernel_eval.Binding_mismatch.t
  | `Io of string
  | `Node_unavailable of string
  | `Run_failed of Proc.status * string
  | `Inference_failed of int * Loop_interp.error
  | `Bad_failure_record of string
  | `Bad_output of string ]

let pp_error ppf : [< error ] -> unit = function
  | `Generate e -> Loop_bundle_wasm.pp_error ppf e
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

type prepared = {
  dir : string;
  module_file : string;
  weights : string;
  template : string;  (** the outputs file's header and zero padding *)
  runner : string;
  bundle : Loop_bundle.t;
  w : Loop_bundle_wasm.t;
  mutable runs : int;
  mutable timings : (string * float) list;
}

let node = ref [ "node" ]
let ( let* ) = Result.bind
let exit_inference = 5
let failure_prefix = "model_error:"
let timing_prefix = "wasm_timing:"

(* The runner places the four regions at the module's default placement, runs
   [model_run], and writes the outputs region. A failure is the record, as a
   line the host parses, and exit status 5; anything else (a compile error, a
   trap) is node's own nonzero exit. The math imports are the host's [Math]. *)
let runner_text =
  {|const fs = require("fs");
const now = () => Number(process.hrtime.bigint()) / 1e6;
const [wasmPath, weightsPath, inputsPath, templatePath, outputsPath, ...rest] = process.argv.slice(2);
const [wAt, iAt, wsAt, oAt, wsBytes, oBytes] = rest.slice(0, 6).map(Number);
const flags = rest.slice(6);
const poison = flags.includes("--poison");
const repeat = Number((flags.find((f) => f.startsWith("--repeat=")) || "--repeat=0").slice(9));
const bytes = fs.readFileSync(wasmPath);
let t = now();
const mod = new WebAssembly.Module(bytes);
const compile = now() - t;
t = now();
const inst = new WebAssembly.Instance(mod, { math: { exp: Math.exp, log: Math.log, sin: Math.sin, cos: Math.cos } });
const instantiate = now() - t;
const mem = new Uint8Array(inst.exports.memory.buffer);
t = now();
const weights = fs.readFileSync(weightsPath), inputs = fs.readFileSync(inputsPath), template = fs.readFileSync(templatePath);
mem.set(weights, wAt); mem.set(inputs, iAt);
if (poison) { mem.fill(0xAB, wsAt, wsAt + wsBytes); mem.fill(0xAB, oAt, oAt + oBytes); }
mem.set(template, oAt);
const copyIn = now() - t;
const fail = () => {
  const dv = new DataView(inst.exports.memory.buffer);
  const words = [dv.getInt32(4, true), dv.getInt32(0, true)];
  for (let k = 0; k < 12; k++) words.push(dv.getBigInt64(8 + 8 * k, true));
  console.log("model_error: " + words.join(" "));
  process.exit(5);
};
t = now();
if (inst.exports.model_run(wAt, iAt, wsAt, oAt) !== 0) fail();
const first = now() - t;
t = now();
const reference = Buffer.from(mem.subarray(oAt, oAt + oBytes));
const copyOut = now() - t;
const times = [];
for (let r = 0; r < repeat; r++) {
  t = now();
  if (inst.exports.model_run(wAt, iAt, wsAt, oAt) !== 0) fail();
  times.push(now() - t);
  if (!reference.equals(Buffer.from(mem.subarray(oAt, oAt + oBytes)))) { console.log("repeat output differs"); process.exit(6); }
}
times.sort((a, b) => a - b);
const warm = times.length ? times[Math.floor(times.length / 2)] : 0;
fs.writeFileSync(outputsPath, reference);
console.log("wasm_timing: " + JSON.stringify({ compile, instantiate, copy_in: copyIn, first_run: first, warm_run: warm, copy_out: copyOut }));
|}

let prepare_r ~dir (b : Loop_bundle.t) ~constants =
  let* w =
    Result.map_error
      (fun e -> `Generate e)
      (Err.payload (Loop_bundle_wasm.build b))
  in
  let* () = Io.unix_io (fun () -> Io.mkdir_p dir) in
  let file name = Filename.concat dir name in
  let* tensors =
    Io.bound_tensors w.Loop_bundle_wasm.weights ~lookup:constants
      ~missing:(fun id -> `Missing_constant id)
  in
  let identity = w.Loop_bundle_wasm.identity in
  let* () =
    Io.io (fun () ->
        Proc.write_file (file "model.wasm") w.Loop_bundle_wasm.bytes;
        Proc.write_file (file "runner.js") runner_text)
  in
  let* () =
    Io.write_payload w.Loop_bundle_wasm.weights ~identity
      ~path:(file "weights.bin") tensors
  in
  (* The outputs file before any run: header and zero padding. *)
  let outputs = w.Loop_bundle_wasm.outputs in
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
            Unix.ftruncate fd (Int64.to_int outputs.P.length);
            let h = P.header outputs ~identity in
            ignore (Unix.write_substring fd h 0 (String.length h))))
  in
  Ok
    {
      dir;
      module_file = file "model.wasm";
      weights = file "weights.bin";
      template = file "outputs.template";
      runner = file "runner.js";
      bundle = b;
      w;
      runs = 0;
      timings = [];
    }

let parse_failure p log =
  let lines = String.split_on_char '\n' log in
  match
    List.find_opt
      (fun l -> String.starts_with ~prefix:failure_prefix l)
      (List.rev lines)
  with
  | None -> Error (`Bad_failure_record "no model_error line")
  | Some line -> (
      let words =
        String.split_on_char ' '
          (String.sub line
             (String.length failure_prefix)
             (String.length line - String.length failure_prefix))
        |> List.filter (fun s -> s <> "")
      in
      match List.map Int64.of_string_opt words with
      | inv :: kind :: v when List.for_all Option.is_some (inv :: kind :: v)
        -> (
          let get = Option.get in
          let invocation = Int64.to_int (get inv) in
          let invocations = p.bundle.Loop_bundle.invocations in
          if invocation < 0 || invocation >= List.length invocations then
            Error (`Bad_failure_record "invocation out of range")
          else
            let inv_p = List.nth invocations invocation in
            let sites = Loop_js_failure.sites inv_p.Loop_bundle.program in
            match
              Loop_c_failure.decode ~sites
                ~kind:(Int64.to_int (get kind))
                ~v:(Array.of_list (List.map get v))
            with
            | Ok row -> Error (`Inference_failed (invocation, row))
            | Error m -> Error (`Bad_failure_record m))
      | _ -> Error (`Bad_failure_record line))

let timing_of log =
  match
    List.find_opt
      (fun l -> String.starts_with ~prefix:timing_prefix l)
      (String.split_on_char '\n' log)
  with
  | None -> []
  | Some line ->
      (* A flat JSON object of numbers: split it by hand. *)
      let body =
        String.sub line
          (String.length timing_prefix)
          (String.length line - String.length timing_prefix)
      in
      let body = String.trim body in
      let body = String.sub body 1 (String.length body - 2) in
      List.filter_map
        (fun kv ->
          match String.split_on_char ':' kv with
          | [ k; v ] -> (
              let k = String.trim k in
              let k = String.sub k 1 (String.length k - 2) in
              match float_of_string_opt (String.trim v) with
              | Some f -> Some (k, f)
              | None -> None)
          | _ -> None)
        (String.split_on_char ',' body)

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
        Io.bound_tensors p.w.Loop_bundle_wasm.inputs ~lookup:bind
          ~missing:(fun id -> `Missing_input id)
      in
      let* () =
        Io.write_payload p.w.Loop_bundle_wasm.inputs
          ~identity:p.w.Loop_bundle_wasm.identity ~path:inputs tensors
      in
      let pl = p.w.Loop_bundle_wasm.placement in
      let ws = p.w.Loop_bundle_wasm.workspace in
      let argv =
        !node
        @ [
            p.runner;
            p.module_file;
            p.weights;
            inputs;
            p.template;
            outputs;
            string_of_int pl.Loop_bundle_wasm.Placement.weights;
            string_of_int pl.Loop_bundle_wasm.Placement.inputs;
            string_of_int pl.Loop_bundle_wasm.Placement.workspace;
            string_of_int pl.Loop_bundle_wasm.Placement.outputs;
            Int64.to_string (C_workspace_plan.bytes ws);
            Int64.to_string p.w.Loop_bundle_wasm.outputs.P.length;
          ]
        @ (if poison then [ "--poison" ] else [])
        @ if repeat > 0 then [ Printf.sprintf "--repeat=%d" repeat ] else []
      in
      match Proc.run argv with
      | Error m -> Error (`Node_unavailable m)
      | Ok (Proc.Exited 0, log) ->
          p.timings <- timing_of log;
          Io.read_payload p.w.Loop_bundle_wasm.outputs
            ~identity:p.w.Loop_bundle_wasm.identity ~path:outputs
      | Ok (Proc.Exited n, log) when n = exit_inference -> parse_failure p log
      | Ok (st, log) -> Error (`Run_failed (st, log)))

let lift r = Err.import Fun.id r
let prepare ~dir b ~constants = lift (prepare_r ~dir b ~constants)
let run ?poison ?repeat p ~bind = lift (run_r ?poison ?repeat p ~bind)
let timings p = p.timings
let bundle_wasm p = p.w
let directory p = p.dir
let module_path p = p.module_file
let weights_path p = p.weights
