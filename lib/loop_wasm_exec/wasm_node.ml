open Loop_ir
module Proc = Loop_c_exec.Proc

let node = ref [ "node" ]
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
// A C-compiled module may import WASI functions it never calls on the inference
// path; any that is called is a defect, so it throws.
const wasi = new Proxy({}, { get: (_, name) => () => { throw new Error("unexpected WASI call: " + String(name)); } });
const inst = new WebAssembly.Instance(mod, { math: { exp: Math.exp, log: Math.log, sin: Math.sin, cos: Math.cos }, wasi_snapshot_preview1: wasi });
const instantiate = now() - t;
const mem = new Uint8Array(inst.exports.memory.buffer);
t = now();
const weights = fs.readFileSync(weightsPath), inputs = fs.readFileSync(inputsPath), template = fs.readFileSync(templatePath);
mem.set(weights, wAt); mem.set(inputs, iAt);
if (poison) { mem.fill(0xAB, wsAt, wsAt + wsBytes); mem.fill(0xAB, oAt, oAt + oBytes); }
mem.set(template, oAt);
const copyIn = now() - t;
const entry = inst.exports.model_run || inst.exports.run;
const errAt = inst.exports.error_ptr ? inst.exports.error_ptr() : 0;
const fail = () => {
  const dv = new DataView(inst.exports.memory.buffer, errAt);
  const words = [dv.getInt32(4, true), dv.getInt32(0, true)];
  for (let k = 0; k < 12; k++) words.push(dv.getBigInt64(8 + 8 * k, true));
  console.log("model_error: " + words.join(" "));
  process.exit(5);
};
t = now();
if (entry(wAt, iAt, wsAt, oAt) !== 0) fail();
const first = now() - t;
t = now();
const reference = Buffer.from(mem.subarray(oAt, oAt + oBytes));
const copyOut = now() - t;
const times = [];
for (let r = 0; r < repeat; r++) {
  t = now();
  if (entry(wAt, iAt, wsAt, oAt) !== 0) fail();
  times.push(now() - t);
  if (!reference.equals(Buffer.from(mem.subarray(oAt, oAt + oBytes)))) { console.log("repeat output differs"); process.exit(6); }
}
times.sort((a, b) => a - b);
const warm = times.length ? times[Math.floor(times.length / 2)] : 0;
fs.writeFileSync(outputsPath, reference);
console.log("wasm_timing: " + JSON.stringify({ compile, instantiate, copy_in: copyIn, first_run: first, warm_run: warm, copy_out: copyOut }));
|}

(* The invocation record's failing position and the interpreter's row for its
   failure, from a runner's [model_error:] line. *)
let parse_failure (bundle : Loop_bundle.t) log =
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
          let invocations = bundle.Loop_bundle.invocations in
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

type placement = {
  weights : int;
  inputs : int;
  workspace : int;
  outputs : int;
  workspace_bytes : int64;
  outputs_bytes : int64;
}

(* Runs the runner on prepared files; the result is the process status and its
   log, for the caller to classify. *)
let execute ~dir ~module_file ~weights ~inputs ~template ~outputs
    (pl : placement) ~poison ~repeat =
  let runner = Filename.concat dir "runner.js" in
  if not (Sys.file_exists runner) then Proc.write_file runner runner_text;
  Proc.run
    (!node
    @ [
        runner;
        module_file;
        weights;
        inputs;
        template;
        outputs;
        string_of_int pl.weights;
        string_of_int pl.inputs;
        string_of_int pl.workspace;
        string_of_int pl.outputs;
        Int64.to_string pl.workspace_bytes;
        Int64.to_string pl.outputs_bytes;
      ]
    @ (if poison then [ "--poison" ] else [])
    @ if repeat > 0 then [ Printf.sprintf "--repeat=%d" repeat ] else [])
