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
  bundle : Loop_bundle.t;
  w : Loop_bundle_wasm.t;
  mutable runs : int;
  mutable timings : (string * float) list;
}

let node = Wasm_node.node
let ( let* ) = Result.bind

let prepare_r ?vector ?numerics ?kernel ~dir (b : Loop_bundle.t) ~constants =
  let* w =
    Result.map_error
      (fun e -> `Generate e)
      (Err.payload (Loop_bundle_wasm.build ?vector ?numerics ?kernel b))
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
        ())
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
      bundle = b;
      w;
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
        Io.bound_tensors p.w.Loop_bundle_wasm.inputs ~lookup:bind
          ~missing:(fun id -> `Missing_input id)
      in
      let* () =
        Io.write_payload p.w.Loop_bundle_wasm.inputs
          ~identity:p.w.Loop_bundle_wasm.identity ~path:inputs tensors
      in
      let pl = p.w.Loop_bundle_wasm.placement in
      let ws = p.w.Loop_bundle_wasm.workspace in
      match
        Wasm_node.execute
          ~features:(Wasm_features.of_module p.w.Loop_bundle_wasm.module_)
          ~dir:p.dir ~module_file:p.module_file ~weights:p.weights ~inputs
          ~template:p.template ~outputs
          {
            Wasm_node.weights = pl.Loop_bundle_wasm.Placement.weights;
            inputs = pl.Loop_bundle_wasm.Placement.inputs;
            workspace = pl.Loop_bundle_wasm.Placement.workspace;
            outputs = pl.Loop_bundle_wasm.Placement.outputs;
            workspace_bytes = C_workspace_plan.bytes ws;
            outputs_bytes = p.w.Loop_bundle_wasm.outputs.P.length;
          }
          ~poison ~repeat
      with
      | Error m -> Error (`Node_unavailable m)
      | Ok (Proc.Exited 0, log) ->
          p.timings <- Wasm_node.timing_of log;
          Io.read_payload p.w.Loop_bundle_wasm.outputs
            ~identity:p.w.Loop_bundle_wasm.identity ~path:outputs
      | Ok (Proc.Exited n, log) when n = Wasm_node.exit_inference ->
          Wasm_node.parse_failure p.bundle log
      | Ok (st, log) -> Error (`Run_failed (st, log)))

let lift r = Err.import Fun.id r

let prepare ?vector ?numerics ?kernel ~dir b ~constants =
  lift (prepare_r ?vector ?numerics ?kernel ~dir b ~constants)

let run ?poison ?repeat p ~bind = lift (run_r ?poison ?repeat p ~bind)
let timings p = p.timings
let bundle_wasm p = p.w
let directory p = p.dir
let module_path p = p.module_file
let weights_path p = p.weights
