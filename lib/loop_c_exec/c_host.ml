open Graph_ir
open Loop_ir
module P = C_payload_layout

type error =
  [ `Generate of Loop_bundle_c.error
  | `Compiler_unavailable of string
  | `Compile_failed of C_proc.status * string
  | `Missing_constant of Tensor_id.t
  | `Missing_input of Tensor_id.t
  | `Binding_mismatch of Kernel_eval.Binding_mismatch.t
  | `Quant_missing of Tensor_id.t
  | `Io of string
  | `Run_failed of C_proc.status * string
  | `Inference_failed of int * Loop_interp.error
  | `Bad_failure_record of string
  | `Bad_output of string ]

let pp_error ppf : [< error ] -> unit = function
  | `Generate e -> Loop_bundle_c.pp_error ppf e
  | `Compiler_unavailable m -> Fmt.pf ppf "C compiler unavailable: %s" m
  | `Compile_failed (st, log) ->
      Fmt.pf ppf "C compilation failed (%a):@.%s" C_proc.pp_status st log
  | `Missing_constant id ->
      Fmt.pf ppf "no value for constant %a" Tensor_id.pp id
  | `Missing_input id -> Fmt.pf ppf "no value for input %a" Tensor_id.pp id
  | `Binding_mismatch m -> Kernel_eval.Binding_mismatch.pp ppf m
  | `Quant_missing id ->
      Fmt.pf ppf "%a: quantized output without parameters" Tensor_id.pp id
  | `Io m -> Fmt.pf ppf "I/O: %s" m
  | `Run_failed (st, log) ->
      Fmt.pf ppf "the model binary failed (%a):@.%s" C_proc.pp_status st log
  | `Inference_failed (i, e) ->
      Fmt.pf ppf "inference failed in invocation %d: %a" i Loop_interp.pp_error
        e
  | `Bad_failure_record m -> Fmt.pf ppf "unreadable failure record: %s" m
  | `Bad_output m -> Fmt.pf ppf "output file rejected: %s" m

type prepared = {
  dir : string;
  exe : string;
  weights : string;
  bundle : Loop_bundle.t;
  c : Loop_bundle_c.t;
  compiler : string list;
  compiler_identity : string;
  mutable runs : int;
}

let base_compiler =
  [
    "gcc";
    "-std=c11";
    "-O2";
    "-ffp-contract=off";
    "-fno-strict-aliasing";
    "-Wall";
    "-Wextra";
    "-Werror";
  ]

(* [LOOP_C_CFLAGS] replaces the optimisation flag, so the same suites run at
   [-O0] or under the sanitizers without a second copy of them. *)
let default_compiler =
  match Sys.getenv_opt "LOOP_C_CFLAGS" with
  | None | Some "" -> base_compiler
  | Some flags ->
      List.filter (fun f -> f <> "-O2") base_compiler
      @ List.filter (fun f -> f <> "") (String.split_on_char ' ' flags)

let ( let* ) = Result.bind
let io f = try Ok (f ()) with Sys_error m | Failure m -> Error (`Io m)

let unix_io f =
  try Ok (f ()) with
  | Unix.Unix_error (e, fn, arg) ->
      Error (`Io (Printf.sprintf "%s(%s): %s" fn arg (Unix.error_message e)))
  | Sys_error m -> Error (`Io m)

let mkdir_p dir =
  let rec go d =
    if not (Sys.file_exists d) then (
      go (Filename.dirname d);
      Unix.mkdir d 0o755)
  in
  go dir

(* A payload file: the header, zero padding, and each entry's tensor at its
   offset. The file is created whole (zero-filled) and the tensors written into
   it, so padding is zero by construction. *)
let write_payload (layout : P.t) ~identity ~path tensors =
  unix_io (fun () ->
      let fd =
        Unix.openfile path [ Unix.O_RDWR; Unix.O_CREAT; Unix.O_TRUNC ] 0o644
      in
      Fun.protect
        ~finally:(fun () -> Unix.close fd)
        (fun () ->
          Unix.ftruncate fd (Int64.to_int layout.P.length);
          let header = P.header layout ~identity in
          let (_ : int) =
            Unix.write_substring fd header 0 (String.length header)
          in
          List.iter2
            (fun (e : P.Entry.t) t ->
              C_blob.copy `In fd (Int64.to_int e.P.Entry.offset) t)
            layout.P.entries tensors))

let bound_tensors (layout : P.t) ~lookup ~missing =
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

let compiler_banner compiler =
  match C_proc.run [ List.hd compiler; "--version" ] with
  | Ok (C_proc.Exited 0, text) -> (
      match String.index_opt text '\n' with
      | Some i -> String.sub text 0 i
      | None -> text)
  | _ -> "unknown"

let prepare_r ?(compiler = default_compiler) ~dir (b : Loop_bundle.t) ~constants
    =
  let* c =
    Result.map_error
      (fun e -> `Generate e)
      (Err.payload (Loop_bundle_c.build b))
  in
  let* () = unix_io (fun () -> mkdir_p dir) in
  let file name = Filename.concat dir name in
  let* tensors =
    bound_tensors c.Loop_bundle_c.weights ~lookup:constants ~missing:(fun id ->
        `Missing_constant id)
  in
  let* () =
    io (fun () ->
        C_proc.write_file (file "model_infer.c") c.Loop_bundle_c.source;
        C_proc.write_file (file "model_main.c") C_main_gen.source)
  in
  let weights = file "weights.bin" in
  let* () =
    write_payload c.Loop_bundle_c.weights ~identity:c.Loop_bundle_c.identity
      ~path:weights tensors
  in
  let exe = file "model" in
  let banner = compiler_banner compiler in
  let* () =
    match
      C_proc.run
        (compiler
        @ [ file "model_main.c"; file "model_infer.c"; "-o"; exe; "-lm" ])
    with
    | Error m -> Error (`Compiler_unavailable m)
    | Ok (C_proc.Exited 0, _) -> Ok ()
    | Ok (st, log) ->
        io (fun () -> C_proc.write_file (file "compile.log") log) |> ignore;
        Error (`Compile_failed (st, log))
  in
  Ok
    {
      dir;
      exe;
      weights;
      bundle = b;
      c;
      compiler;
      compiler_identity = banner;
      runs = 0;
    }

let write_inputs_r p ~bind ~path =
  let* tensors =
    bound_tensors p.c.Loop_bundle_c.inputs ~lookup:bind ~missing:(fun id ->
        `Missing_input id)
  in
  write_payload p.c.Loop_bundle_c.inputs ~identity:p.c.Loop_bundle_c.identity
    ~path tensors

let read_outputs_r p ~path =
  let layout = p.c.Loop_bundle_c.outputs in
  let expected = P.header layout ~identity:p.c.Loop_bundle_c.identity in
  let* size =
    match (Unix.stat path).Unix.st_size with
    | n -> Ok n
    | exception Unix.Unix_error _ -> Error (`Bad_output ("cannot stat " ^ path))
  in
  if Int64.of_int size <> layout.P.length then
    Error
      (`Bad_output
         (Printf.sprintf "%d bytes, expected %Ld" size layout.P.length))
  else
    let* head =
      io (fun () ->
          let ic = open_in_bin path in
          Fun.protect
            ~finally:(fun () -> close_in_noerr ic)
            (fun () -> really_input_string ic 64))
    in
    if head <> expected then Error (`Bad_output "header mismatch")
    else
      unix_io (fun () ->
          let fd = Unix.openfile path [ Unix.O_RDONLY ] 0 in
          Fun.protect
            ~finally:(fun () -> Unix.close fd)
            (fun () ->
              List.map
                (fun (e : P.Entry.t) ->
                  match Err.payload (Tensor.create_of_sig e.P.Entry.sg) with
                  | Error (`Quant_missing _) ->
                      raise (Failure "quantized output")
                  | Ok t ->
                      C_blob.copy `Out fd (Int64.to_int e.P.Entry.offset) t;
                      t)
                layout.P.entries))

let command p ~inputs ~outputs =
  [ p.exe; "--weights"; p.weights; "--inputs"; inputs; "--outputs"; outputs ]

(* The last stderr line of an inference failure: prefix, invocation, kind, and
   the record's words. *)
let parse_failure p log =
  let prefix = C_model_abi.error_line_prefix in
  let lines = String.split_on_char '\n' log in
  match
    List.find_opt (fun l -> String.starts_with ~prefix l) (List.rev lines)
  with
  | None -> Error (`Bad_failure_record "no model_error line")
  | Some line -> (
      let words =
        String.split_on_char ' '
          (String.sub line (String.length prefix)
             (String.length line - String.length prefix))
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

let run_r ?(poison = false) p ~bind =
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
      let* () = write_inputs_r p ~bind ~path:inputs in
      match
        C_proc.run
          (command p ~inputs ~outputs @ if poison then [ "--poison" ] else [])
      with
      | Error m -> Error (`Io m)
      | Ok (C_proc.Exited 0, _) -> read_outputs_r p ~path:outputs
      | Ok (C_proc.Exited n, log) when n = C_model_abi.exit_inference ->
          parse_failure p log
      | Ok (st, log) -> Error (`Run_failed (st, log)))

let directory p = p.dir
let executable p = p.exe
let weights_path p = p.weights
let bundle_c p = p.c
let compiler_identity p = p.compiler_identity
let lift r = Err.import Fun.id r

let prepare ?compiler ~dir b ~constants =
  lift (prepare_r ?compiler ~dir b ~constants)

let write_inputs p ~bind ~path = lift (write_inputs_r p ~bind ~path)
let read_outputs p ~path = lift (read_outputs_r p ~path)
let run ?poison p ~bind = lift (run_r ?poison p ~bind)
let compiler p = p.compiler
