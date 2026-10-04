open Graph_ir
open Loop_ir
module P = C_payload_layout

module type LOADER = sig
  type loaded
  type error

  val pp_error : Format.formatter -> error -> unit
  val load : host_symbols:string list -> string -> (loaded, error) result

  val call :
    loaded ->
    io:(char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t ->
    (int64, error) result

  val close : loaded -> unit
end

type error =
  [ `Generate of Loop_bundle_c.error
  | `Load of string
  | `Call of string
  | `Missing_constant of Tensor_id.t
  | `Missing_input of Tensor_id.t
  | `Binding_mismatch of Kernel_eval.Binding_mismatch.t
  | `Io of string
  | `Run_failed of int
  | `Inference_failed of int * Loop_interp.error
  | `Bad_failure_record of string
  | `Bad_output of string
  | `Closed ]

let pp_error ppf : [< error ] -> unit = function
  | `Generate e -> Loop_bundle_c.pp_error ppf e
  | `Load m -> Fmt.pf ppf "the unit could not be loaded: %s" m
  | `Call m -> Fmt.pf ppf "the model could not be called: %s" m
  | `Missing_constant id ->
      Fmt.pf ppf "no value for constant %a" Tensor_id.pp id
  | `Missing_input id -> Fmt.pf ppf "no value for input %a" Tensor_id.pp id
  | `Binding_mismatch m -> Kernel_eval.Binding_mismatch.pp ppf m
  | `Io m -> Fmt.pf ppf "I/O: %s" m
  | `Run_failed rc -> Fmt.pf ppf "model_run returned %d" rc
  | `Inference_failed (i, e) ->
      Fmt.pf ppf "inference failed in invocation %d: %a" i Loop_interp.pp_error
        e
  | `Bad_failure_record m -> Fmt.pf ppf "unreadable failure record: %s" m
  | `Bad_output m -> Fmt.pf ppf "output rejected: %s" m
  | `Closed -> Fmt.string ppf "the context is closed"

let ( let* ) = Result.bind
let lift r = Err.import Fun.id r
let round_up n a = (n + a - 1) / a * a
let error_bytes = 128

(* Byte offsets of the five parts in a context's region, each a multiple of 64
   and of the workspace alignment, which a page-aligned mapping then honours. *)
type layout = {
  weights : int;
  inputs : int;
  outputs : int;
  workspace : int;
  failure : int;
  total : int;
}

let page = 4096

let layout (c : Loop_bundle_c.t) =
  let align =
    max 64 (Int64.to_int (C_workspace_plan.alignment c.Loop_bundle_c.workspace))
  in
  if align > page then None
  else
    let next at len = round_up (at + len) align in
    let weights = 0 in
    let inputs = next weights (Int64.to_int c.Loop_bundle_c.weights.P.length) in
    let outputs = next inputs (Int64.to_int c.Loop_bundle_c.inputs.P.length) in
    let workspace =
      next outputs (Int64.to_int c.Loop_bundle_c.outputs.P.length)
    in
    let failure =
      next workspace
        (Int64.to_int (C_workspace_plan.bytes c.Loop_bundle_c.workspace))
    in
    Some
      {
        weights;
        inputs;
        outputs;
        workspace;
        failure;
        total = failure + error_bytes;
      }

let adapter l =
  String.concat "\n"
    [
      "long entry(void *io) {";
      "  unsigned char *b = (unsigned char *)io;";
      Printf.sprintf "  memcpy(b + %d, model_outputs_header, 64);" l.outputs;
      Printf.sprintf
        "  return model_run(b + %d, b + %d, b + %d, b + %d, (struct \
         model_error *)(b + %d));"
        l.weights l.inputs l.workspace l.outputs l.failure;
      "}";
      "";
    ]

module Make (L : LOADER) = struct
  type prepared = {
    c : Loop_bundle_c.t;
    bundle : Loop_bundle.t;
    layout : layout;
    text : string;
    loaded : L.loaded;
  }

  type context = {
    p : prepared;
    fd : Unix.file_descr;
    mem :
      (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t;
    mutable open_ : bool;
  }

  let render e = Format.asprintf "%a" L.pp_error e

  let prepare_r ?numerics (bundle : Loop_bundle.t) =
    let* c =
      Result.map_error
        (fun e -> `Generate e)
        (Err.payload
           (Loop_bundle_c.build ~dialect:Loop_c_dialect.Compcert_scalar
              ?numerics bundle))
    in
    match layout c with
    | None -> Error (`Io "workspace alignment above a page")
    | Some layout ->
        let text = c.Loop_bundle_c.source ^ "\n" ^ adapter layout in
        let* loaded =
          Result.map_error
            (fun e -> `Load (render e))
            (L.load ~host_symbols:Loop_c_dialect.host_symbols text)
        in
        Ok { c; bundle; layout; text; loaded }

  let prepare ?numerics bundle = lift (prepare_r ?numerics bundle)

  let context_r p ~constants =
    let c = p.c in
    let* tensors =
      Loop_c_exec.Payload_io.bound_tensors c.Loop_bundle_c.weights
        ~lookup:constants ~missing:(fun id -> `Missing_constant id)
    in
    let* fd, mem =
      Loop_c_exec.Payload_io.unix_io (fun () ->
          let path = Filename.temp_file "loop_c_embed" ".ctx" in
          let fd = Unix.openfile path [ Unix.O_RDWR ] 0o600 in
          Sys.remove path;
          Unix.ftruncate fd p.layout.total;
          ( fd,
            Bigarray.array1_of_genarray
              (Unix.map_file fd Bigarray.char Bigarray.c_layout true
                 [| p.layout.total |]) ))
    in
    let* () =
      Loop_c_exec.Payload_io.write_region c.Loop_bundle_c.weights
        ~identity:c.Loop_bundle_c.identity fd ~base:p.layout.weights tensors
    in
    Ok { p; fd; mem; open_ = true }

  let context p ~constants = lift (context_r p ~constants)

  (* The failure record the model left: int32 kind, int32 invocation, then the
     words. *)
  let failure ctx =
    let at = ctx.p.layout.failure in
    let byte i = Char.code (Bigarray.Array1.get ctx.mem (at + i)) in
    let i32 o =
      byte o
      lor (byte (o + 1) lsl 8)
      lor (byte (o + 2) lsl 16)
      lor (byte (o + 3) lsl 24)
    in
    let i64 o =
      let r = ref 0L in
      for k = 7 downto 0 do
        r := Int64.logor (Int64.shift_left !r 8) (Int64.of_int (byte (o + k)))
      done;
      !r
    in
    let kind = i32 0 and invocation = i32 4 in
    let v =
      Array.init Loop_c_runtime.error_words (fun i -> i64 (8 + (8 * i)))
    in
    let invocations = ctx.p.bundle.Loop_bundle.invocations in
    if invocation < 0 || invocation >= List.length invocations then
      Error (`Bad_failure_record "invocation out of range")
    else
      let inv = List.nth invocations invocation in
      match
        Loop_c_failure.decode
          ~sites:(Loop_js_failure.sites inv.Loop_bundle.program)
          ~kind ~v
      with
      | Ok row -> Error (`Inference_failed (invocation, row))
      | Error m -> Error (`Bad_failure_record m)

  let run_r ?(poison = false) ctx ~bind =
    if not ctx.open_ then Error `Closed
    else
      let p = ctx.p and c = ctx.p.c in
      let* tensors =
        Loop_c_exec.Payload_io.bound_tensors c.Loop_bundle_c.inputs ~lookup:bind
          ~missing:(fun id -> `Missing_input id)
      in
      let* () =
        Loop_c_exec.Payload_io.write_region c.Loop_bundle_c.inputs
          ~identity:c.Loop_bundle_c.identity ctx.fd ~base:p.layout.inputs
          tensors
      in
      if poison then
        Bigarray.Array1.fill
          (Bigarray.Array1.sub ctx.mem p.layout.workspace
             (Int64.to_int (C_workspace_plan.bytes c.Loop_bundle_c.workspace)))
          '\xa5';
      match L.call p.loaded ~io:ctx.mem with
      | Error e -> Error (`Call (render e))
      | Ok 0L ->
          Loop_c_exec.Payload_io.read_region c.Loop_bundle_c.outputs
            ~identity:c.Loop_bundle_c.identity ctx.fd ~base:p.layout.outputs
      | Ok 1L -> failure ctx
      | Ok rc -> Error (`Run_failed (Int64.to_int rc))

  let run ?poison ctx ~bind = lift (run_r ?poison ctx ~bind)

  let close_context ctx =
    if ctx.open_ then begin
      ctx.open_ <- false;
      Unix.close ctx.fd
    end

  let close p = L.close p.loaded
  let source p = p.text
  let bundle_c p = p.c
end
