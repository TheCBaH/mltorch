(* A typed module through Rivet's lowering, relaxation and layout to a laid-out
   image, and a laid-out image into this process. The image is the primary
   native artifact: no assembly text is produced on this route. *)

module P = Driver.Pipeline.Make (Aarch64)

module Error = struct
  type t =
    [ `Load of Native_exec.error
    | `Rivet_lower of string
    | `Rivet_plan of string ]

  let pp fmt : t -> unit = function
    | `Load e -> Native_exec.pp_error fmt e
    | `Rivet_lower m | `Rivet_plan m -> Fmt.string fmt m
end

type laid_out = Image.laid_out

let render e = Foundation.Diag.render e

let plan ~entry modules : (laid_out, Error.t) Err.t =
  let rec lower acc = function
    | [] -> Ok (List.rev acc)
    | m :: rest -> (
        match P.lower ~state:Aarch64.default_state m with
        | Ok l -> lower (l :: acc) rest
        | Error e -> Err.fail (`Rivet_lower (render e)))
  in
  Err.bind (lower [] modules) (fun lowered ->
      match
        match lowered with
        | [ one ] -> P.plan ~entry one
        | many -> P.plan_many ~entry many
      with
      | Ok laid -> Err.return laid
      | Error e -> Err.fail (`Rivet_plan (render e)))

type t = Native_exec.loaded

let load laid : (t, Error.t) Err.t =
  match Native_exec.load ~target:"aarch64" laid with
  | Ok t -> Err.return t
  | Error e -> Err.fail (`Load e)

let close = Native_exec.close
let default_io = Bigarray.Array1.create Bigarray.char Bigarray.c_layout 16

(* Runs the entry once, with [io] as the entry's one pointer argument; the value
   is its integer result register. *)
let call ?(io = default_io) t : (int64, Error.t) Err.t =
  match Native_exec.call t ~io with
  | Ok v -> Err.return v
  | Error e -> Err.fail (`Load e)

let read_global t name : (string, Error.t) Err.t =
  match Native_exec.read_global t name with
  | Ok s -> Err.return s
  | Error e -> Err.fail (`Load e)

let write_global t name bytes : (unit, Error.t) Err.t =
  match Native_exec.write_global t name bytes with
  | Ok () -> Err.return ()
  | Error e -> Err.fail (`Load e)

let host_symbol = Native_exec.host_symbol

(* The address of a buffer's data, which does not move while it is alive. *)
let address = Native_exec.io_address
