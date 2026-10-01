module Role = struct
  type t = Inputs | Outputs | Weights

  let code = function Weights -> 1 | Inputs -> 2 | Outputs -> 3

  let to_string = function
    | Inputs -> "inputs"
    | Outputs -> "outputs"
    | Weights -> "weights"
end

module Entry = struct
  type t = {
    id : Tensor_id.t;
    sg : Tensor_sig.t;
    offset : int64;
    bytes : int64;
  }
end

type t = { role : Role.t; entries : Entry.t list; length : int64 }
type error = [ `Payload_overflow of Role.t * Tensor_id.t ]

let pp_error ppf : [< error ] -> unit = function
  | `Payload_overflow (role, id) ->
      Format.fprintf ppf "%s payload: t%d does not fit the layout's bounds"
        (Role.to_string role) (Tensor_id.to_int id)

let header_bytes = 64L
let alignment = 64L

(* A file's size and every offset stay below this: far above any real model,
   far below where a sum of two of them could wrap. *)
let ceiling = Int64.shift_left 1L 60

let align_up n =
  Int64.mul (Int64.div (Int64.add n (Int64.pred alignment)) alignment) alignment

let tensor_bytes (sg : Tensor_sig.t) =
  let cell = Int64.of_int (Payload.packed_cell_bytes sg.Tensor_sig.fmt) in
  match
    Vec6.numel_bounded ~limit:(Int64.shift_left 1L 40) sg.Tensor_sig.shape
  with
  | Error _ -> None
  | Ok n ->
      (* [n] is below 2^40 and [cell] at most 8. *)
      Some (Int64.mul n cell)

let create role tensors =
  let open Err.Syntax in
  let+ entries, length =
    Err.List.fold_left
      (fun (acc, off) (id, sg) ->
        match tensor_bytes sg with
        | None -> Err.fail (`Payload_overflow (role, id))
        | Some bytes ->
            let next = Int64.add off (align_up bytes) in
            if Int64.compare next ceiling > 0 then
              Err.fail (`Payload_overflow (role, id))
            else Err.return ({ Entry.id; sg; offset = off; bytes } :: acc, next))
      ([], header_bytes) tensors
  in
  { role; entries = List.rev entries; length }

let header t ~identity =
  if String.length identity <> 16 then
    invalid_arg "C_payload_layout.header: identity is not 16 bytes";
  let b = Bytes.make (Int64.to_int header_bytes) '\000' in
  Bytes.blit_string "MLTCPAY\000" 0 b 0 8;
  Bytes.set_int32_le b 8 1l;
  Bytes.set_int32_le b 12 (Int32.of_int (Role.code t.role));
  Bytes.set_int64_le b 16 t.length;
  Bytes.set_int64_le b 24 alignment;
  Bytes.blit_string identity 0 b 32 16;
  Bytes.to_string b
