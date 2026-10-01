(* See result_lease.mli. *)

open Core.Storage_units

module Generation = struct
  type t = int64

  let equal = Int64.equal
  let pp ppf t = Format.fprintf ppf "run%Ld" t
end

type t = {
  arena : Arena.t option;
  constants : Constant_arena.t;
  generation : Generation.t;
  signatures : Tensor_sig.t Tensor_id.Map.t;
  outputs : Tensor.packed Tensor_id.Map.t;
  mutable released : bool;
}

let make ~arena ~constants ~generation ~signatures outputs =
  { arena; constants; generation; signatures; outputs; released = false }

let first_generation = 0L
let next_generation = Int64.succ
let generation t = t.generation
let constants t = t.constants
let released t = t.released

let with_outputs t f =
  if t.released then Err.fail ~pos:__POS__ `Lease_released
  else Err.return (f t.outputs)

let release t =
  if not t.released then begin
    t.released <- true;
    Option.iter Arena.release t.arena
  end

(* A result's payload bytes: a declared signature is bounded by construction. *)
let payload_bytes (sg : Tensor_sig.t) =
  match Err.payload (Alloc_script.alloc ~released:Tensor_id.Set.empty sg) with
  | Ok a -> a.Alloc_script.Alloc.bytes
  | Error _ -> Byte_size.zero

let pp_lease_released ppf `Lease_released =
  Format.pp_print_string ppf "arena: the lease is released"

let bytes t =
  Tensor_id.Map.fold
    (fun id _ acc ->
      match Tensor_id.Map.find_opt id t.signatures with
      | None -> acc
      | Some sg -> Err.or_raise ~pp_error (Byte_size.add acc (payload_bytes sg)))
    t.outputs Byte_size.zero

let copy_out t =
  let open Err.Syntax in
  if t.released then Err.fail ~pos:__POS__ `Lease_released
  else
    let* copies =
      Err.List.map
        (fun (id, (src : Tensor.packed)) ->
          match Tensor_id.Map.find_opt id t.signatures with
          | None -> Err.return (id, src)
          | Some sg ->
              let* dst = Tensor.create_of_sig sg in
              let+ () = Tensor.blit_into src dst in
              (id, dst))
        (Tensor_id.Map.bindings t.outputs)
    in
    let copied =
      {
        Arena.Copies.count = Int64.of_int (List.length copies);
        bytes = bytes t;
      }
    in
    release t;
    Err.return (Tensor_id.Map.of_seq (List.to_seq copies), copied)
