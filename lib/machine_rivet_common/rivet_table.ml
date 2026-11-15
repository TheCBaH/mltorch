(* The regions an image takes from its caller. In the table binding every
   region but a constant is addressed through x18, which the entry sets to the
   caller's table of base addresses: slot [k] holds the base of the [k]th
   region. The image is then code and constants only, immutable and shareable;
   tensors and invocation-private scratch belong to the context that calls it. *)

open Machine_ir
module Art = Machine_model.Mir_artifact

type slot = {
  symbol : string;
  region : Mir_id.Region.t;
  size : int64;
  align : int64;
  section : Art.Section.t;  (** [Bound] or [Bss]: where its bytes come from *)
}

(* The artifact's mutable data symbols in region order: the slot index is the
   position. *)
let of_artifact artifact =
  List.filter_map
    (fun (s : Art.Symbol.t) ->
      match s.Art.Symbol.kind with
      | Art.Symbol.Data { region; size; align; section } -> (
          match section with
          | Art.Section.Bound | Art.Section.Bss ->
              Some { symbol = s.Art.Symbol.name; region; size; align; section }
          | Art.Section.Rodata _ -> None)
      | Art.Symbol.External_function _ | Art.Symbol.Function _
      | Art.Symbol.Generated_attention _ | Art.Symbol.Generated_function _ ->
          None)
    (Art.symbols artifact)
  |> List.sort (fun a b ->
      Int.compare
        (Mir_id.Region.to_int a.region)
        (Mir_id.Region.to_int b.region))

let slot_of_symbol slots name : Core.Table_slot.t option =
  let rec go k = function
    | [] -> None
    | s :: rest ->
        if String.equal s.symbol name then Some (Core.Table_slot.of_int k)
        else go (k + 1) rest
  in
  go 0 slots

(* The entry every table-bound image exports. *)
let entry = "mir_entry"

module Byte_size = Core.Storage_units.Byte_size

(* A slot holds one 64-bit address. *)
let word =
  Err.or_raise ~pp_error:Core.Storage_units.pp_error (Byte_size.of_int64 8L)

(* Bytes of the table a caller supplies. *)
let bytes slots =
  List.fold_left
    (fun acc _ ->
      Err.or_raise ~pp_error:Core.Storage_units.pp_error
        (Byte_size.add acc word))
    Byte_size.zero slots
