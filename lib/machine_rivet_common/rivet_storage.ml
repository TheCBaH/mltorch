(* The caller's storage under a table-bound image: how many bytes a tensor's
   cells occupy, and a zero-filled block for a region no tensor binds. *)

module U = Core.Storage_units

(* A value that cannot overflow here: a tensor's cells already fit in memory. *)
let sure r = Err.or_raise ~pp_error:U.pp_error r

let tensor_bytes (Tensor.Tensor t) : U.Byte_size.t =
  let p = t.Tensor.payload in
  let cells =
    sure
      (U.Element_count.of_int64
         (Int64.of_int (Bigarray.Array1.dim p.Payload.data)))
  in
  let cell =
    sure
      (U.Element_bytes.of_int64
         (Int64.of_int (Payload.cell_bytes p.Payload.fmt)))
  in
  sure (U.Element_count.to_bytes cells cell)

(* One byte at least: a Bigarray is never empty here, so its address is. *)
let bytes (size : U.Byte_size.t) =
  let n = Int64.to_int (U.Byte_size.to_int64 size) in
  let b = Bigarray.Array1.create Bigarray.char Bigarray.c_layout (max n 1) in
  Bigarray.Array1.fill b '\000';
  b

(* The bytes of a table region. *)
let region_bytes (s : Rivet_table.slot) : U.Byte_size.t =
  sure (U.Byte_size.of_int64 s.Rivet_table.size)

(* Slot [k] of a table holds the address [v], little-endian: eight bytes. *)
let set_word table (k : Core.Table_slot.t) v =
  for b = 0 to 7 do
    table.{(8 * (k :> int)) + b} <-
      Char.chr
        (Int64.to_int
           (Int64.logand (Int64.shift_right_logical v (8 * b)) 0xFFL))
  done
