(* A harness slot from byte counts written as plain integers. *)

module H = Machine_rivet_riscv32.Rv32_harness
module U = Core.Storage_units

let sure r = Err.or_raise ~pp_error:U.pp_error r

let make ~size ~align placement =
  {
    H.size = sure (U.Byte_size.of_int64 (Int64.of_int size));
    align = sure (U.Byte_alignment.of_int64 (Int64.of_int align));
    placement;
  }
