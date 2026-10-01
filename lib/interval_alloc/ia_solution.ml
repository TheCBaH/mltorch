(* Offsets in allocation order plus the pool they were packed into. Anyone can
   build one ([Unsafe.make]): it is only a claim until [check] accepts it. *)

open Core.Storage_units

type 'k t = { placements : ('k * Byte_offset.t) list; pool : Byte_size.t }

let placements t = t.placements
let pool t = t.pool

module Unsafe = struct
  let make ~pool placements = { placements; pool }
end
