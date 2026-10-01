(* Offsets in allocation order plus the pool they were packed into. Anyone can
   build one ([Unsafe.make]): it is only a claim until [check] accepts it. *)

type 'k t = { placements : ('k * int64) list; pool : int64 }

let placements t = t.placements
let pool t = t.pool

module Unsafe = struct
  let make ~pool placements = { placements; pool }
end
