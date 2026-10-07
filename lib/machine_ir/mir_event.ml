(* A logical observable: one unit of the work a lowering must preserve, as the
   SSA IR's marks count it. An event carries its multiplicity (a vector
   iteration counts each lane it covers). Primitive load or instruction counts
   measure a transformed implementation and are never one of these. *)
type t = Emitter | Key | Local | Reduction | Scan | Scan_update

let all = [ Emitter; Key; Local; Reduction; Scan; Scan_update ]

let name = function
  | Emitter -> "emitter"
  | Key -> "key"
  | Local -> "local"
  | Reduction -> "reduction"
  | Scan -> "scan"
  | Scan_update -> "scan_update"

let equal (a : t) b = a = b
