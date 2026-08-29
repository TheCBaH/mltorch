(* A unit of logical work. Primitive load counts are diagnostics of a
   transformed program; marks are the work a transformation must preserve. *)
type t = Emitter | Key | Local | Reduction | Scan | Scan_update

let all = [ Emitter; Key; Local; Reduction; Scan; Scan_update ]

let name = function
  | Emitter -> "emitter"
  | Key -> "key"
  | Local -> "local"
  | Reduction -> "reduction"
  | Scan -> "scan"
  | Scan_update -> "scan_update"
