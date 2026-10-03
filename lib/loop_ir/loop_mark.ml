(* A semantically empty statement that names a unit of Region work, so an
   interpreter can count it the way [Region_execution.counters] does and a test
   can assert that a local runs once per key. Backends emit nothing for it. *)
type t = Emitter | Key | Local | Reduction | Scan | Scan_update

let name = function
  | Emitter -> "emitter"
  | Key -> "key"
  | Local -> "local"
  | Reduction -> "reduction"
  | Scan -> "scan"
  | Scan_update -> "scan_update"

(* Closed and alphabetical: the position is the counter's slot in a counting
   Wasm build, so a host reads them back in this order. *)
let all = [ Emitter; Key; Local; Reduction; Scan; Scan_update ]

let index = function
  | Emitter -> 0
  | Key -> 1
  | Local -> 2
  | Reduction -> 3
  | Scan -> 4
  | Scan_update -> 5
