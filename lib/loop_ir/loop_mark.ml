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
