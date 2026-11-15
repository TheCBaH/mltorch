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

(* Each event's running multiplicity, for an interpreter's observation. *)
module Counter : sig
  type event := t
  type t

  val create : unit -> t
  val add : t -> event -> int64 -> unit

  val totals : t -> (event * int64) list
  (** Every event, in [all] order. *)
end = struct
  type t = int64 array

  let slot = function
    | Emitter -> 0
    | Key -> 1
    | Local -> 2
    | Reduction -> 3
    | Scan -> 4
    | Scan_update -> 5

  let create () = Array.make 6 0L
  let add t e n = t.(slot e) <- Int64.add t.(slot e) n
  let totals t = List.map (fun e -> (e, t.(slot e))) all
end
