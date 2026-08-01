(* The four axes Native4D tensors occupy, as their own type.

   This is the enforcement .ai/native4d_design.md §4.3 asks for, and it is
   stronger than the validation it asks for. "T=D=1 alone is insufficient" —
   reducing D is numerically an identity today, but an operation that NAMES D
   makes the dialect depend on a non-Native4D axis. A check can catch that; a
   type stops it being written. [Mean_keepdims {dims : t list}] and
   [Permute4 {perm : (t * t) list}] cannot name T or D at all, so "a four-axis
   permutation does not use T or D as semantic axes" is a property of the
   representation rather than an obligation on a validator someone might forget
   to call.

   Constructors in global alphabetical order, per the repo rule, which is why
   this reads C H N W rather than N H W C. [all] is in FRAME order — the order
   the axes occupy in [Vec6] — because every consumer that iterates axes cares
   about the frame, not the spelling. *)

type t = C | H | N | W

let all = [ N; H; W; C ]
let to_axis = function C -> Axis.C | H -> Axis.H | N -> Axis.N | W -> Axis.W

(* [None] for T and D: the whole point. A caller converting a Native axis has to
   say what it does with the two the dialect has no name for. *)
let of_axis : Axis.t -> t option = function
  | Axis.C -> Some C
  | Axis.H -> Some H
  | Axis.N -> Some N
  | Axis.W -> Some W
  | Axis.D | Axis.T -> None

let equal a b = a = b
let compare a b = Stdlib.compare (to_axis a) (to_axis b)
let to_string a = Axis.to_string (to_axis a)
let pp fmt a = Axis.pp fmt (to_axis a)

let jsont : t Jsont.t =
  Jsont.map ~kind:"axis4"
    ~dec:(fun json ->
      let axis = Json_util.dec Axis.jsont json in
      match of_axis axis with
      | Some a -> a
      | None ->
          Jsont.Error.msgf Jsont.Meta.none "axis %a is outside the dialect"
            Axis.pp axis)
    ~enc:(fun a -> Json_util.enc Axis.jsont (to_axis a))
    Jsont.json
