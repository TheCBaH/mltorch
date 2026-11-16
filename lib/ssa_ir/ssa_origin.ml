(* The logical value an operation came from: the buffer the value stores to and
   the iteration domain it ran over, the extents of its output in axis order. *)
module Value = struct
  type t = {
    buffer : Ssa_id.Buffer.t;
    domain : Core.Dim.extent Core.Dim.t list;
  }

  let equal a b =
    Ssa_id.Buffer.equal a.buffer b.buffer
    && List.equal
         (fun x y -> Core.Dim.to_int x = Core.Dim.to_int y)
         a.domain b.domain

  let pp fmt v =
    Fmt.pf fmt "%a [%a]" Ssa_id.Buffer.pp v.buffer
      Fmt.(list ~sep:(any "x") (using Core.Dim.to_int int))
      v.domain
end

(* Where an operation came from. A pass keeps a stable site when it moves an
   operation and gives a clone its own instance. [Output] names the buffer an
   operation stores to; [Value] the logical value of the placed kernel whose
   lowering made it and its iteration domain, which a pass keeps for what it
   moves or rewrites and a clone also gets. *)
type t = Output of Ssa_id.Buffer.t | Unknown | Value of Value.t

let equal a b =
  match (a, b) with
  | Output a, Output b -> Ssa_id.Buffer.equal a b
  | Unknown, Unknown -> true
  | Value a, Value b -> Value.equal a b
  | (Output _ | Unknown | Value _), _ -> false

let pp fmt = function
  | Output b -> Fmt.pf fmt "out %a" Ssa_id.Buffer.pp b
  | Unknown -> Fmt.string fmt "?"
  | Value v -> Fmt.pf fmt "value %a" Value.pp v
