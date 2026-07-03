(* Core types shared by the random-walk recipes and the walk runner.

   Pure: depends only on [aten_spec] (the functional PCG + spec types), so the
   generated walk modules and the recipes stay off the ATen C++ build, exactly
   as [aten_op_spec] is kept separate from [aten_spec_run]. The runner that links
   ATen ([Op_walk.run]) consumes a [(module Op)] from here. *)

module Pcg = Aten_spec.Pcg
module Sv = Aten_spec.Scalar_value
module Tspec = Aten_spec.Tensor_spec

type 'cfg axis = { name : string; mutate : Pcg.t -> 'cfg -> 'cfg * Pcg.t }
(** A configurable dimension of an op's parameter space. [mutate] picks a new
    value freely (one [Pcg.next] step); cross-parameter validity is restored
    afterwards by the op's [cascade], not by filtering here. *)

(** A walkable op: everything the runner needs, bundled as a module so the
    interface is a single signature rather than a bag of labelled functions. *)
module type Op = sig
  type cfg

  val target : string
  val initial : cfg
  val axes : cfg axis list
  val cascade : cfg -> cfg
  val build : Pcg.t -> cfg -> Aten_spec.Op_spec.t * Pcg.t
  val pp : Format.formatter -> cfg -> unit
end

(* Pick one element from a non-empty list using one [Pcg.next] step. *)
let pick pcg lst =
  let n, pcg = Pcg.next pcg in
  (List.nth lst (n mod List.length lst), pcg)

(* A field axis: pick a candidate value with the PCG and set it via [setter].
   No validity filtering -- the engine applies [cascade] after the mutation. *)
let field_axis name candidates setter =
  {
    name;
    mutate =
      (fun pcg cfg ->
        let v, pcg = pick pcg candidates in
        (setter cfg v, pcg));
  }

(* Draw [n] f32-canonical uniform(-1,1) values, threading the PCG. *)
let gen_values pcg n =
  let rec go i pcg acc =
    if i = 0 then (List.rev acc, pcg)
    else
      let v, pcg = Pcg.uniform ~low:(-1.) ~high:1. pcg in
      go (i - 1) pcg (Sv.Float v :: acc)
  in
  go n pcg []

(* A dtype=F32, Values-source tensor spec of the given shape. The PCG advances
   one step per element; [Values] means the runner's own PCG is never touched. *)
let tensor_spec pcg shape =
  let n = List.fold_left ( * ) 1 shape in
  let vals, pcg = gen_values pcg n in
  ({ Tspec.dtype = Aten_spec.Dtype.F32; shape; source = Tspec.Values vals }, pcg)
