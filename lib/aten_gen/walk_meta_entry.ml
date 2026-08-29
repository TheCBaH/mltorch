(* Hand-crafted meta augmenting native_functions.yaml for the random-walk
   generator (Aten_walk_gen). For ops whose argument shapes are correlated in
   ways the yaml can't express (conv's weight shape, a pool window, a reduction
   over dims), this names the family recipe and supplies the initial config, the
   per-axis candidate sets, and the build body that fills the generated
   Op_<name> record. The generator wraps each entry in the [Walk.Op] skeleton
   (cfg/cascade/pp come from the recipe) and checks [target] resolves to a real
   generated spec, so the meta can't drift from the bindings. *)

type t = {
  module_name : string;  (** emitted walk module name, e.g. "Conv2d_walk" *)
  target : string;  (** op target; must match a generated Op_*.spec *)
  recipe : string;
      (** recipe module under Aten_walk_recipes, e.g. "Recipe_conv" *)
  initial : string;  (** OCaml expr: the initial cfg *)
  axes : string;  (** OCaml expr: the axes list *)
  build : string;
      (** OCaml expr: build body, with [pcg], [c] and an open Aten_walk_recipes
          in scope, returning [Op_spec.t * Pcg.t] *)
}

(* The [t] record used by every op entry below and in the family files this
   was split into (walk_meta_conv.ml, _pool.ml, _reduce.ml, _pointwise.ml,
   _shape.ml, _linalg.ml, _norm.ml, _attention.ml). *)
