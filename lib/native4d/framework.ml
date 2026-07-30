(* Each shared functor applied exactly ONCE, at one stable path.

   Stages 5 onward all need to name the same snapshot and view types. Applying
   the functors at each use site would work — OCaml's named functors are
   applicative, so [Snapshot.Make (Dialect4)] denotes the same types wherever it
   is written — but it would scatter the ownership of "which specialization is
   THE Native4D one" across the library. This module owns it. *)

module View4 = Graph_view.Make (Dialect4)
module Snapshot4 = Snapshot.Make (Dialect4)

(* The [Side.S] bundle: what a cross-dialect pair functor consumes. This is
   where [Output_transfer4] is actually USED — built beside the snapshot and
   never passed, its table would be dead and claim closure would silently keep
   using Native's, which is the wrong table for a Native4D destination. *)
module Side4 = struct
  type op = Op.t

  module Dialect = Dialect4
  module Snapshot = Snapshot4
  module Transfer = Output_transfer4.Transfer

  let symbolic s = Eval_symbolic4.run (Snapshot4.graph s)
  let sig_of s id = View4.sig_of (Snapshot4.view s) id
  let group_root s = (Snapshot4.graph s).Graph_common.Graph.root
end

(* TWO map instances, because the two directions are different PAIRS — not
   because of type identity, which applicativity already settles. Stage 5 lowers
   Native to Native4D and uses [Map_from_native]; stage 7 rewrites Native4D to
   itself and uses [Map4]. *)
module Map_from_native = Graph_map.Make_pair (Native_side) (Side4)
module Map4 = Graph_map.Make_pair (Side4) (Side4)

(* The verifiers, in the same two directions. [Verify_from_native] checks the
   conversion map; [Verify4] is what a same-dialect Native4D rewrite will use.
   Claim closure inside each uses the DESTINATION's transfer table, which is why
   [Side4.Transfer] had to exist. *)
module Verify_from_native = Map_verify.Make_pair (Native_side) (Side4)
module Verify4 = Map_verify.Make_pair (Side4) (Side4)

(* The same-dialect transform framework, at this dialect. Stage 7's point: a
   Native4D rewrite runs through the machinery Native's passes run through, with
   no second implementation of recipes, planning, id discipline or claim
   propagation. *)
module Rewrite4 = Rewrite.Make (Side4)
module Recipe4 = Recipe.Make (Side4)
module Pattern4 = Pattern.Make (Dialect4)
module Region4 = Region.Make (Dialect4)
module Pass4 = Pass.Make (Side4)
