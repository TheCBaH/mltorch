(* Native as a [Side.S].

   ITS OWN UNIT, not a submodule of [Native_dialect]. After stage 4, [Snapshot]
   is specialized AGAINST [Native_dialect], so a [Native_dialect.Side] naming
   [Snapshot] would close a [Native_dialect] <-> [Snapshot] compilation-unit
   cycle. This sits above both and depends on both, which also keeps
   [Native_dialect] free of any framework dependency. *)

type op = Graph_ir.op

module Dialect = Native_dialect
module Snapshot = Snapshot
module Transfer = Output_transfer

let symbolic s = Eval_symbolic.run (Snapshot.graph s)
let sig_of s id = Graph_view.sig_of (Snapshot.view s) id
let group_root s = (Snapshot.graph s).Graph_common.Graph.root
