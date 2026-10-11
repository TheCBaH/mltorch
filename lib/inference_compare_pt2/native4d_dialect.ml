(* Native4D dialect tooling: extraction as a sub-graph of the four-axis graph.
   The twin of {!Native_dialect} over the four-axis op. *)

include Graph_dialect.Make (struct
  type op = Native4d.Op.t

  let ops = Graph_model.native4d_ops
end)
