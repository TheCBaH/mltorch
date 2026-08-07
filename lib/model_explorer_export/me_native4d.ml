(* The Native4D instantiation of [Me_build]. See the .mli. *)

include Me_build.Make (struct
  type op = Native4d.Op.t

  let op_name = Native4d.Op.name
  let operands = Native4d.Op.operands
  let pp_op = Native4d.Op.pp
end)
