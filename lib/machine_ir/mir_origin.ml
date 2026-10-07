(* Where an instruction came from. One CFG operation may expand into several
   instructions; each keeps the operation's location, its role in the expansion
   and a clone instance. Unknown stays unknown: no location is invented. *)

module Cfg_site = struct
  (* The CFG block id and the operation's position in its body; [instr] is
     [-1] for the block's terminator. *)
  type t = { block : int; instr : int }

  let equal (a : t) b = a = b
end

module Role = struct
  type t =
    | Address  (** byte-offset and pointer formation *)
    | Compute  (** the operation's own value *)
    | Decode  (** storage-format decode after a raw load *)
    | Encode  (** storage-format encode before a raw store *)
    | Guard  (** a source-failure test and its branch *)
    | Payload  (** a failure payload value *)
    | Transfer  (** an edge copy or block argument *)

  let name = function
    | Address -> "address"
    | Compute -> "compute"
    | Decode -> "decode"
    | Encode -> "encode"
    | Guard -> "guard"
    | Payload -> "payload"
    | Transfer -> "transfer"
end

type t = {
  cfg : Cfg_site.t option;
  output : Expr.Source.t option;  (** the output the SSA origin names *)
  role : Role.t;
  clone : int;
}

let unknown = { cfg = None; output = None; role = Role.Compute; clone = 0 }

let pp fmt t =
  match t.cfg with
  | None -> Fmt.string fmt "?"
  | Some { Cfg_site.block; instr } ->
      Fmt.pf fmt "cfg bb%d.%d %s" block instr (Role.name t.role)
